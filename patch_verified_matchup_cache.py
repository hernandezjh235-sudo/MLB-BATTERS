from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_MATCHUP_CACHE_V1_2026_08_26"

text = APP.read_text(encoding="utf-8").replace("\r\n", "\n")
if MARKER in text:
    py_compile.compile(str(APP), doraise=True)
    print("Verified matchup cache patch already present.")
    raise SystemExit(0)


def replace_final_function(src, name, alias, wrapper_src):
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not nodes:
        raise RuntimeError(f"Could not find {name}")
    node = nodes[-1]
    lines = src.splitlines(keepends=True)
    old_fn = "".join(lines[node.lineno - 1:node.end_lineno])
    legacy_fn = re.sub(
        rf"^(\s*)def\s+{re.escape(name)}\s*\(",
        rf"\1def {alias}(",
        old_fn,
        count=1,
        flags=re.M,
    )
    if legacy_fn == old_fn:
        raise RuntimeError(f"Could not safely rename {name}")
    replacement = legacy_fn.rstrip() + "\n\n" + wrapper_src.strip() + "\n"
    return "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])


pitcher_wrapper = r'''
@st.cache_data(ttl=21600, show_spinner=False)
def get_statcast_pitch_profile(pitcher_id, days=365):
    """Current exact-pitcher Statcast first; verified persistent last-good only on live failure."""
    payload = {}
    try:
        payload = get_statcast_pitch_profile_live_v1(pitcher_id, days=days) or {}
    except Exception:
        payload = {}
    if isinstance(payload, dict) and payload.get("available") and (payload.get("pitch_type_profile") or payload.get("pitch_mix")):
        _ow_matchup_cache_save("pitcher_pitch", pitcher_id, payload, days=days)
        payload = dict(payload)
        payload["_OW Data State"] = "LIVE_CURRENT_STATCAST"
        payload["_OW Data Source"] = "BASEBALL_SAVANT_STATCAST_SEARCH"
        return payload
    cached = _ow_matchup_cache_load("pitcher_pitch", pitcher_id, max_age_hours=24*21)
    if cached:
        cached["_OW Data State"] = "LAST_GOOD_CURRENT_SEASON_FALLBACK"
        cached["_OW Data Source"] = "PERSISTENT_VERIFIED_MATCHUP_CACHE"
        return cached
    return payload
'''

batter_wrapper = r'''
@st.cache_data(ttl=21600, show_spinner=False)
def get_batter_statcast_pitch_type_profile(batter_id, days=365, pitcher_hand=None):
    """Current exact-batter pitch-type Statcast first; current-season last-good fallback only."""
    payload = {}
    try:
        payload = get_batter_statcast_pitch_type_profile_live_v1(batter_id, days=days, pitcher_hand=pitcher_hand) or {}
    except Exception:
        payload = {}
    cache_key = f"{batter_id}_{str(pitcher_hand or 'ALL').upper()}"
    if isinstance(payload, dict) and payload.get("available") and payload.get("pitch_type_profile"):
        _ow_matchup_cache_save("batter_pitch", cache_key, payload, days=days)
        payload = dict(payload)
        payload["_OW Data State"] = "LIVE_CURRENT_STATCAST"
        payload["_OW Data Source"] = "BASEBALL_SAVANT_STATCAST_SEARCH"
        return payload
    cached = _ow_matchup_cache_load("batter_pitch", cache_key, max_age_hours=24*21)
    if cached:
        cached["_OW Data State"] = "LAST_GOOD_CURRENT_SEASON_FALLBACK"
        cached["_OW Data Source"] = "PERSISTENT_VERIFIED_MATCHUP_CACHE"
        return cached
    return payload
'''

bullpen_wrapper = r'''
@st.cache_data(ttl=900, show_spinner=False)
def get_recent_team_bullpen_usage(team_id, as_of_date, lookback_days=3):
    """Live MLB schedule/boxscore bullpen workload with a short verified persistent fallback."""
    payload = {}
    try:
        payload = get_recent_team_bullpen_usage_live_v1(team_id, as_of_date, lookback_days=lookback_days) or {}
    except Exception:
        payload = {}
    cache_key = f"{team_id}_{as_of_date}_{int(lookback_days)}"
    label = str((payload or {}).get("label") or (payload or {}).get("status") or "").upper()
    note = str((payload or {}).get("message") or "").lower()
    usable = isinstance(payload, dict) and payload and not ("unavailable" in note or "error" in note)
    if usable:
        _ow_matchup_cache_save("bullpen", cache_key, payload, days=lookback_days)
        payload = dict(payload)
        payload["_OW Data State"] = "LIVE_CURRENT_MLB_BOXSCORES"
        payload["_OW Data Source"] = "MLB_STATS_API"
        return payload
    cached = _ow_matchup_cache_load("bullpen", cache_key, max_age_hours=36)
    if cached:
        cached["_OW Data State"] = "LAST_GOOD_RECENT_FALLBACK"
        cached["_OW Data Source"] = "PERSISTENT_VERIFIED_MATCHUP_CACHE"
        return cached
    return payload
'''

# Replace lower functions first so source coordinates remain valid for earlier functions.
for name, alias, wrapper in [
    ("get_batter_statcast_pitch_type_profile", "get_batter_statcast_pitch_type_profile_live_v1", batter_wrapper),
    ("get_recent_team_bullpen_usage", "get_recent_team_bullpen_usage_live_v1", bullpen_wrapper),
    ("get_statcast_pitch_profile", "get_statcast_pitch_profile_live_v1", pitcher_wrapper),
]:
    text = replace_final_function(text, name, alias, wrapper)

# Shared cache helpers are inserted before the earliest renamed function/decorator.
tree = ast.parse(text)
alias_names = {
    "get_statcast_pitch_profile_live_v1",
    "get_batter_statcast_pitch_type_profile_live_v1",
    "get_recent_team_bullpen_usage_live_v1",
}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in alias_names]
if len(nodes) != 3:
    raise RuntimeError("Not all live source aliases were created")
insert_node = min(nodes, key=lambda n: min([d.lineno for d in n.decorator_list] + [n.lineno]))
insert_line = min([d.lineno for d in insert_node.decorator_list] + [insert_node.lineno])

helpers = r'''# VERIFIED_MATCHUP_CACHE_V1_2026_08_26
# Data-only persistence/provenance. Projection formulas and UI are untouched.
def _ow_matchup_cache_root():
    root = Path(STORAGE_DIR) / "verified_matchup_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ow_matchup_cache_json_default(obj):
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def _ow_matchup_cache_save(kind, key, payload, days=None):
    try:
        from datetime import datetime as _ow_dt, timezone as _ow_tz
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:140]
        folder = _ow_matchup_cache_root() / str(kind)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        doc = {
            "schema_version": 1,
            "season": int(_ow_dt.now(_ow_tz.utc).year),
            "fetched_at_utc": _ow_dt.now(_ow_tz.utc).isoformat(),
            "days": days,
            "payload": payload,
        }
        tmp.write_text(json.dumps(doc, default=_ow_matchup_cache_json_default), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _ow_matchup_cache_load(kind, key, max_age_hours):
    try:
        from datetime import datetime as _ow_dt, timezone as _ow_tz
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:140]
        path = _ow_matchup_cache_root() / str(kind) / f"{safe}.json"
        if not path.exists():
            return {}
        doc = json.loads(path.read_text(encoding="utf-8"))
        if int(doc.get("season", 0) or 0) != int(_ow_dt.now(_ow_tz.utc).year):
            return {}
        stamp = _ow_dt.fromisoformat(str(doc.get("fetched_at_utc") or "").replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_ow_tz.utc)
        age = (_ow_dt.now(_ow_tz.utc) - stamp).total_seconds() / 3600.0
        if age < 0 or age > float(max_age_hours):
            return {}
        payload = doc.get("payload") or {}
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}

'''

lines = text.splitlines(keepends=True)
text = "".join(lines[:insert_line - 1]) + helpers + "".join(lines[insert_line - 1:])
ast.parse(text)
APP.write_text(text, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Applied verified live/current pitch-type, arsenal and bullpen caches; formulas/UI unchanged.")
