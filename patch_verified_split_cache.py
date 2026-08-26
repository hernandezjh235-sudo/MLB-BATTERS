from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_SPLIT_CACHE_V1_2026_08_26"

text = APP.read_text(encoding="utf-8").replace("\r\n", "\n")
if MARKER in text:
    py_compile.compile(str(APP), doraise=True)
    print("Verified platoon split cache patch already present.")
    raise SystemExit(0)


def replace_final_function(src, name, alias, wrapper_src):
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
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


key_stats_wrapper = r'''
@st.cache_data(ttl=1800, show_spinner=False)
def _ow_batter_key_matchup_stats_context(player_id, pitcher_hand):
    payload = {}
    try:
        payload = _ow_batter_key_matchup_stats_context_live_v1(player_id, pitcher_hand) or {}
    except Exception:
        payload = {}
    hand = str(pitcher_hand or "").upper()[:1]
    usable = isinstance(payload, dict) and payload and payload.get("Split PA") not in (None, "", "—") and hand in {"R", "L"}
    key = f"{player_id}_{hand}"
    if usable:
        _ow_split_cache_save("batter_key_stats", key, payload)
        payload = dict(payload)
        payload["_OW Split Data State"] = "LIVE_CURRENT_MLB_STATS_API"
        payload["_OW Split Match Method"] = "MLBAM_ID_EXACT"
        return payload
    cached = _ow_split_cache_load("batter_key_stats", key, max_age_hours=24*14)
    if cached:
        cached["_OW Split Data State"] = "LAST_GOOD_CURRENT_SEASON_FALLBACK"
        cached["_OW Split Match Method"] = "MLBAM_ID_EXACT"
        return cached
    return payload
'''

pitcher_wrapper = r'''
@st.cache_data(ttl=1800, show_spinner=False)
def _ow_pitcher_allowed_split_context(pitcher_id, batter_hand=None, market="HRR"):
    payload = {}
    try:
        payload = _ow_pitcher_allowed_split_context_live_v1(pitcher_id, batter_hand, market) or {}
    except Exception:
        payload = {}
    hand = str(batter_hand or "").upper()[:1]
    usable = isinstance(payload, dict) and payload and payload.get("Pitcher Split BF") not in (None, "", "—") and hand in {"R", "L"}
    key = f"{pitcher_id}_{hand}_{str(market or 'HRR').upper()}"
    if usable:
        _ow_split_cache_save("pitcher_allowed_split", key, payload)
        payload = dict(payload)
        payload["_OW Split Data State"] = "LIVE_CURRENT_MLB_STATS_API"
        payload["_OW Split Match Method"] = "MLBAM_ID_EXACT"
        return payload
    cached = _ow_split_cache_load("pitcher_allowed_split", key, max_age_hours=24*14)
    if cached:
        cached["_OW Split Data State"] = "LAST_GOOD_CURRENT_SEASON_FALLBACK"
        cached["_OW Split Match Method"] = "MLBAM_ID_EXACT"
        return cached
    return payload
'''

factor_wrapper = r'''
@st.cache_data(ttl=1800, show_spinner=False)
def _ow_batter_split_factor(player_id, pitcher_hand, market):
    payload = None
    try:
        payload = _ow_batter_split_factor_live_v1(player_id, pitcher_hand, market)
    except Exception:
        payload = None
    hand = str(pitcher_hand or "").upper()[:1]
    note = str(payload[1] if isinstance(payload, (list, tuple)) and len(payload) > 1 else "").lower()
    usable = isinstance(payload, (list, tuple)) and len(payload) >= 2 and hand in {"R", "L"} and "unavailable" not in note and "error" not in note
    key = f"{player_id}_{hand}_{str(market or 'HRR').upper()}"
    if usable:
        _ow_split_cache_save("batter_split_factor", key, {"factor": payload[0], "note": payload[1]})
        return float(payload[0]), str(payload[1]) + " | MLBAM LIVE CURRENT"
    cached = _ow_split_cache_load("batter_split_factor", key, max_age_hours=24*14)
    if cached and cached.get("factor") is not None:
        return float(cached.get("factor")), str(cached.get("note") or "Split fallback") + " | LAST GOOD CURRENT SEASON"
    return payload if isinstance(payload, (list, tuple)) and len(payload) >= 2 else (1.0, "Split unavailable")
'''

for name, alias, wrapper in [
    ("_ow_batter_key_matchup_stats_context", "_ow_batter_key_matchup_stats_context_live_v1", key_stats_wrapper),
    ("_ow_pitcher_allowed_split_context", "_ow_pitcher_allowed_split_context_live_v1", pitcher_wrapper),
    ("_ow_batter_split_factor", "_ow_batter_split_factor_live_v1", factor_wrapper),
]:
    text = replace_final_function(text, name, alias, wrapper)

# Insert self-contained current-season persistent cache helpers before the earliest renamed split function.
tree = ast.parse(text)
alias_names = {
    "_ow_batter_key_matchup_stats_context_live_v1",
    "_ow_pitcher_allowed_split_context_live_v1",
    "_ow_batter_split_factor_live_v1",
}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in alias_names]
if len(nodes) != 3:
    raise RuntimeError("Not all split live aliases were created")
insert_node = min(nodes, key=lambda n: min([d.lineno for d in n.decorator_list] + [n.lineno]))
insert_line = min([d.lineno for d in insert_node.decorator_list] + [insert_node.lineno])

helpers = r'''# VERIFIED_SPLIT_CACHE_V1_2026_08_26
# Data-only: live MLBAM split calls remain primary; cache is current-season fallback only.
def _ow_split_cache_root():
    root = Path(STORAGE_DIR) / "verified_split_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ow_split_cache_save(kind, key, payload):
    try:
        from datetime import datetime as _ow_dt, timezone as _ow_tz
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:140]
        folder = _ow_split_cache_root() / str(kind)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        doc = {
            "schema_version": 1,
            "season": int(_ow_dt.now(_ow_tz.utc).year),
            "fetched_at_utc": _ow_dt.now(_ow_tz.utc).isoformat(),
            "payload": payload,
        }
        tmp.write_text(json.dumps(doc, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _ow_split_cache_load(kind, key, max_age_hours):
    try:
        from datetime import datetime as _ow_dt, timezone as _ow_tz
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:140]
        path = _ow_split_cache_root() / str(kind) / f"{safe}.json"
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
print("Applied live-first verified platoon split cache; formulas/UI unchanged.")
