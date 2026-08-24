from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_CURRENT_SAVANT_V1_2026_08_23"
TARGET = "_ow_baseball_savant_leaderboard"

text = APP.read_text(encoding="utf-8")
if MARKER in text:
    print("Verified current Savant patch already present.")
    py_compile.compile(str(APP), doraise=True)
    raise SystemExit(0)

tree = ast.parse(text)
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == TARGET]
if not nodes:
    raise RuntimeError(f"Could not find {TARGET} in app.py")

# Patch only the final top-level definition because that is the runtime winner in this
# large single-file app. Preserve the exact previous implementation as a legacy alias
# so pitcher behavior remains byte-for-byte equivalent.
node = nodes[-1]
lines = text.splitlines(keepends=True)
old_fn = "".join(lines[node.lineno - 1:node.end_lineno])
legacy_fn = re.sub(
    r"^(\s*)def\s+_ow_baseball_savant_leaderboard\s*\(",
    r"\1def _ow_baseball_savant_leaderboard_legacy_source(",
    old_fn,
    count=1,
    flags=re.M,
)
if legacy_fn == old_fn:
    raise RuntimeError("Could not rename final Savant leaderboard function safely")

replacement = legacy_fn.rstrip() + "\n\n" + r'''# VERIFIED_CURRENT_SAVANT_V1_2026_08_23

def _ow_current_savant_frame_is_valid(df):
    """Accept only a genuine current-Savant style batter table.

    Historical prior tables (career_*, historical_*, 2015-2024 profiles) are
    intentionally rejected even if they contain player names and ordinary AVG/SLG.
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
        keys = set(norm)
        # Explicitly reject the historical prior schema that previously shadowed
        # Baseball Savant on Railway.
        historical_markers = {
            "historicalpa", "careerpa", "careerg", "careerab", "careerh",
            "careerhr", "profile_source", "startyear", "endyear", "seasons",
        }
        if len(keys & historical_markers) >= 2:
            return False
        identity = any(k in keys for k in {
            "lastnamefirstname", "playername", "name", "player", "batter",
            "playerid", "mlbamid",
        })
        metric_groups = [
            any(k in keys for k in {"xwoba", "estwoba", "estimatedwoba"}),
            any(k in keys for k in {"xba", "estba", "estimatedba"}),
            any(k in keys for k in {"xslg", "estslg", "estimatedslg"}),
            any(k in keys for k in {"hardhitpercent", "hardhitpct", "hardhit"}),
            any(k in keys for k in {"barrelbattedrate", "barrelpercent", "barrelpct", "brlpercent"}),
            any(k in keys for k in {"whiffpercent", "whiffpct", "whiff"}),
            any(k in keys for k in {"exvelocityavg", "exitvelocityavg", "avghitspeed", "avgev"}),
            any(k in keys for k in {"kpercent", "kpct", "strikeoutpercent"}),
        ]
        return bool(identity and sum(bool(x) for x in metric_groups) >= 3)
    except Exception:
        return False


def _ow_current_savant_filter_season(df, season):
    """If a year column exists, require/filter the requested season."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
        ycol = norm.get("year") or norm.get("season")
        if not ycol:
            return df
        years = pd.to_numeric(df[ycol], errors="coerce")
        usable = years.notna()
        if not usable.any():
            return df
        keep = years.astype("Int64") == int(season)
        if not keep.any():
            return pd.DataFrame()
        return df.loc[keep].copy()
    except Exception:
        return df


@st.cache_data(ttl=21600, show_spinner=False)
def _ow_baseball_savant_leaderboard(kind="batter", season=2026):
    """Verified current-season Savant source for batters; legacy path for pitchers.

    This changes DATA ROUTING only. Existing HRR/HR/Fantasy formulas, caps, weights,
    sides, probabilities and grading logic remain untouched.
    """
    if str(kind or "").lower() != "batter":
        return _ow_baseball_savant_leaderboard_legacy_source(kind, season)

    app_root = Path(__file__).resolve().parent
    storage_root = Path(STORAGE_DIR) if "STORAGE_DIR" in globals() else app_root / "mlb_engine"
    roots = [
        storage_root / "batter_fantasy_data",
        storage_root,
        app_root / "data" / "raw",
        app_root / "data",
        app_root,
        Path.cwd() / "data" / "raw",
        Path.cwd() / "data",
        Path.cwd(),
    ]
    names = [
        "savant_batter_profiles.csv",
        "savant_batter_stats.csv",
        "savant_hitter_stats.csv",
        "savant_data.csv",
    ]
    seen = set()
    for root in roots:
        for nm in names:
            path = root / nm
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                if not path.exists() or not path.is_file():
                    continue
                df = pd.read_csv(path, low_memory=False)
                df = _ow_current_savant_filter_season(df, season)
                if not _ow_current_savant_frame_is_valid(df):
                    continue
                df = df.copy()
                df["_OW Savant URL"] = f"VERIFIED_CURRENT_SAVANT:LOCAL:{path}"
                df["_OW Savant Current Verified"] = "YES"
                return df
            except Exception:
                continue

    # Preferred live source: Baseball Savant Custom Leaderboard. min=1 is used so
    # rookies and small-sample players are not silently omitted.
    selections = [
        "pa", "hit", "home_run", "strikeout", "walk",
        "k_percent", "bb_percent", "batting_avg", "slg_percent", "on_base_percent",
        "xba", "xslg", "woba", "xwoba", "xobp", "xiso",
        "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
        "barrel_batted_rate", "hard_hit_percent", "avg_best_speed",
        "whiff_percent", "oz_swing_percent", "iz_contact_percent",
        "pull_percent", "flyballs_percent", "groundballs_percent",
    ]
    url = "https://baseballsavant.mlb.com/leaderboard/custom"
    params = {
        "year": int(season),
        "type": "batter",
        "filter": "",
        "min": "1",
        "selections": ",".join(selections),
        "chart": "false",
        "sort": "xwoba",
        "sortDir": "desc",
        "csv": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        body = (r.text or "").strip()
        if r.status_code == 200 and body and not body.startswith("<") and "," in body[:500]:
            df = pd.read_csv(io.StringIO(body), low_memory=False)
            df = _ow_current_savant_filter_season(df, season)
            if _ow_current_savant_frame_is_valid(df):
                df = df.copy()
                df["_OW Savant URL"] = f"VERIFIED_CURRENT_SAVANT:LIVE_CUSTOM:{season}"
                df["_OW Savant Current Verified"] = "YES"
                return df
    except Exception:
        pass

    # No historical substitution here. If verified current Savant is unavailable,
    # return empty and let the existing model's other current/live layers and neutral
    # Savant fallback handle it rather than feeding stale data as if it were current.
    return pd.DataFrame()
'''.lstrip()

new_text = "".join(lines[:node.lineno - 1]) + replacement + "\n" + "".join(lines[node.end_lineno:])
ast.parse(new_text)
APP.write_text(new_text, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Applied verified current Savant routing patch; production formulas unchanged.")
