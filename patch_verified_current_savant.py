from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_CURRENT_SAVANT_V2_2026_08_26"
TARGET = "_ow_baseball_savant_leaderboard"

text = APP.read_text(encoding="utf-8")
if MARKER in text:
    print("Verified current batter + pitcher Savant V2 patch already present.")
    py_compile.compile(str(APP), doraise=True)
    raise SystemExit(0)

tree = ast.parse(text)
nodes = [
    n for n in tree.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == TARGET
]
if not nodes:
    raise RuntimeError(f"Could not find {TARGET} in app.py")

# Patch ONLY the final top-level Savant loader. The prior implementation is
# preserved as an alias for audit/recovery, but the verified wrapper will not
# silently substitute an unverified historical table as current-season data.
node = nodes[-1]
lines = text.splitlines(keepends=True)
old_fn = "".join(lines[node.lineno - 1:node.end_lineno])
legacy_fn = re.sub(
    r"^(\s*)def\s+_ow_baseball_savant_leaderboard\s*\(",
    r"\1def _ow_baseball_savant_leaderboard_legacy_source_v2(",
    old_fn,
    count=1,
    flags=re.M,
)
if legacy_fn == old_fn:
    raise RuntimeError("Could not rename final Savant leaderboard function safely")

replacement = legacy_fn.rstrip() + "\n\n" + r'''# VERIFIED_CURRENT_SAVANT_V2_2026_08_26

def _ow_v2_norm_cols(df):
    return {
        re.sub(r"[^a-z0-9]+", "", str(c).lower()): c
        for c in getattr(df, "columns", [])
    }


def _ow_v2_current_savant_filter_season(df, season):
    # Filter a real year/season column to the requested season when present.
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        norm = _ow_v2_norm_cols(df)
        ycol = norm.get("year") or norm.get("season")
        if not ycol:
            return df
        years = pd.to_numeric(df[ycol], errors="coerce")
        if not years.notna().any():
            return df
        keep = years.astype("Int64") == int(season)
        if not keep.any():
            return pd.DataFrame()
        return df.loc[keep].copy()
    except Exception:
        return pd.DataFrame()


def _ow_v2_current_savant_frame_is_valid(df, kind="batter", season=2026, min_rows=1):
    # Reject historical/career substitutes and require a current Statcast schema.
    # DATA validation only: no HRR/HR/FS projection or grading math changes.
    try:
        if not isinstance(df, pd.DataFrame) or df.empty or len(df) < int(min_rows):
            return False

        norm = _ow_v2_norm_cols(df)
        keys = set(norm)

        historical_markers = {
            "historicalpa", "careerpa", "careerg", "careerab", "careerh",
            "careerhr", "profilesource", "startyear", "endyear", "seasons",
            "historicalgames", "careerops",
        }
        if len(keys & historical_markers) >= 2:
            return False

        # MLBAM/player_id is the canonical join key for a verified data table.
        id_col = None
        for key in ("playerid", "mlbamid", "mlbam", "keymlbam", "batterid", "pitcherid"):
            if key in norm:
                id_col = norm[key]
                break
        if id_col is None:
            return False

        ids = pd.to_numeric(df[id_col], errors="coerce")
        id_coverage = float(ids.notna().mean()) if len(ids) else 0.0
        if id_coverage < 0.80:
            return False

        identity = any(
            k in keys
            for k in ("lastnamefirstname", "playername", "name", "player", "batter", "pitcher")
        )
        if not identity:
            return False

        groups = [
            any(k in keys for k in ("xwoba", "estwoba", "estimatedwoba")),
            any(k in keys for k in ("xba", "estba", "estimatedba")),
            any(k in keys for k in ("xslg", "estslg", "estimatedslg")),
            any(k in keys for k in ("hardhitpercent", "hardhitpct", "hardhit")),
            any(k in keys for k in ("barrelbattedrate", "barrelpercent", "barrelpct", "brlpercent")),
            any(k in keys for k in ("whiffpercent", "whiffpct", "whiff")),
            any(k in keys for k in ("exvelocityavg", "exitvelocityavg", "avghitspeed", "avgev")),
            any(k in keys for k in ("kpercent", "kpct", "strikeoutpercent")),
            any(k in keys for k in ("bbpercent", "bbpct", "walkpercent")),
        ]
        if sum(bool(x) for x in groups) < 4:
            return False

        ycol = norm.get("year") or norm.get("season")
        if ycol:
            years = pd.to_numeric(df[ycol], errors="coerce")
            if years.notna().any() and not (years.astype("Int64") == int(season)).any():
                return False

        return True
    except Exception:
        return False


def _ow_v2_data_roots():
    # Prefer a Railway persistent volume, then explicit/storage/repo roots.
    roots = []
    try:
        import os as _ow_os
        vol = str(_ow_os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
        explicit = str(_ow_os.environ.get("MLB_VERIFIED_DATA_ROOT", "") or "").strip()
        if explicit:
            roots.append(Path(explicit))
        if vol:
            roots.append(Path(vol) / "verified_current_data")
    except Exception:
        pass

    try:
        if "STORAGE_DIR" in globals() and STORAGE_DIR:
            roots.append(Path(STORAGE_DIR) / "verified_current_data")
    except Exception:
        pass

    app_root = Path(__file__).resolve().parent
    roots.append(app_root / "data" / "verified_current")

    out, seen = [], set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _ow_v2_read_manifest(root, which="current"):
    try:
        import json as _ow_json
        p = Path(root) / "manifests" / f"{which}_manifest.json"
        if not p.exists():
            return {}
        obj = _ow_json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _ow_v2_manifest_age_hours(manifest):
    try:
        from datetime import datetime as _ow_dt, timezone as _ow_tz
        raw = str(manifest.get("generated_at_utc") or manifest.get("fetched_at_utc") or "").strip()
        if not raw:
            return None
        stamp = _ow_dt.fromisoformat(raw.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_ow_tz.utc)
        return max(0.0, (_ow_dt.now(_ow_tz.utc) - stamp).total_seconds() / 3600.0)
    except Exception:
        return None


def _ow_v2_dataset_manifest_ok(manifest, kind, season):
    try:
        if not manifest:
            return False
        if int(manifest.get("season", season)) != int(season):
            return False
        ds = (manifest.get("datasets") or {}).get(str(kind).lower()) or {}
        if not isinstance(ds, dict):
            return False
        if str(ds.get("status", "PASS")).upper() not in ("PASS", "VERIFIED", "OK"):
            return False
        return int(ds.get("rows", 0) or 0) >= 100
    except Exception:
        return False


def _ow_v2_local_file(kind, which, season, allow_stale=False):
    # Load only a promoted current/last_good verified dataset.
    kind = str(kind or "").lower()
    names = {
        "batter": ("savant_batter_profiles.csv", "savant_batter_stats.csv"),
        "pitcher": ("savant_pitcher_stats.csv", "savant_pitcher_profiles.csv"),
    }.get(kind, ())
    if not names:
        return pd.DataFrame()

    for root in _ow_v2_data_roots():
        manifest = _ow_v2_read_manifest(root, which)
        manifest_ok = _ow_v2_dataset_manifest_ok(manifest, kind, season)
        age_h = _ow_v2_manifest_age_hours(manifest)

        # Fresh current is preferred. Stale verified current is retried only after
        # the live route, and last_good remains the final safe fallback.
        if which == "current" and not allow_stale and age_h is not None and age_h > 36.0:
            continue

        for nm in names:
            path = Path(root) / which / nm
            try:
                if not path.exists() or not path.is_file():
                    continue
                df = pd.read_csv(path, low_memory=False)
                df = _ow_v2_current_savant_filter_season(df, season)
                if not _ow_v2_current_savant_frame_is_valid(df, kind, season, min_rows=100):
                    continue
                df = df.copy()
                state = "MANIFEST_VERIFIED" if manifest_ok else "SCHEMA_VERIFIED_UNMANIFESTED"
                df["_OW Savant URL"] = f"VERIFIED_CURRENT_SAVANT:{kind.upper()}:{which.upper()}:{path}"
                df["_OW Savant Current Verified"] = "YES"
                df["_OW Savant Data State"] = state
                df["_OW Savant Data Age Hours"] = age_h
                return df
            except Exception:
                continue
    return pd.DataFrame()


def _ow_v2_fetch_live_custom(kind="batter", season=2026):
    kind = str(kind or "").lower()
    if kind not in ("batter", "pitcher"):
        return pd.DataFrame()

    # Existing contact-quality / discipline fields only. This improves provenance
    # and freshness; it does not add or reweight projection signals.
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
        "type": kind,
        "filter": "",
        "min": "1",
        "selections": ",".join(selections),
        "chart": "false",
        "sort": "xwoba",
        "sortDir": "desc",
        "csv": "true",
    }
    try:
        r = requests.get(
            url,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        body = (r.text or "").strip()
        if r.status_code != 200 or not body or body.startswith("<") or "," not in body[:800]:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(body), low_memory=False)
        df = _ow_v2_current_savant_filter_season(df, season)
        if not _ow_v2_current_savant_frame_is_valid(df, kind, season, min_rows=100):
            return pd.DataFrame()
        df = df.copy()
        df["_OW Savant URL"] = f"VERIFIED_CURRENT_SAVANT:{kind.upper()}:LIVE_CUSTOM:{season}"
        df["_OW Savant Current Verified"] = "YES"
        df["_OW Savant Data State"] = "LIVE_VERIFIED"
        df["_OW Savant Data Age Hours"] = 0.0
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def _ow_baseball_savant_leaderboard(kind="batter", season=2026):
    # Verified-current Savant route for BOTH batters and pitchers.
    # Priority: fresh current -> live -> stale verified current -> last_good -> empty.
    kind = str(kind or "").lower()
    if kind not in ("batter", "pitcher"):
        try:
            return _ow_baseball_savant_leaderboard_legacy_source_v2(kind, season)
        except Exception:
            return pd.DataFrame()

    df = _ow_v2_local_file(kind, "current", season, allow_stale=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    df = _ow_v2_fetch_live_custom(kind, season)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    df = _ow_v2_local_file(kind, "current", season, allow_stale=True)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.copy()
        df["_OW Savant Data State"] = "STALE_CURRENT_VERIFIED_FALLBACK"
        return df

    df = _ow_v2_local_file(kind, "last_good", season, allow_stale=True)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.copy()
        df["_OW Savant Data State"] = "LAST_GOOD_VERIFIED_FALLBACK"
        return df

    # Never feed a historical/career profile into a 2026 Statcast slot.
    return pd.DataFrame()
'''.lstrip()

new_text = (
    "".join(lines[:node.lineno - 1])
    + replacement
    + "\n"
    + "".join(lines[node.end_lineno:])
)
ast.parse(new_text)
APP.write_text(new_text, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Applied verified-current batter + pitcher Savant V2 routing; formulas/UI unchanged.")
