#!/usr/bin/env python3
"""Static/runtime contract verifier for the 2026 MLB batter/HRR data pipeline.

Run AFTER the data-only startup patches and BEFORE Streamlit. This script does not
modify app.py or projection formulas. It writes an audit JSON into persistent
storage when possible.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
SEASON = int(os.getenv("MLB_SEASON", "2026"))

REQUIRED_MARKERS = [
    "VERIFIED_CURRENT_SAVANT_V2_2026_08_26",
    "OW_RAILWAY_PERSISTENCE_V1_2026_08_26",
    "VERIFIED_MLBAM_MATCHING_V1_2026_08_26",
    "VERIFIED_SPLIT_CACHE_V1_2026_08_26",
    "VERIFIED_MATCHUP_CACHE_V1_2026_08_26",
]


def fail(msg: str) -> None:
    raise AssertionError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def storage_dir() -> Path:
    explicit = str(os.getenv("MLB_STORAGE_DIR", "") or "").strip()
    volume = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
    if explicit:
        return Path(explicit)
    if volume:
        return Path(volume) / "mlb_engine"
    return ROOT / "mlb_engine"


def verified_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = str(os.getenv("MLB_VERIFIED_DATA_ROOT", "") or "").strip()
    volume = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
    if explicit:
        roots.append(Path(explicit))
    if volume:
        roots.append(Path(volume) / "verified_current_data")
    roots.append(storage_dir() / "verified_current_data")
    roots.append(ROOT / "data" / "verified_current")
    out, seen = [], set()
    for p in roots:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def read_manifest(root: Path, state: str = "current") -> dict:
    p = root / "manifests" / f"{state}_manifest.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalized_cols(df: pd.DataFrame) -> set[str]:
    return {re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in df.columns}


def verify_promoted_dataset(root: Path, manifest: dict, kind: str, min_rows: int = 100) -> dict:
    ds = (manifest.get("datasets") or {}).get(kind) or {}
    if not ds:
        return {"status": "NOT_PROMOTED"}
    if str(ds.get("status", "")).upper() not in {"PASS", "VERIFIED", "OK"}:
        fail(f"{kind}: manifest status is not PASS")
    rel = str(ds.get("path") or "")
    if not rel:
        fail(f"{kind}: manifest path missing")
    path = root / rel
    if not path.exists():
        fail(f"{kind}: promoted file missing: {path}")
    expected = str(ds.get("sha256") or "").lower().strip()
    if expected and sha256_file(path) != expected:
        fail(f"{kind}: SHA-256 mismatch")
    df = pd.read_csv(path, low_memory=False)
    if len(df) < min_rows:
        fail(f"{kind}: row count below {min_rows}")
    cols = normalized_cols(df)
    if not any(c in cols for c in {"playerid", "mlbamid", "mlbam", "keymlbam", "batterid", "pitcherid"}):
        fail(f"{kind}: MLBAM/player ID missing")
    if "season" in cols:
        scol = next(c for c in df.columns if re.sub(r"[^a-z0-9]+", "", str(c).lower()) == "season")
        yrs = pd.to_numeric(df[scol], errors="coerce")
        if yrs.notna().any() and not (yrs.astype("Int64") == SEASON).all():
            fail(f"{kind}: non-{SEASON} rows present")
    return {"status": "PASS", "rows": int(len(df)), "path": str(path)}


def function_source(tree: ast.AST, text: str, name: str) -> str:
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not nodes:
        return ""
    node = nodes[-1]
    lines = text.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def main() -> int:
    py_compile.compile(str(APP), doraise=True)
    text = APP.read_text(encoding="utf-8")
    tree = ast.parse(text)
    checks: dict[str, object] = {}

    # 1/2/11 current batter + pitcher verified routing.
    for marker in REQUIRED_MARKERS:
        if marker not in text:
            fail(f"startup data marker missing: {marker}")
    checks["startup_markers"] = "PASS"

    savant_src = function_source(tree, text, "_ow_baseball_savant_leaderboard")
    for token in (
        '_ow_v2_local_file(kind, "current"',
        "_ow_v2_fetch_live_custom(kind, season)",
        '_ow_v2_local_file(kind, "last_good"',
        "STALE_CURRENT_VERIFIED_FALLBACK",
        "LAST_GOOD_VERIFIED_FALLBACK",
    ):
        if token not in savant_src:
            fail(f"verified Savant route missing: {token}")
    checks["current_batter_pitcher_routing"] = "PASS"

    # 3 historical tables cannot be current-Savant substitutes.
    verified_block = "\n".join([
        function_source(tree, text, "_ow_v2_local_file"),
        function_source(tree, text, "_ow_v2_fetch_live_custom"),
        savant_src,
    ])
    for forbidden in ("cleaned_batting_stats.csv", '"batter_profiles.csv"'):
        if forbidden in verified_block:
            fail(f"historical file leaked into verified current route: {forbidden}")
    checks["historical_current_separation"] = "PASS"

    # 4 MLBAM-first batter Savant matching.
    match_src = function_source(tree, text, "_ow_savant_player_row")
    if "MLBAM_ID_EXACT" not in match_src or "NORMALIZED_NAME_EXACT" not in match_src:
        fail("MLBAM-first Savant matcher incomplete")
    if "fuzzy" in match_src.lower() and "No loose fuzzy" not in match_src:
        fail("unexpected fuzzy current-Savant matching")
    hctx_src = function_source(tree, text, "_ow_baseball_savant_hitter_context")
    if "player_id=None" not in hctx_src or "player_id=player_id" not in hctx_src:
        fail("Savant hitter context is not carrying MLBAM ID")
    checks["mlbam_first_matching"] = "PASS"

    # 5/7 verifier/validation structure.
    valid_src = function_source(tree, text, "_ow_v2_current_savant_frame_is_valid")
    for token in ("playerid", "historical_markers", "xwoba", "xba", "xslg", "hardhit", "barrel", "whiff"):
        if token.lower() not in valid_src.lower():
            fail(f"current data validator missing {token}")
    checks["schema_validation"] = "PASS"

    # 6/8 nightly refresh and timing files exist.
    refresh = ROOT / "refresh_verified_current_data.py"
    workflow = ROOT / ".github" / "workflows" / "nightly_verified_current_data.yml"
    if not refresh.exists() or not workflow.exists():
        fail("nightly refresh files missing")
    py_compile.compile(str(refresh), doraise=True)
    wtxt = workflow.read_text(encoding="utf-8")
    if 'cron: "30 6 * * *"' not in wtxt:
        fail("nightly refresh is not fixed at 06:30 UTC")
    checks["nightly_refresh_after_10pm_pacific"] = "PASS"

    # 9 exact matchup refreshes are live/current, not daytime GitHub commits.
    ps = function_source(tree, text, "get_statcast_pitch_profile")
    bp = function_source(tree, text, "get_batter_statcast_pitch_type_profile")
    if "get_statcast_pitch_profile_live_v1" not in ps or "get_batter_statcast_pitch_type_profile_live_v1" not in bp:
        fail("live exact pitch matchup wrappers missing")
    checks["live_matchup_refresh"] = "PASS"

    # 10 persistence routing.
    if "RAILWAY_VOLUME_MOUNT_PATH" not in text or "MLB_STORAGE_DIR" not in text:
        fail("Railway persistent storage routing missing")
    checks["railway_volume_aware_storage"] = "PASS"

    # 12 exact pitch arsenal / batter pitch type, by MLBAM ID.
    live_pitcher = function_source(tree, text, "get_statcast_pitch_profile_live_v1")
    live_batter_pitch = function_source(tree, text, "get_batter_statcast_pitch_type_profile_live_v1")
    if "pitchers_lookup[]" not in live_pitcher or "statcast_search/csv" not in live_pitcher:
        fail("exact pitcher Statcast route missing MLBAM lookup")
    if "batters_lookup[]" not in live_batter_pitch or "statcast_search/csv" not in live_batter_pitch:
        fail("exact batter pitch-type route missing MLBAM lookup")
    checks["exact_pitch_arsenal_and_batter_pitch_type"] = "PASS"

    # Batter/pitcher platoon live-first current-season caches.
    for name, live_name in (
        ("_ow_batter_key_matchup_stats_context", "_ow_batter_key_matchup_stats_context_live_v1"),
        ("_ow_pitcher_allowed_split_context", "_ow_pitcher_allowed_split_context_live_v1"),
        ("_ow_batter_split_factor", "_ow_batter_split_factor_live_v1"),
    ):
        src = function_source(tree, text, name)
        if live_name not in src or "LAST_GOOD_CURRENT_SEASON" not in src:
            fail(f"verified current split fallback missing for {name}")
    checks["batter_pitcher_platoon_splits"] = "PASS"

    # 13 source/state provenance.
    if "_OW Data State" not in ps or "_OW Data Source" not in ps:
        fail("pitch matchup provenance missing")
    checks["provenance"] = "PASS"

    # Bullpen live recent workload from MLB source + short fallback.
    bullpen = function_source(tree, text, "get_recent_team_bullpen_usage")
    bullpen_live = function_source(tree, text, "get_recent_team_bullpen_usage_live_v1")
    schedule = function_source(tree, text, "_v3_team_schedule_context_map")
    if "get_recent_team_bullpen_usage_live_v1" not in bullpen or "LAST_GOOD_RECENT_FALLBACK" not in bullpen:
        fail("bullpen live/short-fallback wrapper missing")
    if "boxscore" not in bullpen_live.lower():
        fail("bullpen live source is not reading MLB boxscores")
    if "Opp Team ID" not in schedule:
        fail("schedule map does not expose opponent team ID for bullpen route")
    checks["bullpen_current_context"] = "PASS"

    # 14 HRR OVER/UNDER is side-consistent. Do not modify formula; certify current relation.
    hrr_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ow_build_hrr_rows_from_ud"]
    if not hrr_nodes:
        fail("HRR builder missing")
    hrr_src = function_source(tree, text, "_ow_build_hrr_rows_from_ud")
    # The final wrapper may call the protected production builder, so inspect all definitions as one block.
    lines = text.splitlines()
    hrr_all = "\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in hrr_nodes)
    required_formula_tokens = (
        '_ow_logistic_probability(edge, 1.10)',
        '0.65 * raw_over_rate + 0.35 * model_over_prob',
        'pick = "OVER" if over_prob >= 0.50 else "UNDER"',
        'win_prob = (over_prob if pick == "OVER" else (1.0 - over_prob)) * 100',
    )
    for token in required_formula_tokens:
        if token not in hrr_all:
            fail(f"HRR side/probability contract missing: {token}")
    checks["hrr_over_under_side_consistency"] = "PASS"

    # 15 weak-pitcher inputs are present but are not an automatic side override.
    if "Pitcher Allowed xwOBA" not in text or "Pitcher BAA" not in text or "Pitch Mix Matchup Factor" not in text:
        fail("pitcher vulnerability inputs missing from app")
    if 'pick = "OVER" if over_prob >= 0.50 else "UNDER"' not in hrr_all:
        fail("HRR pick is not probability-derived")
    checks["weak_pitcher_not_forced_over"] = "PASS"

    # Current/last_good promoted files, if a nightly promotion exists.
    promoted = {"state": "LIVE_ROUTE_PRIMARY_NO_PROMOTION_FOUND"}
    for root in verified_roots():
        manifest = read_manifest(root, "current")
        if not manifest:
            continue
        if int(manifest.get("season", 0) or 0) != SEASON:
            fail(f"verified current manifest is not {SEASON}: {root}")
        promoted = {
            "state": "PROMOTED_VERIFIED",
            "root": str(root),
            "batter": verify_promoted_dataset(root, manifest, "batter", 100),
            "pitcher": verify_promoted_dataset(root, manifest, "pitcher", 100),
        }
        break
    checks["promoted_verified_data"] = promoted

    # Bootstrap exists: normal operation should not require manual supporting-data uploads.
    bootstrap = ROOT / "bootstrap_verified_data.py"
    if not bootstrap.exists():
        fail("verified-data bootstrap missing")
    py_compile.compile(str(bootstrap), doraise=True)
    checks["manual_upload_required"] = False

    report = {
        "schema_version": 1,
        "season": SEASON,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "DATA_PIPELINE_ONLY_NO_HRR_FORMULA_OR_UI_CHANGE",
        "checks": checks,
        "railway_volume_env_present": bool(str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()),
        "explicit_storage_env_present": bool(str(os.getenv("MLB_STORAGE_DIR", "") or "").strip()),
    }
    try:
        out = storage_dir() / "data_pipeline_1_16_status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    print("DATA_PIPELINE_1_16_PASS")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
