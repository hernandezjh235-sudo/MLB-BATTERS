#!/usr/bin/env python3
"""Nightly verified-current MLB Statcast refresh.

DATA PIPELINE ONLY. This script does not import or change app.py projection formulas.
It downloads batter + pitcher current-season Statcast tables into staging, validates
both, then promotes them together. If either dataset fails validation, current and
last_good are left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SAVANT_URL = "https://baseballsavant.mlb.com/leaderboard/custom"
SELECTIONS = [
    "pa", "hit", "home_run", "strikeout", "walk",
    "k_percent", "bb_percent", "batting_avg", "slg_percent", "on_base_percent",
    "xba", "xslg", "woba", "xwoba", "xobp", "xiso",
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "barrel_batted_rate", "hard_hit_percent", "avg_best_speed",
    "whiff_percent", "oz_swing_percent", "iz_contact_percent",
    "pull_percent", "flyballs_percent", "groundballs_percent",
]
FILE_NAMES = {
    "batter": "savant_batter_profiles.csv",
    "pitcher": "savant_pitcher_stats.csv",
}
MIN_ROWS = {"batter": 200, "pitcher": 200}


def norm_cols(df: pd.DataFrame) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}


def metric_presence(df: pd.DataFrame) -> dict[str, bool]:
    keys = set(norm_cols(df))
    groups = {
        "xwoba": ("xwoba", "estwoba", "estimatedwoba"),
        "xba": ("xba", "estba", "estimatedba"),
        "xslg": ("xslg", "estslg", "estimatedslg"),
        "ev": ("exvelocityavg", "exitvelocityavg", "avghitspeed", "avgev"),
        "hardhit": ("hardhitpercent", "hardhitpct", "hardhit"),
        "barrel": ("barrelbattedrate", "barrelpercent", "barrelpct", "brlpercent"),
        "whiff": ("whiffpercent", "whiffpct", "whiff"),
        "k_pct": ("kpercent", "kpct", "strikeoutpercent"),
        "bb_pct": ("bbpercent", "bbpct", "walkpercent"),
        "sweetspot": ("sweetspotpercent", "sweetspotpct"),
        "zone_contact": ("izcontactpercent", "zonecontactpercent", "zcontactpercent"),
    }
    return {name: any(k in keys for k in aliases) for name, aliases in groups.items()}


def validate(df: pd.DataFrame, kind: str, season: int) -> tuple[bool, dict]:
    info: dict = {"kind": kind, "season": season, "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0}
    if not isinstance(df, pd.DataFrame) or df.empty:
        info["error"] = "empty dataframe"
        return False, info

    n = norm_cols(df)
    keys = set(n)
    historical_markers = {
        "historicalpa", "careerpa", "careerg", "careerab", "careerh", "careerhr",
        "profilesource", "startyear", "endyear", "seasons", "careerops",
    }
    if len(keys & historical_markers) >= 2:
        info["error"] = "historical/career schema rejected"
        return False, info

    if len(df) < MIN_ROWS[kind]:
        info["error"] = f"row count below minimum {MIN_ROWS[kind]}"
        return False, info

    id_col = next((n[k] for k in ("playerid", "mlbamid", "mlbam", "keymlbam", "batterid", "pitcherid") if k in n), None)
    if not id_col:
        info["error"] = "missing MLBAM/player_id"
        return False, info

    ids = pd.to_numeric(df[id_col], errors="coerce")
    info["id_column"] = str(id_col)
    info["id_coverage"] = round(float(ids.notna().mean()), 6)
    info["unique_ids"] = int(ids.dropna().nunique())
    info["duplicate_id_rows"] = int(ids.dropna().duplicated().sum())
    if info["id_coverage"] < 0.90:
        info["error"] = "MLBAM/player_id coverage below 90%"
        return False, info
    if info["unique_ids"] < int(len(df) * 0.70):
        info["error"] = "too many duplicate player ids"
        return False, info

    name_col = next((n[k] for k in ("lastnamefirstname", "playername", "name", "player", "batter", "pitcher") if k in n), None)
    if not name_col:
        info["error"] = "missing player name identity"
        return False, info
    info["name_column"] = str(name_col)
    info["name_coverage"] = round(float(df[name_col].astype(str).str.strip().ne("").mean()), 6)

    metrics = metric_presence(df)
    info["metric_presence"] = metrics
    if sum(metrics.values()) < 7:
        info["error"] = "insufficient Statcast metric coverage"
        return False, info
    for required in ("xwoba", "xba", "xslg", "hardhit", "barrel", "whiff", "k_pct", "bb_pct"):
        if not metrics.get(required):
            info["error"] = f"required Statcast field missing: {required}"
            return False, info

    ycol = n.get("season") or n.get("year")
    if ycol:
        years = pd.to_numeric(df[ycol], errors="coerce")
        if years.notna().any() and not (years.astype("Int64") == season).all():
            info["error"] = f"non-{season} rows present"
            return False, info

    info["status"] = "PASS"
    return True, info


def fetch(kind: str, season: int) -> tuple[pd.DataFrame, str]:
    params = {
        "year": season,
        "type": kind,
        "filter": "",
        "min": "1",
        "selections": ",".join(SELECTIONS),
        "chart": "false",
        "sort": "xwoba",
        "sortDir": "desc",
        "csv": "true",
    }
    response = requests.get(
        SAVANT_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (MLB-BATTERS verified nightly refresh)"},
    )
    response.raise_for_status()
    body = (response.text or "").strip()
    if not body or body.startswith("<") or "," not in body[:1000]:
        raise RuntimeError(f"{kind}: Savant did not return CSV")
    df = pd.read_csv(io.StringIO(body), low_memory=False)
    df["season"] = int(season)
    df["_OW Source"] = f"BASEBALL_SAVANT_CUSTOM_{kind.upper()}"
    df["_OW Fetched At UTC"] = datetime.now(timezone.utc).isoformat()
    return df, response.url


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def promote(root: Path, season: int, staged: dict[str, tuple[Path, dict, str]]) -> dict:
    current = root / "current"
    last_good = root / "last_good"
    manifests = root / "manifests"
    current_manifest_path = manifests / "current_manifest.json"
    last_good_manifest_path = manifests / "last_good_manifest.json"
    manifests.mkdir(parents=True, exist_ok=True)

    if current.exists() and any(current.iterdir()):
        safe_rmtree(last_good)
        shutil.copytree(current, last_good)
        if current_manifest_path.exists():
            shutil.copy2(current_manifest_path, last_good_manifest_path)

    current.mkdir(parents=True, exist_ok=True)
    datasets = {}
    generated = datetime.now(timezone.utc).isoformat()
    for kind, (stage_path, info, source_url) in staged.items():
        dest = current / FILE_NAMES[kind]
        tmp = current / (FILE_NAMES[kind] + ".tmp")
        shutil.copy2(stage_path, tmp)
        os.replace(tmp, dest)
        rec = dict(info)
        rec.update({
            "status": "PASS",
            "path": str(dest.relative_to(root)),
            "sha256": sha256_file(dest),
            "source_url": source_url,
            "fetched_at_utc": generated,
        })
        datasets[kind] = rec

    manifest = {
        "schema_version": 2,
        "season": int(season),
        "generated_at_utc": generated,
        "promotion": "STAGING_TO_CURRENT_ATOMIC_AFTER_ALL_DATASETS_PASS",
        "datasets": datasets,
    }
    tmp_manifest = manifests / "current_manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, current_manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--root", default=os.environ.get("MLB_VERIFIED_DATA_ROOT", "data/verified_current"))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    staged: dict[str, tuple[Path, dict, str]] = {}
    failures = []
    for kind in ("batter", "pitcher"):
        try:
            df, source_url = fetch(kind, args.season)
            ok, info = validate(df, kind, args.season)
            if not ok:
                raise RuntimeError(info.get("error", "validation failed"))
            stage_path = staging / FILE_NAMES[kind]
            df.to_csv(stage_path, index=False)
            reread = pd.read_csv(stage_path, low_memory=False)
            ok2, info2 = validate(reread, kind, args.season)
            if not ok2:
                raise RuntimeError("post-write validation failed: " + info2.get("error", "unknown"))
            staged[kind] = (stage_path, info2, source_url)
            print(f"{kind}: PASS rows={len(reread)} id_coverage={info2['id_coverage']:.3f}")
        except Exception as exc:
            failures.append(f"{kind}: {exc}")
            print(f"{kind}: FAIL - {exc}", file=sys.stderr)

    if failures or set(staged) != {"batter", "pitcher"}:
        print("REFRESH_ABORTED_CURRENT_UNCHANGED", file=sys.stderr)
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2

    manifest = promote(root, args.season, staged)
    safe_rmtree(staging)
    print("REFRESH_PROMOTED")
    print(json.dumps({
        "season": manifest["season"],
        "generated_at_utc": manifest["generated_at_utc"],
        "batter_rows": manifest["datasets"]["batter"]["rows"],
        "pitcher_rows": manifest["datasets"]["pitcher"]["rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
