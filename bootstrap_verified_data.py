#!/usr/bin/env python3
"""Copy only verified current/last_good supporting files into app persistent storage.

This is DATA PIPELINE ONLY. It never imports app.py and never changes projection math/UI.
It removes the need for manual data uploads when a verified nightly promotion exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEASON = int(os.getenv("MLB_SEASON", "2026"))

CANONICAL = {
    "batter": "savant_batter_profiles.csv",
    "pitcher": "savant_pitcher_stats.csv",
    "batter_platoon": f"savant_batter_platoon_{SEASON}.csv",
    "pitcher_platoon": f"savant_pitcher_platoon_{SEASON}.csv",
    "bullpen": "bullpen_context.csv",
    "player_identity": f"player_identity_{SEASON}.csv",
}


def storage_dir() -> Path:
    explicit = str(os.getenv("MLB_STORAGE_DIR", "") or "").strip()
    volume = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
    if explicit:
        return Path(explicit)
    if volume:
        return Path(volume) / "mlb_engine"
    return ROOT / "mlb_engine"


def candidate_verified_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = str(os.getenv("MLB_VERIFIED_DATA_ROOT", "") or "").strip()
    volume = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
    if explicit:
        roots.append(Path(explicit))
    if volume:
        roots.append(Path(volume) / "verified_current_data")
    roots.append(ROOT / "data" / "verified_current")
    out, seen = [], set()
    for p in roots:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def read_manifest(root: Path, state: str) -> dict:
    p = root / "manifests" / f"{state}_manifest.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_ok(manifest: dict, kind: str, season: int) -> bool:
    try:
        if int(manifest.get("season", 0)) != int(season):
            return False
        ds = (manifest.get("datasets") or {}).get(kind) or {}
        return str(ds.get("status", "")).upper() in {"PASS", "VERIFIED", "OK"} and int(ds.get("rows", 0) or 0) > 0
    except Exception:
        return False


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_name(dst.stem + ".last_good" + dst.suffix)
        tmp_backup = backup.with_suffix(backup.suffix + ".tmp")
        shutil.copy2(dst, tmp_backup)
        os.replace(tmp_backup, backup)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main() -> int:
    target_root = storage_dir() / "batter_fantasy_data"
    target_root.mkdir(parents=True, exist_ok=True)
    copied = []
    source_used = None

    for root in candidate_verified_roots():
        manifest = read_manifest(root, "current")
        if int(manifest.get("season", 0) or 0) != SEASON:
            continue
        any_copy = False
        for kind, filename in CANONICAL.items():
            if not dataset_ok(manifest, kind, SEASON):
                continue
            ds = (manifest.get("datasets") or {}).get(kind) or {}
            rel = str(ds.get("path") or f"current/{filename}")
            src = root / rel
            if not src.exists():
                src = root / "current" / filename
            if not src.exists() or not src.is_file():
                continue
            expected_sha = str(ds.get("sha256") or "").strip().lower()
            actual_sha = sha256(src)
            if expected_sha and expected_sha != actual_sha:
                continue
            dst = target_root / filename
            atomic_copy(src, dst)
            copied.append({"kind": kind, "source": str(src), "target": str(dst), "sha256": actual_sha})
            any_copy = True
        if any_copy:
            source_used = str(root)
            break

    status = {
        "schema_version": 1,
        "season": SEASON,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_dir": str(storage_dir()),
        "verified_root": source_used,
        "copied": copied,
        "manual_upload_required": False,
        "note": "Verified nightly files are auto-installed when available; live MLB/Savant routes remain the primary fallback.",
    }
    (target_root / "verified_bootstrap_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
