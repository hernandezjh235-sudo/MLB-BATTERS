# -*- coding: utf-8 -*-
"""
ONE WAY PICKZ — MLB HITS + RUNS + RBI ENGINE
Daily Underdog H+R+RBI projection and selection app.

Built from the user's app 75.py workflow, but intentionally removes all active
Strikeout, Pitching Outs, Pitcher Fantasy, and Moneyline screens/models.

Core workflow
-------------
1. Pull ONLY active Underdog MLB Hits + Runs + RBIs lines.
2. Match those players to official MLB player IDs.
3. Pull current-season batter data from Opening Day through today, prior-season
   data for regression, today's probable pitcher, current pitcher data, current
   lineup status, team offense, bullpen workload, venue and MLB weather.
4. Build Bayesian/regressed batter and pitcher profiles with optional Statcast plus the bundled cleaned 2015-2024 batter-history prior.
5. Run a correlated base/out-state Monte Carlo simulation so Hits, Runs and RBI
   are not treated as independent.
6. Rank Over and Under candidates with data-quality, model-agreement, role-risk,
   lineup, volatility, market-odds and learning gates.
7. Save official snapshots, grade automatically, learn, export CSV/JSON, and
   optionally back up files to GitHub.

No fake prop lines are created. If Underdog cannot be reached, use the manual
H+R+RBI CSV/text fallback in the Data Manager.
"""

from __future__ import annotations

import base64
import csv
import difflib
import hashlib
import html
import io
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import pytz
except Exception:  # pragma: no cover
    pytz = None

# ============================================================
# APP / STORAGE
# ============================================================
APP_VERSION = "ONE WAY PICKZ MLB H+R+RBI v2.1 — HISTORICAL BATTER PRIORS"
MODEL_VERSION = "HRR_FULL_LINEUP_PA_BULLPEN_HIST_PRIOR_MC_2026_07_13"
TZ_NAME = "America/Los_Angeles"
MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
UNDERDOG_URLS = [
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
    "https://api.underdogfantasy.com/beta/v2/over_under_lines",
    "https://api.underdogfantasy.com/v1/over_under_lines",
]

LOCAL_DIR = Path(os.getenv("HRR_STORAGE_DIR", "mlb_hrr_engine"))
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = LOCAL_DIR / "profiles"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PICK_LOG = LOCAL_DIR / "hrr_official_picks.json"
RESULT_LOG = LOCAL_DIR / "hrr_results.json"
LEARNING_FILE = LOCAL_DIR / "hrr_learning.json"
SAVED_ODDS_FILE = LOCAL_DIR / "hrr_saved_odds.json"
REQUEST_LOG_FILE = LOCAL_DIR / "hrr_request_log.json"
GRADED_CSV = LOCAL_DIR / "hrr_graded_history.csv"
MANUAL_LINES_FILE = LOCAL_DIR / "hrr_manual_lines.csv"
PROFILE_BUILD_STATE = LOCAL_DIR / "hrr_profile_build_state.json"

# Optional offline profiles produced from the user's planned 2021-2025 builder.
# The daily app runs without these files, but will use them automatically when present.
HIST_BATTER_PROFILE_CANDIDATES = [
    Path("data/batter_profiles.parquet"),
    Path("data/batter_profiles.csv"),
    Path("learning_data/batter_profiles.parquet"),
    Path("learning_data/batter_profiles.csv"),
    CACHE_DIR / "batter_profiles_2021_2025.csv",
]
RAW_HIST_BATTER_CANDIDATES = [
    Path("data/raw/cleaned_batting_stats.csv"),
    Path("data/cleaned_batting_stats.csv"),
    Path("cleaned_batting_stats.csv"),
]
GENERATED_HIST_BATTER_PROFILE = Path("data/batter_profiles.csv")
HIST_PITCHER_PROFILE_CANDIDATES = [
    Path("data/pitcher_profiles.parquet"),
    Path("data/pitcher_profiles.csv"),
    Path("learning_data/pitcher_profiles.parquet"),
    Path("learning_data/pitcher_profiles.csv"),
    CACHE_DIR / "pitcher_profiles_2021_2025.csv",
]

# Conservative league priors. Historical/offline files replace these when available.
LEAGUE = {
    "bb_pa": 0.082,
    "hbp_pa": 0.012,
    "k_pa": 0.225,
    "single_pa": 0.155,
    "double_pa": 0.045,
    "triple_pa": 0.004,
    "hr_pa": 0.031,
    "runs_per_game": 4.45,
    "pa_per_team_game": 37.3,
    "obp": 0.315,
    "slg": 0.405,
    "ops": 0.720,
}

# Conservative outcome-specific park factors. These are complete fallbacks for
# every current MLB venue/temporary venue. If a user supplies a park_factors.csv
# file in data/ or learning_data/, those values override these defaults.
PARK_FACTORS: Dict[str, Dict[str, float]] = {
    "American Family Field": {"1B": 1.00, "2B": 1.01, "3B": 0.96, "HR": 1.04, "R": 1.02},
    "Angel Stadium": {"1B": 0.99, "2B": 0.98, "3B": 0.96, "HR": 0.98, "R": 0.98},
    "Busch Stadium": {"1B": 1.00, "2B": 1.01, "3B": 1.03, "HR": 0.92, "R": 0.97},
    "Chase Field": {"1B": 1.01, "2B": 1.04, "3B": 1.02, "HR": 1.01, "R": 1.02},
    "Citi Field": {"1B": 0.99, "2B": 0.99, "3B": 1.00, "HR": 0.96, "R": 0.98},
    "Citizens Bank Park": {"1B": 1.00, "2B": 1.00, "3B": 0.96, "HR": 1.07, "R": 1.04},
    "Comerica Park": {"1B": 1.01, "2B": 1.06, "3B": 1.10, "HR": 0.93, "R": 1.00},
    "Coors Field": {"1B": 1.05, "2B": 1.10, "3B": 1.08, "HR": 1.12, "R": 1.10},
    "Daikin Park": {"1B": 0.99, "2B": 1.00, "3B": 0.93, "HR": 1.04, "R": 1.00},
    "Minute Maid Park": {"1B": 0.99, "2B": 1.00, "3B": 0.93, "HR": 1.04, "R": 1.00},
    "Dodger Stadium": {"1B": 0.98, "2B": 1.00, "3B": 0.95, "HR": 1.03, "R": 1.00},
    "Fenway Park": {"1B": 1.02, "2B": 1.12, "3B": 0.96, "HR": 1.01, "R": 1.05},
    "George M. Steinbrenner Field": {"1B": 1.01, "2B": 1.02, "3B": 1.00, "HR": 1.03, "R": 1.02},
    "Globe Life Field": {"1B": 1.00, "2B": 1.02, "3B": 0.98, "HR": 1.02, "R": 1.01},
    "Great American Ball Park": {"1B": 1.00, "2B": 1.01, "3B": 0.96, "HR": 1.10, "R": 1.05},
    "Guaranteed Rate Field": {"1B": 1.00, "2B": 0.99, "3B": 0.94, "HR": 1.06, "R": 1.02},
    "Rate Field": {"1B": 1.00, "2B": 0.99, "3B": 0.94, "HR": 1.06, "R": 1.02},
    "Kauffman Stadium": {"1B": 1.02, "2B": 1.05, "3B": 1.10, "HR": 0.93, "R": 1.00},
    "loanDepot park": {"1B": 0.98, "2B": 0.98, "3B": 1.00, "HR": 0.93, "R": 0.95},
    "Nationals Park": {"1B": 1.00, "2B": 1.01, "3B": 0.98, "HR": 1.02, "R": 1.01},
    "Oracle Park": {"1B": 1.01, "2B": 1.05, "3B": 1.12, "HR": 0.90, "R": 0.97},
    "Oriole Park at Camden Yards": {"1B": 1.00, "2B": 1.02, "3B": 0.96, "HR": 0.98, "R": 0.99},
    "Petco Park": {"1B": 0.99, "2B": 0.98, "3B": 0.98, "HR": 0.94, "R": 0.96},
    "PNC Park": {"1B": 1.01, "2B": 1.05, "3B": 1.08, "HR": 0.91, "R": 0.98},
    "Progressive Field": {"1B": 1.00, "2B": 1.03, "3B": 0.98, "HR": 1.00, "R": 1.00},
    "Rogers Centre": {"1B": 1.00, "2B": 1.00, "3B": 0.94, "HR": 1.04, "R": 1.02},
    "Sutter Health Park": {"1B": 1.02, "2B": 1.03, "3B": 1.01, "HR": 1.05, "R": 1.04},
    "Target Field": {"1B": 1.00, "2B": 1.02, "3B": 1.02, "HR": 0.98, "R": 0.99},
    "T-Mobile Park": {"1B": 0.98, "2B": 0.97, "3B": 0.98, "HR": 0.92, "R": 0.94},
    "Truist Park": {"1B": 1.00, "2B": 1.01, "3B": 0.97, "HR": 1.04, "R": 1.02},
    "Wrigley Field": {"1B": 1.00, "2B": 1.02, "3B": 1.00, "HR": 1.00, "R": 1.00},
    "Yankee Stadium": {"1B": 0.99, "2B": 0.95, "3B": 0.88, "HR": 1.08, "R": 1.02},
    "Tropicana Field": {"1B": 0.99, "2B": 0.98, "3B": 1.00, "HR": 0.96, "R": 0.97},
}

# Coordinates and center-field bearings are used only for small weather-vector
# adjustments. Unknown venues safely fall back to MLB-reported weather and a
# neutral orientation. Bearings are degrees clockwise from true north.
STADIUM_META: Dict[str, Dict[str, Any]] = {
    "American Family Field": {"lat": 43.0280, "lon": -87.9712, "bearing": 63, "roof": "retractable"},
    "Angel Stadium": {"lat": 33.8003, "lon": -117.8827, "bearing": 55, "roof": "open"},
    "Busch Stadium": {"lat": 38.6226, "lon": -90.1928, "bearing": 80, "roof": "open"},
    "Chase Field": {"lat": 33.4453, "lon": -112.0667, "bearing": 0, "roof": "retractable"},
    "Citi Field": {"lat": 40.7571, "lon": -73.8458, "bearing": 51, "roof": "open"},
    "Citizens Bank Park": {"lat": 39.9061, "lon": -75.1665, "bearing": 9, "roof": "open"},
    "Comerica Park": {"lat": 42.3390, "lon": -83.0485, "bearing": 164, "roof": "open"},
    "Coors Field": {"lat": 39.7559, "lon": -104.9942, "bearing": 32, "roof": "open"},
    "Daikin Park": {"lat": 29.7573, "lon": -95.3555, "bearing": 30, "roof": "retractable"},
    "Minute Maid Park": {"lat": 29.7573, "lon": -95.3555, "bearing": 30, "roof": "retractable"},
    "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "bearing": 24, "roof": "open"},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "bearing": 54, "roof": "open"},
    "George M. Steinbrenner Field": {"lat": 27.9804, "lon": -82.5067, "bearing": 28, "roof": "open"},
    "Globe Life Field": {"lat": 32.7473, "lon": -97.0847, "bearing": 41, "roof": "retractable"},
    "Great American Ball Park": {"lat": 39.0979, "lon": -84.5082, "bearing": 52, "roof": "open"},
    "Guaranteed Rate Field": {"lat": 41.8299, "lon": -87.6338, "bearing": 148, "roof": "open"},
    "Rate Field": {"lat": 41.8299, "lon": -87.6338, "bearing": 148, "roof": "open"},
    "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803, "bearing": 63, "roof": "open"},
    "loanDepot park": {"lat": 25.7781, "lon": -80.2197, "bearing": 91, "roof": "retractable"},
    "Nationals Park": {"lat": 38.8730, "lon": -77.0074, "bearing": 28, "roof": "open"},
    "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "bearing": 60, "roof": "open"},
    "Oriole Park at Camden Yards": {"lat": 39.2839, "lon": -76.6217, "bearing": 31, "roof": "open"},
    "Petco Park": {"lat": 32.7073, "lon": -117.1566, "bearing": 5, "roof": "open"},
    "PNC Park": {"lat": 40.4469, "lon": -80.0057, "bearing": 113, "roof": "open"},
    "Progressive Field": {"lat": 41.4962, "lon": -81.6852, "bearing": 1, "roof": "open"},
    "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "bearing": 0, "roof": "retractable"},
    "Sutter Health Park": {"lat": 38.5803, "lon": -121.5139, "bearing": 63, "roof": "open"},
    "Target Field": {"lat": 44.9817, "lon": -93.2776, "bearing": 91, "roof": "open"},
    "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "bearing": 49, "roof": "retractable"},
    "Truist Park": {"lat": 33.8908, "lon": -84.4677, "bearing": 71, "roof": "open"},
    "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "bearing": 24, "roof": "open"},
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "bearing": 75, "roof": "open"},
    "Tropicana Field": {"lat": 27.7682, "lon": -82.6534, "bearing": 39, "roof": "fixed"},
}

PARK_FACTOR_FILE_CANDIDATES = [
    Path("data/park_factors.csv"), Path("learning_data/park_factors.csv"), CACHE_DIR / "park_factors.csv"
]

# ============================================================
# STREAMLIT PAGE / CSS
# ============================================================
st.set_page_config(page_title="One Way Pickz — MLB H+R+RBI", page_icon="⚾", layout="wide")
st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at top left,#20123d 0%,#09070f 42%,#050507 100%); color:#f5f3ff;}
.hero {padding:22px 24px;border-radius:22px;background:linear-gradient(135deg,rgba(112,45,190,.32),rgba(10,8,18,.95));border:1px solid rgba(240,190,70,.45);box-shadow:0 14px 40px rgba(0,0,0,.35);margin-bottom:16px;}
.hero h1 {margin:0;color:#fff;font-size:34px;font-weight:950;letter-spacing:-.5px;}
.hero p {margin:7px 0 0;color:#d8cdf4;font-weight:650;}
.hrr-card {padding:18px;border-radius:19px;background:linear-gradient(145deg,rgba(31,20,48,.96),rgba(8,7,12,.98));border:1px solid rgba(177,111,255,.34);box-shadow:0 10px 28px rgba(0,0,0,.32);margin:10px 0;}
.player {font-size:23px;font-weight:950;color:#fff;}
.sub {font-size:13px;color:#bdb2d3;margin-top:3px;}
.pick-over {color:#57f39a;font-weight:950;font-size:25px;}
.pick-under {color:#ff7373;font-weight:950;font-size:25px;}
.pick-pass {color:#ffd66b;font-weight:950;font-size:25px;}
.metric-grid {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-top:12px;}
.metric {background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.1);border-radius:13px;padding:10px;min-height:67px;}
.metric .k {color:#b9adc9;font-size:11px;text-transform:uppercase;font-weight:850;letter-spacing:.05em;}
.metric .v {color:#fff;font-size:20px;font-weight:950;margin-top:4px;}
.factor-good {color:#69f3a7}.factor-bad{color:#ff8585}.factor-neutral{color:#ffd778}
.section {font-size:23px;font-weight:950;border-left:5px solid #e3b64d;padding-left:11px;margin:18px 0 10px;}
.small-note {font-size:12px;color:#bdb2d3;line-height:1.45;}
@media (max-width:950px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.hero h1{font-size:27px}.player{font-size:20px}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# GENERIC HELPERS
# ============================================================
def la_now() -> datetime:
    if pytz:
        return datetime.now(pytz.timezone(TZ_NAME))
    return datetime.utcnow() - timedelta(hours=7)


def now_iso() -> str:
    return la_now().isoformat(timespec="seconds")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "" or str(value).lower() in {"nan", "none", "—"}:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_name(value: Any) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def write_json(path: Path, data: Any, protect: bool = True) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if protect and path.exists():
            old = read_json(path, None)
            if isinstance(old, list) and isinstance(data, list) and len(old) >= 25:
                if len(data) == 0 or len(data) < int(len(old) * 0.85):
                    return False
            try:
                path.with_suffix(path.suffix + ".bak").write_text(json.dumps(old, indent=2, default=str))
            except Exception:
                pass
        path.write_text(json.dumps(data, indent=2, default=str))
        return True
    except Exception:
        return False


def append_request_log(url: str, status: str, detail: str = "") -> None:
    rows = read_json(REQUEST_LOG_FILE, [])
    rows.append({"time": now_iso(), "url": url, "status": status, "detail": str(detail)[:400]})
    write_json(REQUEST_LOG_FILE, rows[-300:], protect=False)


def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 20) -> Optional[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 OneWayPickz-HRR/1.0",
        "Accept": "application/json,text/plain,*/*",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        if r.status_code != 200:
            append_request_log(url, f"HTTP {r.status_code}", r.text[:200])
            return None
        return r.json()
    except Exception as exc:
        append_request_log(url, "REQUEST_ERROR", str(exc))
        return None


def american_to_prob(odds: Any) -> Optional[float]:
    o = safe_float(odds)
    if o is None or o == 0:
        return None
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def fair_american(prob: float) -> str:
    p = clamp(float(prob), 0.001, 0.999)
    if p >= 0.5:
        return str(int(round(-100 * p / (1 - p))))
    return f"+{int(round(100 * (1 - p) / p))}"


def no_vig_probability(over_odds: Any, under_odds: Any, side: str) -> Optional[float]:
    po, pu = american_to_prob(over_odds), american_to_prob(under_odds)
    if po is None or pu is None or po + pu <= 0:
        return None
    return po / (po + pu) if side.upper() == "OVER" else pu / (po + pu)


def safe_read_table(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_existing_profile(paths: Sequence[Path]) -> pd.DataFrame:
    """Load all available profile layers without collapsing name-only rows.

    MLB-ID Statcast profiles and name-only historical batting priors are allowed
    to coexist. They are merged later by offline_profile_for().
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = safe_read_table(path)
        if not df.empty:
            df = df.copy()
            df["_profile_source"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ["end_year", "updated_at", "historical_pa", "sample_size"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, na_position="first")

    id_col = next((c for c in ["player_id", "Player ID", "mlbam_id", "batter", "pitcher"] if c in merged.columns), None)
    name_col = next((c for c in ["normalized_name", "player_name", "Player", "Name", "full_name"] if c in merged.columns), None)
    pieces: List[pd.DataFrame] = []
    if id_col:
        ids = pd.to_numeric(merged[id_col], errors="coerce")
        with_id = merged[ids.notna()].copy()
        if not with_id.empty:
            with_id["_profile_id"] = pd.to_numeric(with_id[id_col], errors="coerce")
            with_id = with_id.drop_duplicates(subset=["_profile_id"], keep="last").drop(columns=["_profile_id"])
            pieces.append(with_id)
        merged = merged[ids.isna()].copy()
    if not merged.empty and name_col:
        if name_col == "normalized_name":
            merged["_profile_name"] = merged[name_col].astype(str)
        else:
            merged["_profile_name"] = merged[name_col].astype(str).map(normalize_name)
        merged = merged[merged["_profile_name"].ne("")]
        merged = merged.drop_duplicates(subset=["_profile_name"], keep="last").drop(columns=["_profile_name"])
    if not merged.empty:
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True, sort=False).reset_index(drop=True) if pieces else pd.DataFrame()


def _repair_legacy_name(value: Any) -> Tuple[str, str]:
    raw = str(value or "").strip()
    for _ in range(2):
        if not any(ch in raw for ch in ("Ã", "Â", "â", "ð", "�")):
            break
        repaired = None
        for encoding in ("latin1", "cp1252"):
            try:
                candidate = raw.encode(encoding).decode("utf-8")
                if candidate != raw:
                    repaired = candidate
                    break
            except Exception:
                continue
        if not repaired:
            break
        raw = repaired
    side = "S" if raw.endswith("#") else ("L" if raw.endswith("*") else "R")
    return re.sub(r"[*#]+$", "", raw).strip(), side


def build_legacy_batter_prior(raw_path: Path, output_path: Path = GENERATED_HIST_BATTER_PROFILE) -> pd.DataFrame:
    """Convert the uploaded 2015-2024 player-season file to one row/player.

    Traded-player duplicate seasons prefer the combined 2TM/3TM row. Rates are
    PA-weighted with a three-year recency half-life. The original source remains
    untouched in data/raw/.
    """
    raw = safe_read_table(raw_path)
    required = {"Player", "Year", "PA", "H", "R", "RBI", "2B", "3B", "HR", "BB", "SO", "HBP"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    df = raw.copy()
    repaired = df["Player"].map(_repair_legacy_name)
    df["player_name"] = [v[0] for v in repaired]
    df["bats"] = [v[1] for v in repaired]
    df["normalized_name"] = df["player_name"].map(normalize_name)
    numeric_cols = ["Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+", "rOBA", "Rbat+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "WAR", "Year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["normalized_name"].ne("") & df["Year"].notna() & df["PA"].fillna(0).gt(0)].copy()
    if df.empty:
        return pd.DataFrame()
    df["Year"] = df["Year"].astype(int)
    df["_combined_team"] = df.get("Team", pd.Series(index=df.index, dtype=str)).astype(str).str.match(r"^\d+TM$", na=False)
    df = df.sort_values(["normalized_name", "Year", "_combined_team", "PA"], ascending=[True, True, False, False])
    df = df.drop_duplicates(["normalized_name", "Year"], keep="first")
    df["1B"] = (df["H"].fillna(0) - df["2B"].fillna(0) - df["3B"].fillna(0) - df["HR"].fillna(0)).clip(lower=0)
    max_year = int(df["Year"].max())
    rows: List[Dict[str, Any]] = []
    rate_columns = {"bb_pa": "BB", "hbp_pa": "HBP", "k_pa": "SO", "single_pa": "1B", "double_pa": "2B", "triple_pa": "3B", "hr_pa": "HR", "run_pa": "R", "rbi_pa": "RBI", "gidp_pa": "GIDP", "sf_pa": "SF", "sb_pa": "SB"}
    for normalized, group in df.groupby("normalized_name", sort=False):
        group = group.sort_values("Year")
        last = group.iloc[-1]
        years = group["Year"].to_numpy(dtype=float)
        pa = group["PA"].fillna(0).to_numpy(dtype=float)
        recency = np.power(0.5, (max_year - years) / 3.0)
        weighted_pa = float(np.sum(pa * recency))
        historical_pa = float(np.sum(pa))
        record: Dict[str, Any] = {
            "player_name": str(last["player_name"]), "Name": str(last["player_name"]),
            "normalized_name": normalized, "bats": str(last.get("bats") or "R"),
            "start_year": int(group["Year"].min()), "end_year": int(group["Year"].max()),
            "seasons": int(group["Year"].nunique()), "historical_pa": round(historical_pa, 1),
            "career_pa": round(historical_pa, 1), "effective_pa": round(weighted_pa, 1),
            "age_last": safe_float(last.get("Age")), "profile_source": "cleaned_batting_stats_2015_2024",
            "updated_at": now_iso(),
        }
        for rate_name, numerator in rate_columns.items():
            values = group.get(numerator, pd.Series(0, index=group.index)).fillna(0).to_numpy(dtype=float)
            record[rate_name] = float(np.sum(values * recency) / weighted_pa) if weighted_pa > 0 else None
        hrr_values = (group["H"].fillna(0) + group["R"].fillna(0) + group["RBI"].fillna(0)).to_numpy(dtype=float)
        record["hrr_pa"] = float(np.sum(hrr_values * recency) / weighted_pa) if weighted_pa > 0 else None
        for stat in ["BA", "OBP", "SLG", "OPS", "OPS+", "rOBA", "Rbat+", "WAR"]:
            if stat not in group:
                continue
            values = group[stat].to_numpy(dtype=float)
            valid = np.isfinite(values) & (pa > 0)
            if valid.any():
                record[stat.lower().replace("+", "_plus")] = float(np.average(values[valid], weights=(pa * recency)[valid]))
        rows.append(record)
    output = pd.DataFrame(rows).sort_values("player_name").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def ensure_legacy_batter_prior() -> None:
    raw_path = next((p for p in RAW_HIST_BATTER_CANDIDATES if p.exists()), None)
    if raw_path is None:
        return
    needs_build = not GENERATED_HIST_BATTER_PROFILE.exists()
    if not needs_build:
        try:
            needs_build = raw_path.stat().st_mtime > GENERATED_HIST_BATTER_PROFILE.stat().st_mtime
        except Exception:
            needs_build = False
    if needs_build:
        try:
            build_legacy_batter_prior(raw_path, GENERATED_HIST_BATTER_PROFILE)
        except Exception as exc:
            append_request_log(str(raw_path), "HISTORICAL_PROFILE_BUILD_ERROR", str(exc))


@st.cache_data(ttl=3600, show_spinner=False)
def historical_batter_profiles() -> pd.DataFrame:
    ensure_legacy_batter_prior()
    return first_existing_profile(HIST_BATTER_PROFILE_CANDIDATES)


@st.cache_data(ttl=3600, show_spinner=False)
def historical_pitcher_profiles() -> pd.DataFrame:
    return first_existing_profile(HIST_PITCHER_PROFILE_CANDIDATES)

# ============================================================
# SEASON / MLB ID / SCHEDULE
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def discover_opening_day(season: int) -> str:
    start, end = f"{season}-03-01", f"{season}-04-20"
    data = safe_get_json(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "startDate": start, "endDate": end, "gameTypes": "R"},
        timeout=25,
    ) or {}
    dates = [d.get("date") for d in data.get("dates", []) if d.get("games")]
    if dates:
        return min(dates)
    # 2026 official fallback; generic fallback for later seasons.
    return "2026-03-26" if season == 2026 else f"{season}-03-27"


@st.cache_data(ttl=86400, show_spinner=False)
def search_mlb_person(name: str) -> Dict[str, Any]:
    clean = str(name or "").strip()
    if not clean:
        return {}
    data = safe_get_json(f"{MLB_BASE}/people/search", params={"names": clean, "sportIds": 1}, timeout=15) or {}
    people = data.get("people") or []
    if not people:
        return {}
    target = normalize_name(clean)
    best, score = {}, 0.0
    for p in people:
        ratio = difflib.SequenceMatcher(None, target, normalize_name(p.get("fullName"))).ratio()
        if ratio > score:
            best, score = p, ratio
    if score < 0.76:
        return {}
    return best


@st.cache_data(ttl=21600, show_spinner=False)
def get_person(person_id: int) -> Dict[str, Any]:
    data = safe_get_json(f"{MLB_BASE}/people/{person_id}", params={"hydrate": "currentTeam"}, timeout=15) or {}
    people = data.get("people") or []
    return people[0] if people else {}


@st.cache_data(ttl=180, show_spinner=False)
def get_schedule_for_date(game_date: str) -> List[Dict[str, Any]]:
    data = safe_get_json(
        f"{MLB_BASE}/schedule",
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,venue,linescore,weather",
        },
        timeout=20,
    ) or {}
    out: List[Dict[str, Any]] = []
    for day in data.get("dates") or []:
        for g in day.get("games") or []:
            away = ((g.get("teams") or {}).get("away") or {}).get("team") or {}
            home = ((g.get("teams") or {}).get("home") or {}).get("team") or {}
            away_prob = ((g.get("teams") or {}).get("away") or {}).get("probablePitcher") or {}
            home_prob = ((g.get("teams") or {}).get("home") or {}).get("probablePitcher") or {}
            out.append(
                {
                    "game_pk": g.get("gamePk"),
                    "date": game_date,
                    "status": ((g.get("status") or {}).get("detailedState") or ""),
                    "start_time": g.get("gameDate"),
                    "venue": (g.get("venue") or {}).get("name"),
                    "away_id": away.get("id"),
                    "away": away.get("abbreviation") or away.get("teamCode") or away.get("name"),
                    "home_id": home.get("id"),
                    "home": home.get("abbreviation") or home.get("teamCode") or home.get("name"),
                    "away_pitcher_id": away_prob.get("id"),
                    "away_pitcher": away_prob.get("fullName"),
                    "home_pitcher_id": home_prob.get("id"),
                    "home_pitcher": home_prob.get("fullName"),
                }
            )
    return out


def match_player_game(team_id: Any, schedule: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tid = safe_int(team_id)
    for g in schedule:
        if tid in {safe_int(g.get("away_id")), safe_int(g.get("home_id"))}:
            is_home = tid == safe_int(g.get("home_id"))
            result = dict(g)
            result["is_home"] = is_home
            result["team"] = g.get("home") if is_home else g.get("away")
            result["opponent"] = g.get("away") if is_home else g.get("home")
            result["opponent_team_id"] = g.get("away_id") if is_home else g.get("home_id")
            result["opp_pitcher_id"] = g.get("away_pitcher_id") if is_home else g.get("home_pitcher_id")
            result["opp_pitcher"] = g.get("away_pitcher") if is_home else g.get("home_pitcher")
            return result
    return {}


@st.cache_data(ttl=120, show_spinner=False)
def get_live_game_context(game_pk: Any) -> Dict[str, Any]:
    if not game_pk:
        return {}
    data = safe_get_json(f"{MLB_LIVE}/game/{game_pk}/feed/live", timeout=20) or {}
    gd = data.get("gameData") or {}
    live = data.get("liveData") or {}
    box = live.get("boxscore") or {}
    weather = gd.get("weather") or {}
    venue = gd.get("venue") or {}
    teams = gd.get("teams") or {}
    return {"raw": data, "boxscore": box, "weather": weather, "venue": venue, "teams": teams}


def confirmed_lineup_slot(game_ctx: Dict[str, Any], team_is_home: bool, player_id: int) -> Tuple[Optional[int], str]:
    side = "home" if team_is_home else "away"
    try:
        team_box = ((game_ctx.get("boxscore") or {}).get("teams") or {}).get(side) or {}
        order = team_box.get("battingOrder") or []
        ids = [safe_int(v) for v in order]
        if safe_int(player_id) in ids:
            return ids.index(safe_int(player_id)) + 1, "CONFIRMED"
    except Exception:
        pass
    return None, "PROJECTED"

# ============================================================
# UNDERDOG H+R+RBI PULL — ACTIVE PLAYERS ONLY
# ============================================================
HRR_TERMS = [
    "hits + runs + rbis", "hits+runs+rbis", "hits + runs + rbi", "hits+runs+rbi",
    "hits runs rbis", "hits runs rbi", "h+r+r", "h + r + r", "h+r+rbi", "h + r + rbi",
]
HRR_BAD = re.compile(
    r"Total Bases|Home Runs|\bRuns\s+O/U\b|\bRBIs?\s+O/U\b|Batter Strikeouts|"
    r"Batter Walks|Walks O/U|Stolen Bases|Singles|Doubles|Fantasy|Shots|Goals|Assists|"
    r"Saves|Blocks|Tackles|Strokes|Tourney|Finishing Position|Soccer|NHL|NBA|NFL|Golf|"
    r"Hockey|Basketball|Football|Pitching Outs|Strikeouts",
    re.I,
)
HRR_TITLE = re.compile(
    r"([A-Z][A-Za-zÀ-ÿ.'’\-]+(?:\s+(?:[A-Z][A-Za-zÀ-ÿ.'’\-]+|Jr\.|Sr\.|II|III|IV)){1,5})\s+"
    r"(?:Hits\s*\+\s*Runs\s*\+\s*RBIs?|H\s*\+\s*R\s*\+\s*RBIs?|H\s*\+\s*R\s*\+\s*R)\s+(?:O/U|Over/Under)",
    re.I,
)


def _ud_attrs(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out = dict(obj.get("attributes") or {})
    for k, v in obj.items():
        if k not in {"attributes", "relationships", "included", "data"} and k not in out:
            out[k] = v
    return out


def _ud_collect(obj: Any, parent: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        row = dict(obj)
        row.setdefault("_parent_key", parent)
        out.append(row)
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.extend(_ud_collect(v, str(k)))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_ud_collect(v, parent))
    return out


def _ud_text(obj: Dict[str, Any]) -> str:
    values: List[str] = []
    for d in (obj, _ud_attrs(obj)):
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)):
                values.append(f"{k} {v}")
    try:
        values.append(json.dumps(obj, default=str)[:1800])
    except Exception:
        pass
    return " | ".join(values)


def _looks_hrr(text: str) -> bool:
    raw = str(text or "")
    low = raw.lower()
    if not any(term in low for term in HRR_TERMS) and not HRR_TITLE.search(raw):
        return False
    # Remove the exact combo wording before checking neighboring single-stat markets.
    cleaned = re.sub(r"Hits\s*\+\s*Runs\s*\+\s*RBIs?|H\s*\+\s*R\s*\+\s*R(?:BI)?", "", raw, flags=re.I)
    return not HRR_BAD.search(cleaned)


def _structured_line(*objects: Optional[dict]) -> Optional[float]:
    keys = ["stat_value", "line", "over_under_line", "target_value", "line_score", "overUnderLine", "display_stat_value"]
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for d in (_ud_attrs(obj), obj):
            for k in keys:
                value = safe_float(d.get(k))
                if value is not None and 0.5 <= value <= 7.5 and abs(value * 2 - round(value * 2)) < 1e-7:
                    return float(value)
    return None


def _extract_player_name(text: str) -> str:
    m = HRR_TITLE.search(text or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(" -|•:")
    return ""


@st.cache_data(ttl=90, show_spinner=False)
def fetch_underdog_hrr_rows() -> List[Dict[str, Any]]:
    """Pull only active MLB Hits + Runs + RBIs rows from Underdog.

    Underdog has used both flattened payloads and JSON:API relationship payloads.
    This parser supports line -> over_under -> appearance -> player links using
    type+id maps, then falls back to exact H+R+RBI titles. It never takes a
    random number from the full JSON blob as a prop line.
    """
    def typ(obj: Any, fallback: str = "") -> str:
        if not isinstance(obj, dict):
            return str(fallback or "").lower().replace("-", "_")
        return str(obj.get("type") or fallback or obj.get("_parent_key", "")).lower().replace("-", "_")

    def oid(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        value = obj.get("id") or _ud_attrs(obj).get("id")
        return str(value) if value not in {None, ""} else None

    def build_maps(objects: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], dict], Dict[str, List[dict]]]:
        by_key: Dict[Tuple[str, str], dict] = {}
        by_id: Dict[str, List[dict]] = defaultdict(list)
        for obj in objects:
            object_id = oid(obj)
            if not object_id:
                continue
            object_type = typ(obj)
            for candidate in {object_type, object_type.rstrip("s"), object_type + "s"}:
                by_key[(candidate, object_id)] = obj
            by_id[object_id].append(obj)
        return by_key, by_id

    def related(obj: Optional[dict], names: Sequence[str], by_key: Dict[Tuple[str, str], dict], by_id: Dict[str, List[dict]]) -> Optional[dict]:
        if not isinstance(obj, dict):
            return None
        rels = obj.get("relationships") or {}
        for name in names:
            keys = {name, name.replace("_", "-"), name.replace("_", ""), name.rstrip("s"), name + "s"}
            for key in keys:
                node = rels.get(key)
                if node is None:
                    continue
                data = node.get("data") if isinstance(node, dict) else node
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    relation_id = item.get("id")
                    relation_type = str(item.get("type") or key).lower().replace("-", "_")
                    if relation_id in {None, ""}:
                        continue
                    relation_id = str(relation_id)
                    for candidate_type in [relation_type, relation_type.rstrip("s"), relation_type + "s", key, key.rstrip("s"), key + "s"]:
                        hit = by_key.get((candidate_type, relation_id))
                        if hit is not None:
                            return hit
                    candidates = by_id.get(relation_id, [])
                    if candidates:
                        for candidate in candidates:
                            ct = typ(candidate)
                            if key.rstrip("s") in ct or ct.rstrip("s") in key:
                                return candidate
                        return candidates[0]
        return None

    def text_from(*objects: Optional[dict]) -> str:
        wanted = [
            "title", "display_title", "name", "player_name", "full_name", "first_name", "last_name",
            "display_name", "stat", "stat_type", "appearance_stat", "display_stat", "label", "market",
            "market_name", "sport", "league", "sport_name", "league_name", "description", "over_under",
            "over_under_title", "scoring_type", "projection_type", "stat_value", "line_score",
            "appearance_name", "position",
        ]
        parts: List[str] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for d in (_ud_attrs(obj), obj):
                if not isinstance(d, dict):
                    continue
                for key in wanted:
                    value = d.get(key)
                    if isinstance(value, dict):
                        for nested_key in wanted:
                            if value.get(nested_key) not in {None, ""}:
                                parts.append(str(value.get(nested_key)))
                    elif value not in {None, ""} and not isinstance(value, (dict, list)):
                        parts.append(str(value))
        try:
            for obj in objects:
                if isinstance(obj, dict):
                    parts.append(json.dumps(obj, default=str)[:700])
        except Exception:
            pass
        return " | ".join(parts)

    def active_ok(*objects: Optional[dict]) -> bool:
        status_blob = " ".join(
            str(_ud_attrs(obj).get(key, ""))
            for obj in objects if isinstance(obj, dict)
            for key in ["status", "state", "display_status", "over_status", "under_status", "hidden", "active"]
        ).lower()
        return not any(word in status_blob for word in ["suspended", "removed", "hidden true", "inactive", "closed", "disabled"])

    def line_near_hrr(text: str) -> Optional[float]:
        match = HRR_TITLE.search(text or "")
        if not match:
            return None
        nearby = text[max(0, match.start() - 120): min(len(text), match.end() + 280)]
        values = []
        for m in re.finditer(r"(?<!\d)(\d+(?:\.5|\.0)?)(?!\d)", nearby):
            value = safe_float(m.group(1))
            if value is not None and 0.5 <= value <= 7.5 and abs(value * 2 - round(value * 2)) < 1e-7:
                values.append(value)
        half = [v for v in values if abs(v % 1 - 0.5) < 1e-7]
        return float(half[0]) if half else (float(values[0]) if values else None)

    def clean_player(*objects: Optional[dict]) -> str:
        combined = text_from(*objects)
        exact = _extract_player_name(combined)
        if exact:
            return exact
        candidates: List[str] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            attrs = _ud_attrs(obj)
            first_last = (str(attrs.get("first_name", "")).strip() + " " + str(attrs.get("last_name", "")).strip()).strip()
            for key in ["display_name", "full_name", "name", "player_name", "title", "description", "appearance_name"]:
                value = attrs.get(key)
                if isinstance(value, str):
                    candidates.append(value)
            if first_last:
                candidates.append(first_last)
        cleaned = []
        for candidate in candidates:
            name = re.sub(r"\s+(?:Hits\s*\+\s*Runs\s*\+\s*RBIs?|H\s*\+\s*R\s*\+\s*R(?:BI)?)\s+(?:O/U|Over/Under).*$", "", candidate, flags=re.I).strip(" -|•:")
            if name and len(normalize_name(name).split()) >= 2 and len(name) <= 60 and not re.search(r"\b(Hits|Runs|RBIs?|Over|Under|Batter|Line|Total|Bases)\b", name, re.I):
                cleaned.append(name)
        return sorted(set(cleaned), key=lambda x: (len(x.split()), len(x)), reverse=True)[0] if cleaned else ""

    rows: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"urls": [], "objects": 0, "line_candidates": 0, "hrr_objects": 0, "mlb_matched": 0, "samples": []}
    for url in UNDERDOG_URLS:
        data = safe_get_json(url, timeout=25)
        if not data:
            debug["urls"].append({"url": url, "status": "NO_DATA"})
            continue
        debug["urls"].append({"url": url, "status": "OK"})
        objects = _ud_collect(data)
        debug["objects"] += len(objects)
        by_key, by_id = build_maps(objects)
        candidates = []
        for obj in objects:
            attrs = _ud_attrs(obj)
            object_type = typ(obj)
            if "over_under_line" in object_type or any(attrs.get(key) not in {None, ""} for key in ["stat_value", "line_score", "over_under_line", "target_value", "line"]):
                candidates.append(obj)
        debug["line_candidates"] += len(candidates)

        for line_obj in candidates:
            ou_obj = related(line_obj, ["over_under", "over_unders"], by_key, by_id)
            appearance_obj = related(ou_obj, ["appearance", "appearances"], by_key, by_id) or related(line_obj, ["appearance", "appearances"], by_key, by_id)
            player_obj = related(appearance_obj, ["player", "players"], by_key, by_id) or related(ou_obj, ["player", "players"], by_key, by_id) or related(line_obj, ["player", "players"], by_key, by_id)
            blob = text_from(line_obj, ou_obj, appearance_obj, player_obj)
            if len(debug["samples"]) < 12 and any(x in blob.lower() for x in ["hits", "rbi", "h+r"]):
                debug["samples"].append(blob[:500])
            if not _looks_hrr(blob) or not active_ok(line_obj, ou_obj, appearance_obj, player_obj):
                continue
            debug["hrr_objects"] += 1
            prop_line = _structured_line(line_obj, ou_obj, appearance_obj)
            player_name = clean_player(player_obj, appearance_obj, ou_obj, line_obj)
            if prop_line is None or not player_name:
                continue
            person = search_mlb_person(player_name)
            if not person.get("id"):
                continue
            debug["mlb_matched"] += 1
            rows.append({
                "Source": "Underdog", "Player": person.get("fullName") or player_name,
                "Player ID": person.get("id"), "Line": float(prop_line),
                "Market": "Hits + Runs + RBIs", "Evidence": blob[:300],
            })

        # Flattened exact-title fallback.
        for obj in objects:
            blob = text_from(obj)
            if not blob or not _looks_hrr(blob):
                continue
            sport_blob = " ".join(str(_ud_attrs(obj).get(k, "")) for k in ["sport", "sport_name", "league", "league_name"]).lower()
            if any(x in sport_blob for x in ["nhl", "nba", "nfl", "soccer", "golf", "tennis", "hockey", "basketball", "football"]):
                continue
            player_name = _extract_player_name(blob)
            if not player_name:
                continue
            prop_line = _structured_line(obj)
            if prop_line is None:
                continue
            person = search_mlb_person(player_name)
            if not person.get("id"):
                continue
            debug["mlb_matched"] += 1
            rows.append({
                "Source": "Underdog", "Player": person.get("fullName") or player_name,
                "Player ID": person.get("id"), "Line": float(prop_line),
                "Market": "Hits + Runs + RBIs", "Evidence": "exact-title fallback: " + blob[:260],
            })
        if rows:
            break

    st.session_state["hrr_ud_debug"] = debug
    dedup: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        key = (normalize_name(row.get("Player")), float(row.get("Line")))
        dedup[key] = row
    return list(dedup.values())


def parse_manual_hrr_lines(text: str) -> List[Dict[str, Any]]:
    if not str(text or "").strip():
        return []
    rows: List[Dict[str, Any]] = []
    raw = str(text).strip()
    try:
        df = pd.read_csv(io.StringIO(raw))
        pcol = next((c for c in df.columns if c.strip().lower() in {"player", "name", "batter"}), None)
        lcol = next((c for c in df.columns if c.strip().lower() in {"line", "ud line", "underdog line"}), None)
        if pcol and lcol:
            for _, r in df.iterrows():
                player, line = str(r[pcol]).strip(), safe_float(r[lcol])
                if player and line is not None:
                    person = search_mlb_person(player)
                    if person.get("id"):
                        rows.append({"Source": "Manual", "Player": person.get("fullName") or player, "Player ID": person.get("id"), "Line": line, "Market": "Hits + Runs + RBIs", "Evidence": "manual CSV"})
            return rows
    except Exception:
        pass
    for line_text in raw.splitlines():
        m = re.match(r"\s*(.+?)[,|\-–—]\s*(\d+(?:\.5|\.0)?)\s*$", line_text)
        if not m:
            continue
        player, line = m.group(1).strip(" •"), safe_float(m.group(2))
        person = search_mlb_person(player)
        if line is not None and person.get("id"):
            rows.append({"Source": "Manual", "Player": person.get("fullName") or player, "Player ID": person.get("id"), "Line": line, "Market": "Hits + Runs + RBIs", "Evidence": "manual text"})
    return rows


def load_manual_lines() -> List[Dict[str, Any]]:
    if not MANUAL_LINES_FILE.exists():
        return []
    try:
        return parse_manual_hrr_lines(MANUAL_LINES_FILE.read_text())
    except Exception:
        return []

# ============================================================
# PLAYER GAME LOGS / SEASON STATS
# ============================================================
def _stats_splits(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in data.get("stats") or []:
        out.extend(block.get("splits") or [])
    return out


@st.cache_data(ttl=900, show_spinner=False)
def player_game_log(player_id: int, group: str, season: int) -> pd.DataFrame:
    data = safe_get_json(
        f"{MLB_BASE}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": group, "season": season},
        timeout=20,
    ) or {}
    rows: List[Dict[str, Any]] = []
    for s in _stats_splits(data):
        stat = s.get("stat") or {}
        row = {"Date": s.get("date"), "GamePk": (s.get("game") or {}).get("gamePk")}
        if group == "hitting":
            row.update(
                {
                    "PA": safe_float(stat.get("plateAppearances"), 0) or 0,
                    "AB": safe_float(stat.get("atBats"), 0) or 0,
                    "H": safe_float(stat.get("hits"), 0) or 0,
                    "2B": safe_float(stat.get("doubles"), 0) or 0,
                    "3B": safe_float(stat.get("triples"), 0) or 0,
                    "HR": safe_float(stat.get("homeRuns"), 0) or 0,
                    "R": safe_float(stat.get("runs"), 0) or 0,
                    "RBI": safe_float(stat.get("rbi"), 0) or 0,
                    "BB": safe_float(stat.get("baseOnBalls"), 0) or 0,
                    "IBB": safe_float(stat.get("intentionalWalks"), 0) or 0,
                    "HBP": safe_float(stat.get("hitByPitch"), 0) or 0,
                    "SO": safe_float(stat.get("strikeOuts"), 0) or 0,
                    "SF": safe_float(stat.get("sacFlies"), 0) or 0,
                    "SB": safe_float(stat.get("stolenBases"), 0) or 0,
                    "CS": safe_float(stat.get("caughtStealing"), 0) or 0,
                }
            )
            row["1B"] = max(0.0, row["H"] - row["2B"] - row["3B"] - row["HR"])
            row["HRR"] = row["H"] + row["R"] + row["RBI"]
        else:
            row.update(
                {
                    "IP": stat.get("inningsPitched"),
                    "BF": safe_float(stat.get("battersFaced"), 0) or 0,
                    "H": safe_float(stat.get("hits"), 0) or 0,
                    "2B": safe_float(stat.get("doubles"), 0) or 0,
                    "3B": safe_float(stat.get("triples"), 0) or 0,
                    "HR": safe_float(stat.get("homeRuns"), 0) or 0,
                    "BB": safe_float(stat.get("baseOnBalls"), 0) or 0,
                    "HBP": safe_float(stat.get("hitBatsmen"), 0) or 0,
                    "SO": safe_float(stat.get("strikeOuts"), 0) or 0,
                    "R": safe_float(stat.get("runs"), 0) or 0,
                    "ER": safe_float(stat.get("earnedRuns"), 0) or 0,
                    "GS": safe_float(stat.get("gamesStarted"), 0) or 0,
                    "SV": safe_float(stat.get("saves"), 0) or 0,
                    "HLD": safe_float(stat.get("holds"), 0) or 0,
                    "GF": safe_float(stat.get("gamesFinished"), 0) or 0,
                    "Pitches": safe_float(stat.get("numberOfPitches"), safe_float(stat.get("pitchesThrown"), 0)) or 0,
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(ttl=21600, show_spinner=False)
def player_season_stat(player_id: int, group: str, season: int) -> Dict[str, Any]:
    data = safe_get_json(
        f"{MLB_BASE}/people/{player_id}/stats",
        params={"stats": "season", "group": group, "season": season},
        timeout=20,
    ) or {}
    splits = _stats_splits(data)
    return (splits[0].get("stat") or {}) if splits else {}


@st.cache_data(ttl=21600, show_spinner=False)
def player_hand_split(player_id: int, group: str, season: int, opponent_hand: str) -> Dict[str, Any]:
    if opponent_hand not in {"R", "L"}:
        return {}
    sit = ("vrhp" if opponent_hand == "R" else "vlhp") if group == "hitting" else ("vr" if opponent_hand == "R" else "vl")
    data = safe_get_json(
        f"{MLB_BASE}/people/{player_id}/stats",
        params={"stats": "statSplits", "group": group, "season": season, "sitCodes": sit},
        timeout=20,
    ) or {}
    splits = _stats_splits(data)
    if not splits:
        return {}
    # Prefer the largest sample row.
    return max((s.get("stat") or {} for s in splits), key=lambda x: safe_float(x.get("plateAppearances"), safe_float(x.get("battersFaced"), 0)) or 0)


@st.cache_data(ttl=900, show_spinner=False)
def team_season_stat(team_id: int, group: str, season: int) -> Dict[str, Any]:
    data = safe_get_json(
        f"{MLB_BASE}/teams/{team_id}/stats",
        params={"stats": "season", "group": group, "season": season},
        timeout=20,
    ) or {}
    splits = _stats_splits(data)
    return (splits[0].get("stat") or {}) if splits else {}


def _logs_before(logs: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Return only games completed before the slate date (strict leakage guard)."""
    if logs.empty or "Date" not in logs:
        return logs.copy()
    cutoff = pd.Timestamp(as_of_date).normalize()
    return logs[pd.to_datetime(logs["Date"], errors="coerce") < cutoff].copy()


def _aggregate_hitting_logs(logs: pd.DataFrame) -> Dict[str, Any]:
    if logs.empty:
        return {}
    sums = {c: float(pd.to_numeric(logs.get(c, 0), errors="coerce").fillna(0).sum()) for c in [
        "PA", "AB", "H", "2B", "3B", "HR", "R", "RBI", "BB", "IBB", "HBP", "SO", "SF", "SB", "CS"
    ]}
    tb = sums["H"] + sums["2B"] + 2 * sums["3B"] + 3 * sums["HR"]
    obp_den = sums["AB"] + sums["BB"] + sums["HBP"] + sums["SF"]
    avg = sums["H"] / sums["AB"] if sums["AB"] else 0.0
    obp = (sums["H"] + sums["BB"] + sums["HBP"]) / obp_den if obp_den else LEAGUE["obp"]
    slg = tb / sums["AB"] if sums["AB"] else LEAGUE["slg"]
    return {
        "gamesPlayed": int(len(logs)), "plateAppearances": sums["PA"], "atBats": sums["AB"],
        "hits": sums["H"], "doubles": sums["2B"], "triples": sums["3B"], "homeRuns": sums["HR"],
        "runs": sums["R"], "rbi": sums["RBI"], "baseOnBalls": sums["BB"],
        "intentionalWalks": sums["IBB"], "hitByPitch": sums["HBP"], "strikeOuts": sums["SO"],
        "sacFlies": sums["SF"], "stolenBases": sums["SB"], "caughtStealing": sums["CS"],
        "avg": round(avg, 4), "obp": round(obp, 4), "slg": round(slg, 4), "ops": round(obp + slg, 4),
    }


def _aggregate_pitching_logs(logs: pd.DataFrame) -> Dict[str, Any]:
    if logs.empty:
        return {}
    cols = ["BF", "H", "2B", "3B", "HR", "BB", "HBP", "SO", "R", "ER", "Pitches", "GS", "SV", "HLD", "GF"]
    sums = {c: float(pd.to_numeric(logs.get(c, 0), errors="coerce").fillna(0).sum()) for c in cols}
    innings = float(sum(innings_to_float(v) for v in logs.get("IP", pd.Series(dtype=object)).tolist()))
    return {
        "gamesPlayed": int(len(logs)), "gamesPitched": int(len(logs)), "gamesStarted": sums["GS"],
        "battersFaced": sums["BF"], "hits": sums["H"], "doubles": sums["2B"], "triples": sums["3B"],
        "homeRuns": sums["HR"], "baseOnBalls": sums["BB"], "hitBatsmen": sums["HBP"],
        "strikeOuts": sums["SO"], "runs": sums["R"], "earnedRuns": sums["ER"],
        "inningsPitched": round(innings, 2), "numberOfPitches": sums["Pitches"],
        "saves": sums["SV"], "holds": sums["HLD"], "gamesFinished": sums["GF"],
    }


@st.cache_data(ttl=900, show_spinner=False)
def player_asof_stat(player_id: int, group: str, season: int, as_of_date: str) -> Dict[str, Any]:
    logs = _logs_before(player_game_log(player_id, group, season), as_of_date)
    return _aggregate_hitting_logs(logs) if group == "hitting" else _aggregate_pitching_logs(logs)


@st.cache_data(ttl=1800, show_spinner=False)
def team_asof_stat(team_id: int, group: str, season: int, opening_day: str, as_of_date: str) -> Dict[str, Any]:
    """Date-range team stats. Never fall back to full-season data for an old slate."""
    end_date = (pd.Timestamp(as_of_date).date() - timedelta(days=1)).isoformat()
    if end_date < opening_day:
        return {}
    data = safe_get_json(
        f"{MLB_BASE}/teams/{team_id}/stats",
        params={"stats": "byDateRange", "group": group, "season": season, "startDate": opening_day, "endDate": end_date},
        timeout=25,
    ) or {}
    splits = _stats_splits(data)
    if splits:
        return splits[0].get("stat") or {}
    # A current-day fallback is safe because the season endpoint only contains completed games.
    if pd.Timestamp(as_of_date).date() >= la_now().date():
        return team_season_stat(team_id, group, season)
    return {}

# ============================================================
# STATCAST — ACTIVE BOARD PLAYERS/PITCHERS ONLY
# ============================================================
def _read_savant(params: Dict[str, Any], timeout: int = 35) -> pd.DataFrame:
    try:
        r = requests.get(SAVANT_CSV, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip():
            append_request_log(SAVANT_CSV, f"HTTP {r.status_code}", r.text[:200])
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as exc:
        append_request_log(SAVANT_CSV, "REQUEST_ERROR", str(exc))
        return pd.DataFrame()


def _statcast_common(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"available": False, "rows": 0}
    desc = df.get("description", pd.Series(index=df.index, dtype=str)).astype(str).str.lower()
    swings = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt", "missed_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"])
    whiffs = desc.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
    zone = pd.to_numeric(df.get("zone"), errors="coerce") if "zone" in df else pd.Series(index=df.index, dtype=float)
    out_zone = zone.isin([11, 12, 13, 14])
    in_zone = zone.isin(range(1, 10))
    launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce") if "launch_speed" in df else pd.Series(index=df.index, dtype=float)
    launch_angle = pd.to_numeric(df.get("launch_angle"), errors="coerce") if "launch_angle" in df else pd.Series(index=df.index, dtype=float)
    barrel = pd.to_numeric(df.get("barrel"), errors="coerce") if "barrel" in df else pd.Series(index=df.index, dtype=float)
    bbe = launch_speed.notna()
    result = {
        "available": True,
        "rows": int(len(df)),
        "swings": int(swings.sum()),
        "whiff_pct": float(whiffs.sum() / swings.sum()) if swings.sum() else None,
        "contact_pct": float(1 - whiffs.sum() / swings.sum()) if swings.sum() else None,
        "chase_pct": float((out_zone & swings).sum() / out_zone.sum()) if out_zone.sum() else None,
        "zone_contact_pct": float(1 - (in_zone & whiffs).sum() / (in_zone & swings).sum()) if (in_zone & swings).sum() else None,
        "avg_ev": float(launch_speed[bbe].mean()) if bbe.sum() else None,
        "hard_hit_pct": float((launch_speed[bbe] >= 95).mean()) if bbe.sum() else None,
        "sweet_spot_pct": float(launch_angle[bbe].between(8, 32).mean()) if bbe.sum() else None,
        "barrel_pct": float(barrel[bbe].fillna(0).mean()) if bbe.sum() else None,
    }
    for col, key in [
        ("estimated_ba_using_speedangle", "xba"),
        ("estimated_woba_using_speedangle", "xwoba"),
        ("estimated_slg_using_speedangle", "xslg"),
    ]:
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            result[key] = float(vals.mean()) if len(vals) else None
        else:
            result[key] = None
    return result


@st.cache_data(ttl=21600, show_spinner=False)
def batter_statcast_profile(player_id: int, start_date: str, end_date: str, pitcher_hand: Optional[str] = None) -> Dict[str, Any]:
    query_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date().isoformat()
    params = {
        "all": "true", "player_type": "batter", "batters_lookup[]": str(player_id),
        "game_date_gt": query_start, "game_date_lt": end_date, "type": "details",
    }
    df = _read_savant(params)
    if pitcher_hand in {"R", "L"} and not df.empty and "p_throws" in df:
        split = df[df["p_throws"].astype(str).str.upper().eq(pitcher_hand)]
        if len(split) >= 75:
            df = split
    result = _statcast_common(df)
    if df.empty:
        return result
    events = df.get("events", pd.Series(index=df.index, dtype=str)).astype(str).str.lower()
    pa_events = events[~events.isin(["", "nan", "none"])]
    counts = Counter(pa_events)
    result["event_pa"] = int(len(pa_events))
    result["events"] = dict(counts)
    # Batter pitch-type performance for arsenal interaction.
    pitch_rows = []
    if "pitch_type" in df:
        for pitch_type, g in df.groupby("pitch_type"):
            if len(g) < 15:
                continue
            base = _statcast_common(g)
            ev = g.get("events", pd.Series(index=g.index, dtype=str)).astype(str).str.lower()
            hit_n = ev.isin(["single", "double", "triple", "home_run"]).sum()
            ab_n = (~ev.isin(["", "nan", "none", "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"])).sum()
            pitch_rows.append({"pitch_type": pitch_type, "pitches": len(g), "whiff_pct": base.get("whiff_pct"), "avg_ev": base.get("avg_ev"), "hit_rate": float(hit_n / ab_n) if ab_n else None})
    result["pitch_types"] = pitch_rows
    return result


@st.cache_data(ttl=21600, show_spinner=False)
def pitcher_statcast_profile(player_id: int, start_date: str, end_date: str, batter_side: Optional[str] = None) -> Dict[str, Any]:
    query_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date().isoformat()
    params = {
        "all": "true", "player_type": "pitcher", "pitchers_lookup[]": str(player_id),
        "game_date_gt": query_start, "game_date_lt": end_date, "type": "details",
    }
    df = _read_savant(params)
    if batter_side in {"R", "L"} and not df.empty and "stand" in df:
        split = df[df["stand"].astype(str).str.upper().eq(batter_side)]
        if len(split) >= 75:
            df = split
    result = _statcast_common(df)
    if df.empty:
        return result
    events = df.get("events", pd.Series(index=df.index, dtype=str)).astype(str).str.lower()
    pa_events = events[~events.isin(["", "nan", "none"])]
    result["event_pa"] = int(len(pa_events))
    result["events"] = dict(Counter(pa_events))
    if "pitch_type" in df:
        mix = df["pitch_type"].dropna().astype(str).value_counts(normalize=True)
        result["pitch_mix"] = {k: float(v) for k, v in mix.items()}
    else:
        result["pitch_mix"] = {}
    return result


def statcast_event_rates(profile: Dict[str, Any]) -> Dict[str, float]:
    events = {str(k).lower(): safe_float(v, 0) or 0 for k, v in (profile.get("events") or {}).items()}
    pa = safe_float(profile.get("event_pa"), 0) or 0
    if pa <= 0:
        return {}
    return {
        "pa": pa,
        "bb_pa": (events.get("walk", 0) + events.get("intent_walk", 0)) / pa,
        "hbp_pa": events.get("hit_by_pitch", 0) / pa,
        "k_pa": (events.get("strikeout", 0) + events.get("strikeout_double_play", 0)) / pa,
        "single_pa": events.get("single", 0) / pa,
        "double_pa": events.get("double", 0) / pa,
        "triple_pa": events.get("triple", 0) / pa,
        "hr_pa": events.get("home_run", 0) / pa,
    }


def historical_statcast_record(player_id: int, player_name: str, player_type: str, start_year: int = 2021, end_year: int = 2025) -> Dict[str, Any]:
    lookup_key = "batters_lookup[]" if player_type == "batter" else "pitchers_lookup[]"
    params = {
        "all": "true", "player_type": player_type, lookup_key: str(player_id),
        "game_date_gt": f"{start_year}-03-01", "game_date_lt": f"{end_year}-11-30", "type": "details",
    }
    df = _read_savant(params, timeout=90)
    if df.empty:
        return {}
    summary = _statcast_common(df)
    events = df.get("events", pd.Series(index=df.index, dtype=str)).astype(str).str.lower()
    pa_events = events[~events.isin(["", "nan", "none"])]
    summary["event_pa"] = int(len(pa_events))
    summary["events"] = dict(Counter(pa_events))
    rates = statcast_event_rates(summary)
    record: Dict[str, Any] = {
        "player_id": int(player_id), "player_name": player_name, "player_type": player_type,
        "start_year": start_year, "end_year": end_year, "pitch_rows": int(len(df)),
        "sample_size": int(summary.get("event_pa") or 0), "updated_at": now_iso(),
    }
    for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
        record[key] = safe_float(rates.get(key))
    for key in ["xba", "xwoba", "xslg", "avg_ev", "hard_hit_pct", "sweet_spot_pct", "barrel_pct", "whiff_pct", "contact_pct", "chase_pct", "zone_contact_pct"]:
        record[key] = safe_float(summary.get(key))
    return record


def _merge_profile_records(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    valid = [dict(r) for r in records if r and safe_int(r.get("player_id"))]
    if not valid:
        return 0
    old = safe_read_table(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(valid)
    merged = pd.concat([old, new], ignore_index=True, sort=False) if not old.empty else new
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")
    merged = merged.dropna(subset=["player_id"]).sort_values("updated_at" if "updated_at" in merged else "player_id")
    merged = merged.drop_duplicates(subset=["player_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    merged.to_csv(csv_path, index=False)
    return len(valid)


def build_targeted_historical_profiles(board: pd.DataFrame, include_bullpen: bool = True, max_pitchers: int = 40) -> Dict[str, Any]:
    if board is None or board.empty:
        return {"batters": 0, "pitchers": 0, "errors": ["Build a board first"]}
    existing_b = historical_batter_profiles()
    existing_p = historical_pitcher_profiles()
    have_b = set(pd.to_numeric(existing_b.get("player_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()) if not existing_b.empty else set()
    have_p = set(pd.to_numeric(existing_p.get("player_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()) if not existing_p.empty else set()
    batter_pairs = []
    for _, row in board[[c for c in ["Player ID", "Player"] if c in board.columns]].drop_duplicates().iterrows():
        pid = safe_int(row.get("Player ID"))
        if pid and pid not in have_b:
            batter_pairs.append((pid, str(row.get("Player") or get_person(pid).get("fullName") or pid)))
    pitcher_pairs: List[Tuple[int, str]] = []
    for _, row in board.iterrows():
        pid = safe_int(row.get("Pitcher ID"))
        if pid and pid not in have_p:
            pitcher_pairs.append((pid, str(row.get("Pitcher") or get_person(pid).get("fullName") or pid)))
        if include_bullpen:
            raw = row.get("Bullpen IDs")
            ids: List[int] = []
            if isinstance(raw, str):
                try:
                    ids = [safe_int(v) for v in json.loads(raw)]
                except Exception:
                    ids = [safe_int(v) for v in re.findall(r"\d+", raw)]
            elif isinstance(raw, (list, tuple)):
                ids = [safe_int(v) for v in raw]
            for rid in ids:
                if rid and rid not in have_p:
                    pitcher_pairs.append((rid, str(get_person(rid).get("fullName") or rid)))
    # Preserve order while de-duplicating and cap relief calls.
    seen: set = set()
    deduped_pitchers: List[Tuple[int, str]] = []
    for pid, name in pitcher_pairs:
        if pid in seen:
            continue
        seen.add(pid)
        deduped_pitchers.append((pid, name))
    pitcher_pairs = deduped_pitchers[:max_pitchers]
    b_records, p_records, errors = [], [], []
    for pid, name in batter_pairs:
        try:
            rec = historical_statcast_record(pid, name, "batter")
            if rec:
                b_records.append(rec)
        except Exception as exc:
            errors.append(f"Batter {name}: {exc}")
    for pid, name in pitcher_pairs:
        try:
            rec = historical_statcast_record(pid, name, "pitcher")
            if rec:
                p_records.append(rec)
        except Exception as exc:
            errors.append(f"Pitcher {name}: {exc}")
    b_path = CACHE_DIR / "batter_profiles_2021_2025.csv"
    p_path = CACHE_DIR / "pitcher_profiles_2021_2025.csv"
    b_added = _merge_profile_records(b_path, b_records)
    p_added = _merge_profile_records(p_path, p_records)
    historical_batter_profiles.clear()
    historical_pitcher_profiles.clear()
    write_json(PROFILE_BUILD_STATE, {"updated": now_iso(), "batters_added": b_added, "pitchers_added": p_added, "errors": errors[-50:]}, protect=False)
    return {"batters": b_added, "pitchers": p_added, "errors": errors}


# ============================================================
# BULLPEN / WEATHER / LINEUP OPPORTUNITY
# ============================================================
def innings_to_float(value: Any) -> float:
    try:
        s = str(value or "0")
        if "." not in s:
            return float(s)
        whole, frac = s.split(".", 1)
        outs = int(frac[:1]) if frac else 0
        return int(whole) + (outs / 3 if outs in {0, 1, 2} else float("0." + frac))
    except Exception:
        return 0.0


@st.cache_data(ttl=1800, show_spinner=False)
def active_team_roster(team_id: int, as_of_date: str) -> List[Dict[str, Any]]:
    data = safe_get_json(
        f"{MLB_BASE}/teams/{team_id}/roster",
        params={"rosterType": "active", "date": as_of_date, "hydrate": "person"},
        timeout=20,
    ) or {}
    rows: List[Dict[str, Any]] = []
    for entry in data.get("roster") or []:
        person = entry.get("person") or {}
        position = entry.get("position") or {}
        rows.append({
            "player_id": safe_int(person.get("id")),
            "name": person.get("fullName") or "",
            "position_type": position.get("type") or "",
            "position_code": position.get("code") or "",
            "status": ((entry.get("status") or {}).get("description") or "Active"),
        })
    return rows


@st.cache_data(ttl=600, show_spinner=False)
def recent_reliever_usage(team_id: int, as_of_date: str, days: int = 4) -> Dict[str, Any]:
    """Pregame-only bullpen workload ending the day before the slate."""
    end = pd.Timestamp(as_of_date).date() - timedelta(days=1)
    start = end - timedelta(days=max(days - 1, 0))
    if end < start:
        return {"by_pitcher": {}, "pitches_3d": 0.0, "innings_3d": 0.0, "appearances": 0, "fatigue_factor": 1.0}
    data = safe_get_json(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "teamId": team_id, "startDate": start.isoformat(), "endDate": end.isoformat(), "gameTypes": "R"},
        timeout=25,
    ) or {}
    games: List[Tuple[str, int]] = []
    for day in data.get("dates") or []:
        for game in day.get("games") or []:
            if game.get("gamePk"):
                games.append((str(day.get("date")), int(game["gamePk"])))
    by_pitcher: Dict[str, Dict[str, Any]] = {}
    total_pitches = total_innings = 0.0
    appearances = 0
    for game_date, game_pk in games:
        box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15) or {}
        for side in ["away", "home"]:
            team = ((box.get("teams") or {}).get(side) or {})
            if safe_int((team.get("team") or {}).get("id")) != safe_int(team_id):
                continue
            pitcher_ids = [safe_int(v) for v in (team.get("pitchers") or []) if safe_int(v)]
            for idx, pid in enumerate(pitcher_ids):
                if idx == 0:  # starting pitcher
                    continue
                player = (team.get("players") or {}).get(f"ID{pid}") or {}
                stat = ((player.get("stats") or {}).get("pitching") or {})
                pitches = safe_float(stat.get("pitchesThrown"), safe_float(stat.get("numberOfPitches"), 0)) or 0
                innings = innings_to_float(stat.get("inningsPitched"))
                rec = by_pitcher.setdefault(str(pid), {
                    "player_id": pid,
                    "name": ((player.get("person") or {}).get("fullName") or ""),
                    "appearances": 0,
                    "pitches": 0.0,
                    "innings": 0.0,
                    "dates": [],
                    "pitches_by_date": {},
                })
                rec["appearances"] += 1
                rec["pitches"] += pitches
                rec["innings"] += innings
                rec["dates"].append(game_date)
                rec["pitches_by_date"][game_date] = rec["pitches_by_date"].get(game_date, 0.0) + pitches
                total_pitches += pitches
                total_innings += innings
                appearances += 1
    yesterday = (pd.Timestamp(as_of_date).date() - timedelta(days=1)).isoformat()
    two_days = (pd.Timestamp(as_of_date).date() - timedelta(days=2)).isoformat()
    three_days = (pd.Timestamp(as_of_date).date() - timedelta(days=3)).isoformat()
    for rec in by_pitcher.values():
        pbd = rec.get("pitches_by_date") or {}
        rec["pitches_1d"] = round(float(pbd.get(yesterday, 0.0)), 1)
        rec["pitches_2d"] = round(float(pbd.get(yesterday, 0.0) + pbd.get(two_days, 0.0)), 1)
        rec["pitches_3d"] = round(float(pbd.get(yesterday, 0.0) + pbd.get(two_days, 0.0) + pbd.get(three_days, 0.0)), 1)
        consecutive = bool(pbd.get(yesterday, 0) and pbd.get(two_days, 0))
        availability = 1.0
        if rec["pitches_1d"] >= 35:
            availability *= 0.08
        elif rec["pitches_1d"] >= 25:
            availability *= 0.30
        elif rec["pitches_1d"] >= 18:
            availability *= 0.62
        if consecutive:
            availability *= 0.42
        if rec["pitches_3d"] >= 55:
            availability *= 0.65
        rec["consecutive_days"] = consecutive
        rec["availability"] = round(clamp(availability, 0.03, 1.0), 3)
    fatigue = clamp((total_pitches - 105) / 150, -0.10, 0.18)
    return {
        "by_pitcher": by_pitcher,
        "pitches_3d": round(total_pitches, 1),
        "innings_3d": round(total_innings, 2),
        "appearances": appearances,
        "repeat_relievers": sum(1 for v in by_pitcher.values() if len(set(v.get("dates") or [])) >= 2),
        "fatigue_factor": round(1 + fatigue, 3),
    }


@st.cache_data(ttl=600, show_spinner=False)
def bullpen_recent_workload(team_id: int, as_of: str, days: int = 3) -> Dict[str, Any]:
    return recent_reliever_usage(team_id, as_of, max(3, days))


@st.cache_data(ttl=3600, show_spinner=False)
def park_factor_overrides() -> Dict[str, Dict[str, float]]:
    for path in PARK_FACTOR_FILE_CANDIDATES:
        if not path.exists():
            continue
        df = safe_read_table(path)
        if df.empty:
            continue
        out: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            venue = str(row.get("Venue") or row.get("venue") or row.get("Park") or "").strip()
            if not venue:
                continue
            out[venue] = {
                "1B": clamp(safe_float(row.get("1B"), 1.0) or 1.0, 0.82, 1.20),
                "2B": clamp(safe_float(row.get("2B"), 1.0) or 1.0, 0.78, 1.28),
                "3B": clamp(safe_float(row.get("3B"), 1.0) or 1.0, 0.70, 1.35),
                "HR": clamp(safe_float(row.get("HR"), 1.0) or 1.0, 0.78, 1.25),
                "R": clamp(safe_float(row.get("R"), 1.0) or 1.0, 0.82, 1.20),
            }
        if out:
            return out
    return {}


def get_park_factors(venue: str) -> Dict[str, float]:
    override = park_factor_overrides().get(str(venue or ""))
    base = override or PARK_FACTORS.get(str(venue or "")) or {"1B": 1.0, "2B": 1.0, "3B": 1.0, "HR": 1.0, "R": 1.0}
    return {k: float(clamp(safe_float(base.get(k), 1.0) or 1.0, 0.78 if k != "3B" else 0.70, 1.28 if k != "3B" else 1.35)) for k in ["1B", "2B", "3B", "HR", "R"]}


def _parse_mlb_wind(wind_text: str) -> Tuple[float, Optional[float]]:
    text = str(wind_text or "")
    mph_match = re.search(r"(\d+(?:\.\d+)?)\s*mph", text, re.I)
    mph = safe_float(mph_match.group(1), 0.0) if mph_match else 0.0
    # MLB sometimes provides only descriptive direction. Direction degrees are
    # supplied by Open-Meteo when available.
    return float(mph or 0.0), None


@st.cache_data(ttl=900, show_spinner=False)
def open_meteo_weather(venue: str, game_start_time: Optional[str], game_date: str) -> Dict[str, Any]:
    meta = STADIUM_META.get(str(venue or "")) or {}
    if not meta or not game_start_time:
        return {}
    try:
        game_ts = pd.Timestamp(game_start_time)
        if game_ts.tzinfo is None:
            game_ts = game_ts.tz_localize("UTC")
        else:
            game_ts = game_ts.tz_convert("UTC")
        today = la_now().date()
        target_date = pd.Timestamp(game_date).date()
        if target_date < today - timedelta(days=5):
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": meta["lat"], "longitude": meta["lon"],
            "start_date": target_date.isoformat(), "end_date": target_date.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
        }
        data = safe_get_json(url, params=params, timeout=20) or {}
        hourly = data.get("hourly") or {}
        times = pd.to_datetime(hourly.get("time") or [], utc=True, errors="coerce")
        if len(times) == 0:
            return {}
        diffs = np.abs((times - game_ts).total_seconds())
        idx = int(np.nanargmin(diffs))
        def val(key: str, default: Optional[float] = None) -> Optional[float]:
            arr = hourly.get(key) or []
            return safe_float(arr[idx], default) if idx < len(arr) else default
        return {
            "temperature": val("temperature_2m"),
            "humidity": val("relative_humidity_2m"),
            "pressure": val("surface_pressure"),
            "wind_speed": val("wind_speed_10m", 0.0),
            "wind_direction": val("wind_direction_10m"),
            "precipitation": val("precipitation", 0.0),
            "source": "Open-Meteo archive" if "archive" in url else "Open-Meteo forecast",
        }
    except Exception as exc:
        append_request_log("Open-Meteo", "WEATHER_ERROR", str(exc))
        return {}


def parse_weather_factor(weather: Dict[str, Any], venue: str, game_start_time: Optional[str] = None, game_date: Optional[str] = None) -> Dict[str, Any]:
    meta = STADIUM_META.get(str(venue or "")) or {}
    om = open_meteo_weather(venue, game_start_time, game_date or la_now().date().isoformat()) if game_start_time else {}
    temp = safe_float(om.get("temperature"), safe_float(weather.get("temp"), 72)) or 72
    humidity = safe_float(om.get("humidity"), 50) or 50
    pressure = safe_float(om.get("pressure"), 1013.0) or 1013.0
    wind_text = str(weather.get("wind") or "")
    mlb_mph, _ = _parse_mlb_wind(wind_text)
    mph = safe_float(om.get("wind_speed"), mlb_mph) or 0.0
    direction = safe_float(om.get("wind_direction"))
    condition = str(weather.get("condition") or "Unknown")
    roof_type = str(meta.get("roof") or "open")
    low = f"{condition} {wind_text}".lower()
    roof_closed = roof_type == "fixed" or any(x in low for x in ["roof closed", "dome", "indoors", "indoor"])

    thermal = clamp((temp - 72) * 0.00125, -0.028, 0.040)
    pressure_adj = clamp((1013.0 - pressure) * 0.000075, -0.018, 0.020)
    humidity_adj = clamp((humidity - 50) * 0.00005, -0.004, 0.004)
    wind_alignment = 0.0
    if direction is not None and meta.get("bearing") is not None:
        # Meteorological direction is where wind comes FROM; add 180 to get travel direction.
        toward = (float(direction) + 180.0) % 360.0
        wind_alignment = math.cos(math.radians(toward - float(meta["bearing"])))
    elif "out" in low:
        wind_alignment = 0.8
    elif "in" in low:
        wind_alignment = -0.8
    wind_adj = clamp(float(mph) * wind_alignment * 0.0032, -0.065, 0.065)
    if roof_closed:
        carry = 1.0
        wind_adj = 0.0
        thermal *= 0.15
        pressure_adj *= 0.15
    else:
        carry = 1.0 + thermal + pressure_adj + humidity_adj + wind_adj
    carry = clamp(carry, 0.92, 1.09)
    # Singles react less to carry than extra-base events.
    return {
        "temp": round(temp, 1), "humidity": round(humidity, 1), "pressure": round(pressure, 1),
        "wind": wind_text or (f"{mph:.0f} mph @ {direction:.0f}°" if direction is not None else f"{mph:.0f} mph"),
        "wind_speed": round(float(mph), 1), "wind_direction": round(float(direction), 1) if direction is not None else None,
        "wind_alignment": round(wind_alignment, 3), "condition": condition or "Unknown",
        "roof_status": "CLOSED" if roof_closed else "OPEN/OUTDOOR",
        "carry_factor": round(carry, 3),
        "1B_factor": round(clamp(1 + (carry - 1) * 0.10, 0.99, 1.01), 3),
        "2B_factor": round(clamp(1 + (carry - 1) * 0.45, 0.96, 1.05), 3),
        "3B_factor": round(clamp(1 + (carry - 1) * 0.40, 0.96, 1.05), 3),
        "HR_factor": round(carry, 3),
        "precipitation": safe_float(om.get("precipitation"), 0.0),
        "source": om.get("source") or "MLB game weather",
    }


@st.cache_data(ttl=21600, show_spinner=False)
def recent_lineup_slots(player_id: int, season: int, limit: int = 5, as_of_date: Optional[str] = None) -> List[int]:
    logs = player_game_log(player_id, "hitting", season)
    if as_of_date:
        logs = _logs_before(logs, as_of_date)
    if logs.empty or "GamePk" not in logs:
        return []
    slots: List[int] = []
    for game_pk in logs["GamePk"].dropna().astype(int).tolist()[-max(limit * 2, 6):][::-1]:
        box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15) or {}
        found = False
        for side in ["away", "home"]:
            order = ((((box.get("teams") or {}).get(side) or {}).get("battingOrder")) or [])
            ids = [safe_int(v) for v in order]
            if safe_int(player_id) in ids:
                slots.append(ids.index(safe_int(player_id)) + 1)
                found = True
                break
        if found and len(slots) >= limit:
            break
    return slots


def confirmed_lineup_ids(game_ctx: Dict[str, Any], team_is_home: bool) -> List[int]:
    side = "home" if team_is_home else "away"
    team_box = (((game_ctx.get("boxscore") or {}).get("teams") or {}).get(side) or {})
    return [safe_int(v) for v in (team_box.get("battingOrder") or []) if safe_int(v)]


@st.cache_data(ttl=1800, show_spinner=False)
def last_known_team_lineup(team_id: int, season: int, as_of_date: str) -> List[int]:
    end = pd.Timestamp(as_of_date).date() - timedelta(days=1)
    start = end - timedelta(days=14)
    data = safe_get_json(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "teamId": team_id, "startDate": start.isoformat(), "endDate": end.isoformat(), "gameTypes": "R"},
        timeout=20,
    ) or {}
    games = [(d.get("date"), g.get("gamePk")) for d in data.get("dates") or [] for g in d.get("games") or [] if g.get("gamePk")]
    for _, game_pk in games[::-1]:
        box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15) or {}
        for side in ["away", "home"]:
            team_box = (((box.get("teams") or {}).get(side) or {}))
            if safe_int((team_box.get("team") or {}).get("id")) == safe_int(team_id):
                ids = [safe_int(v) for v in (team_box.get("battingOrder") or []) if safe_int(v)]
                if len(ids) >= 9:
                    return ids[:9]
    return []


def resolve_lineup_ids(game_ctx: Dict[str, Any], team_id: int, team_is_home: bool, season: int, as_of_date: str, target_player_id: int, target_slot: int) -> Tuple[List[int], str]:
    ids = confirmed_lineup_ids(game_ctx, team_is_home)
    status = "CONFIRMED" if len(ids) >= 9 else "PROJECTED"
    if len(ids) < 9:
        ids = last_known_team_lineup(team_id, season, as_of_date)
    ids = [safe_int(v) for v in ids if safe_int(v)]
    target = safe_int(target_player_id)
    if target in ids:
        ids.remove(target)
    insert_at = int(clamp(target_slot, 1, 9)) - 1
    ids.insert(min(insert_at, len(ids)), target)
    # De-duplicate and keep target at the projected/confirmed position.
    cleaned: List[int] = []
    for pid in ids:
        if pid and pid not in cleaned:
            cleaned.append(pid)
    ids = cleaned[:9]
    # If the previous lineup is incomplete, fill with active position players.
    if len(ids) < 9:
        for row in active_team_roster(team_id, as_of_date):
            if row.get("position_type") == "Pitcher":
                continue
            pid = safe_int(row.get("player_id"))
            if pid and pid not in ids:
                ids.append(pid)
            if len(ids) >= 9:
                break
    return ids[:9], status


def projected_pa(slot: int, is_home: bool, team_runs: float, lineup_obp: float = LEAGUE["obp"]) -> Tuple[float, Dict[int, float]]:
    base = {1: 4.72, 2: 4.62, 3: 4.53, 4: 4.43, 5: 4.31, 6: 4.18, 7: 4.06, 8: 3.94, 9: 3.82}.get(int(slot), 4.20)
    base += clamp((team_runs - LEAGUE["runs_per_game"]) * 0.12, -0.25, 0.30)
    base += clamp((lineup_obp - LEAGUE["obp"]) * 1.8, -0.10, 0.12)
    if is_home:
        base -= 0.08
    mean = clamp(base, 3.25, 5.25)
    values = np.array([3, 4, 5, 6], dtype=float)
    sigma = 0.68
    weights = np.exp(-0.5 * ((values - mean) / sigma) ** 2)
    weights /= weights.sum()
    return round(mean, 2), {int(v): float(w) for v, w in zip(values, weights)}

# ============================================================
# BAYESIAN PLAYER / PITCHER PROFILES
# ============================================================
def _rate(stat: Dict[str, Any], num_key: str, denom_key: str = "plateAppearances") -> Optional[float]:
    num, den = safe_float(stat.get(num_key), 0) or 0, safe_float(stat.get(denom_key), 0) or 0
    return num / den if den > 0 else None


def _batter_rates_from_stat(stat: Dict[str, Any]) -> Dict[str, float]:
    pa = safe_float(stat.get("plateAppearances"), 0) or 0
    h = safe_float(stat.get("hits"), 0) or 0
    d2 = safe_float(stat.get("doubles"), 0) or 0
    d3 = safe_float(stat.get("triples"), 0) or 0
    hr = safe_float(stat.get("homeRuns"), 0) or 0
    return {
        "pa": pa,
        "bb_pa": (safe_float(stat.get("baseOnBalls"), 0) or 0) / pa if pa else LEAGUE["bb_pa"],
        "hbp_pa": (safe_float(stat.get("hitByPitch"), 0) or 0) / pa if pa else LEAGUE["hbp_pa"],
        "k_pa": (safe_float(stat.get("strikeOuts"), 0) or 0) / pa if pa else LEAGUE["k_pa"],
        "single_pa": max(0.0, h - d2 - d3 - hr) / pa if pa else LEAGUE["single_pa"],
        "double_pa": d2 / pa if pa else LEAGUE["double_pa"],
        "triple_pa": d3 / pa if pa else LEAGUE["triple_pa"],
        "hr_pa": hr / pa if pa else LEAGUE["hr_pa"],
        "obp": safe_float(stat.get("obp"), LEAGUE["obp"]) or LEAGUE["obp"],
        "slg": safe_float(stat.get("slg"), LEAGUE["slg"]) or LEAGUE["slg"],
        "ops": safe_float(stat.get("ops"), LEAGUE["ops"]) or LEAGUE["ops"],
    }


def beta_blend(current: float, n_current: float, prior: float, prior_strength: float) -> float:
    n = max(0.0, float(n_current))
    return (current * n + prior * prior_strength) / max(n + prior_strength, 1e-9)


def offline_profile_for(df: pd.DataFrame, player_id: int, player_name: str) -> Dict[str, Any]:
    """Combine name-based career priors with ID-based Statcast profiles."""
    if df.empty:
        return {}
    target_name = normalize_name(player_name)
    name_matches = pd.DataFrame()
    for c in ["normalized_name", "player_name", "Player", "Name", "full_name"]:
        if c not in df.columns:
            continue
        values = df[c].astype(str) if c == "normalized_name" else df[c].astype(str).map(normalize_name)
        name_matches = df[values.eq(target_name)]
        if not name_matches.empty:
            break
    id_matches = pd.DataFrame()
    for c in ["player_id", "Player ID", "mlbam_id", "batter", "pitcher"]:
        if c in df.columns:
            id_matches = df[pd.to_numeric(df[c], errors="coerce").eq(player_id)]
            if not id_matches.empty:
                break
    records: List[Dict[str, Any]] = []
    # Career/name profile first, then MLB-ID Statcast profile can override rates.
    for frame in [name_matches, id_matches]:
        if frame.empty:
            continue
        sort_cols = [c for c in ["end_year", "updated_at", "historical_pa", "sample_size"] if c in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols, na_position="first")
        records.extend(frame.to_dict("records"))
    if not records:
        return {}
    combined: Dict[str, Any] = {}
    sources: List[str] = []
    for record in records:
        source = str(record.get("_profile_source") or record.get("profile_source") or "").strip()
        if source and source not in sources:
            sources.append(source)
        for key, value in record.items():
            if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).lower() in {"nan", "none"}:
                continue
            combined[key] = value
    combined["_profile_sources"] = " | ".join(sources)
    return combined


def historical_prior_rates(profile: Dict[str, Any], fallback_2025: Dict[str, float]) -> Dict[str, float]:
    """Blend the latest-season prior with the uploaded multi-season prior.

    The uploaded 2015-2024 file strengthens true talent, but it does not erase
    2025. When 2025 has no PA, the historical profile becomes the primary prior.
    """
    aliases = {
        "bb_pa": ["bb_pa", "BB%", "bb_rate"], "hbp_pa": ["hbp_pa", "HBP%"],
        "k_pa": ["k_pa", "K%", "k_rate"], "single_pa": ["single_pa", "1B/PA", "single_rate"],
        "double_pa": ["double_pa", "2B/PA", "double_rate"], "triple_pa": ["triple_pa", "3B/PA", "triple_rate"],
        "hr_pa": ["hr_pa", "HR/PA", "hr_rate"],
    }
    out = dict(fallback_2025)
    prior_pa = safe_float(fallback_2025.get("pa"), safe_float(fallback_2025.get("bf"), 0)) or 0
    historical_pa = safe_float(profile.get("effective_pa"), safe_float(profile.get("historical_pa"), safe_float(profile.get("career_pa"), safe_float(profile.get("sample_size"), 0)))) or 0
    historical_equivalent = clamp(math.sqrt(max(historical_pa, 0)) * 3.0, 35, 225) if historical_pa > 0 else 0
    recent_share = prior_pa / max(prior_pa + historical_equivalent, 1e-9) if prior_pa > 0 else 0.0
    recent_share = clamp(recent_share, 0.20, 0.86) if prior_pa > 0 and historical_equivalent > 0 else (1.0 if prior_pa > 0 else 0.0)
    for key, cols in aliases.items():
        historical_value = None
        for c in cols:
            value = safe_float(profile.get(c))
            if value is not None:
                historical_value = value / 100.0 if value > 1 else value
                break
        if historical_value is None:
            continue
        recent_value = safe_float(fallback_2025.get(key))
        out[key] = historical_value if prior_pa <= 0 or recent_value is None else recent_value * recent_share + historical_value * (1 - recent_share)
    out["historical_pa"] = historical_pa
    out["prior_2025_pa"] = prior_pa
    out["prior_2025_share"] = recent_share
    return out


def effective_batter_side(raw_side: str, pitcher_hand: str) -> str:
    side = str(raw_side or "R").upper()[:1]
    if side == "S":
        return "L" if str(pitcher_hand).upper()[:1] == "R" else "R"
    return side if side in {"R", "L"} else "R"


def apply_platoon_shape(rates: Dict[str, float], batter_side: str, pitcher_hand: str, strength: float = 1.0) -> Dict[str, float]:
    out = dict(rates)
    same = effective_batter_side(batter_side, pitcher_hand) == str(pitcher_hand or "R").upper()[:1]
    # Small fallback only; player-specific Statcast/split data receives priority.
    hit_factor = (0.975 if same else 1.020) ** strength
    bb_factor = (0.985 if same else 1.012) ** strength
    k_factor = (1.030 if same else 0.982) ** strength
    for key in ["single_pa", "double_pa", "triple_pa", "hr_pa"]:
        out[key] = out.get(key, LEAGUE[key]) * hit_factor
    out["bb_pa"] = out.get("bb_pa", LEAGUE["bb_pa"]) * bb_factor
    out["k_pa"] = out.get("k_pa", LEAGUE["k_pa"]) * k_factor
    return normalize_outcome_probs(out)


def build_batter_profile(player_id: int, player_name: str, season: int, opening_day: str, today: str, pitcher_hand: str, line: float) -> Dict[str, Any]:
    current_stat = player_asof_stat(player_id, "hitting", season, today)
    prior_stat = player_season_stat(player_id, "hitting", season - 1)
    prior_split_stat = player_hand_split(player_id, "hitting", season - 1, pitcher_hand)
    current = _batter_rates_from_stat(current_stat)
    prior_2025 = _batter_rates_from_stat(prior_stat)
    prior_split = _batter_rates_from_stat(prior_split_stat) if prior_split_stat else {}
    offline = offline_profile_for(historical_batter_profiles(), player_id, player_name)
    prior = historical_prior_rates(offline, prior_2025)
    pa = current.get("pa", 0)
    prior_split_pa = prior_split.get("pa", 0) if prior_split else 0
    strengths = {"bb_pa": 90, "hbp_pa": 170, "k_pa": 75, "single_pa": 180, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}
    rates: Dict[str, float] = {}
    for key, strength in strengths.items():
        base = beta_blend(current.get(key, LEAGUE[key]), pa, prior.get(key, LEAGUE[key]), strength)
        if prior_split and prior_split_pa >= 40:
            split_weight = clamp(prior_split_pa / 1000, 0.04, 0.16)
            base = base * (1 - split_weight) + prior_split.get(key, base) * split_weight
        rates[key] = float(base)

    statcast = batter_statcast_profile(player_id, opening_day, today, pitcher_hand)
    sc_rates = statcast_event_rates(statcast)
    if sc_rates and sc_rates.get("pa", 0) >= 25:
        split_weight = clamp(sc_rates["pa"] / 750, 0.05, 0.20)
        for key in strengths:
            rates[key] = rates[key] * (1 - split_weight) + sc_rates.get(key, rates[key]) * split_weight
    if statcast.get("available"):
        xba = safe_float(statcast.get("xba"))
        observed_ba = safe_float(current_stat.get("avg"))
        if xba is not None and observed_ba is not None:
            hit_adj = clamp((xba - observed_ba) * 0.20, -0.018, 0.018)
            total_non_hr_hits = rates["single_pa"] + rates["double_pa"] + rates["triple_pa"]
            if total_non_hr_hits > 0:
                rates["single_pa"] += hit_adj * (rates["single_pa"] / total_non_hr_hits)
                rates["double_pa"] += hit_adj * (rates["double_pa"] / total_non_hr_hits)
                rates["triple_pa"] += hit_adj * (rates["triple_pa"] / total_non_hr_hits)
        barrel = safe_float(statcast.get("barrel_pct"))
        hard = safe_float(statcast.get("hard_hit_pct"))
        if barrel is not None:
            rates["hr_pa"] *= clamp(0.92 + barrel / 0.075 * 0.08, 0.88, 1.14)
        if hard is not None:
            rates["double_pa"] *= clamp(0.94 + hard / 0.38 * 0.06, 0.90, 1.10)

    person = get_person(player_id)
    raw_side = str((person.get("batSide") or {}).get("code") or "R").upper()[:1]
    base_rates = normalize_outcome_probs(rates)
    rates = apply_platoon_shape(base_rates, raw_side, pitcher_hand, strength=0.35 if sc_rates else 0.75)
    logs = _logs_before(player_game_log(player_id, "hitting", season), today)
    recent: Dict[str, Any] = {}
    for n in [5, 10, 20, 30]:
        sub = logs.tail(n) if not logs.empty else pd.DataFrame()
        recent[f"l{n}_avg"] = round(float(sub["HRR"].mean()), 2) if not sub.empty else None
        recent[f"l{n}_over"] = round(float((sub["HRR"] > line).mean()), 3) if not sub.empty else None
        recent[f"l{n}_under"] = round(float((sub["HRR"] < line).mean()), 3) if not sub.empty else None
    games = len(logs)
    recent["games"] = games
    recent["season_hrr_avg"] = round(float(logs["HRR"].mean()), 2) if games else None
    recent["season_over"] = round(float((logs["HRR"] > line).mean()), 3) if games else None
    recent["season_under"] = round(float((logs["HRR"] < line).mean()), 3) if games else None
    speed = safe_float(offline.get("sprint_speed"), 27.0) or 27.0
    historical_pa = safe_float(offline.get("historical_pa"), safe_float(offline.get("career_pa"), safe_float(offline.get("sample_size"), 0))) or 0
    history_bonus = min(12.0, math.sqrt(max(historical_pa, 0)) / 6.0) if historical_pa > 0 else 0.0
    historical_obp = safe_float(offline.get("obp"), LEAGUE["obp"]) or LEAGUE["obp"]
    return {
        "player_id": player_id, "name": player_name, "side": raw_side,
        "rates": normalize_outcome_probs(rates), "base_rates": base_rates, "current_pa": pa, "split_pa": safe_float(sc_rates.get("pa"), prior_split_pa) or 0,
        "current_stat": current_stat, "prior_stat": prior_stat, "split_stat": prior_split_stat,
        "statcast": statcast, "logs": logs, "recent": recent, "offline_profile": offline,
        "historical_pa": historical_pa, "historical_source": offline.get("_profile_sources") or offline.get("profile_source"),
        "prior_2025_share": safe_float(prior.get("prior_2025_share")),
        "obp": safe_float(current_stat.get("obp"), safe_float(prior_stat.get("obp"), historical_obp)) or historical_obp,
        "sprint_speed": speed,
        "data_quality": round(clamp(30 + min(35, pa / 5) + (20 if statcast.get("available") else 0) + history_bonus, 25, 98), 1),
    }


def build_compact_batter_profile(player_id: int, season: int, as_of_date: str, pitcher_hand: str) -> Dict[str, Any]:
    person = get_person(player_id)
    name = str(person.get("fullName") or player_id)
    raw_side = str((person.get("batSide") or {}).get("code") or "R").upper()[:1]
    current_stat = player_asof_stat(player_id, "hitting", season, as_of_date)
    prior_stat = player_season_stat(player_id, "hitting", season - 1)
    current, prior_fallback = _batter_rates_from_stat(current_stat), _batter_rates_from_stat(prior_stat)
    offline = offline_profile_for(historical_batter_profiles(), player_id, name)
    prior = historical_prior_rates(offline, prior_fallback)
    pa = current.get("pa", 0)
    rates = {}
    for key, strength in {"bb_pa": 90, "hbp_pa": 170, "k_pa": 75, "single_pa": 180, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}.items():
        rates[key] = beta_blend(current.get(key, LEAGUE[key]), pa, prior.get(key, LEAGUE[key]), strength)
    base_rates = normalize_outcome_probs(rates)
    rates = apply_platoon_shape(base_rates, raw_side, pitcher_hand, strength=0.75)
    historical_pa = safe_float(offline.get("historical_pa"), safe_float(offline.get("career_pa"), safe_float(offline.get("sample_size"), 0))) or 0
    historical_obp = safe_float(offline.get("obp"), LEAGUE["obp"]) or LEAGUE["obp"]
    obp = safe_float(current_stat.get("obp"), safe_float(prior_stat.get("obp"), historical_obp)) or historical_obp
    history_bonus = min(12.0, math.sqrt(max(historical_pa, 0)) / 6.0) if historical_pa > 0 else 0.0
    return {
        "player_id": player_id, "name": name, "side": raw_side, "rates": rates, "base_rates": base_rates,
        "current_pa": pa, "obp": obp, "offline_profile": offline, "historical_pa": historical_pa,
        "historical_source": offline.get("_profile_sources") or offline.get("profile_source"),
        "sprint_speed": safe_float(offline.get("sprint_speed"), 27.0) or 27.0,
        "data_quality": round(clamp(25 + min(45, pa / 4) + history_bonus, 20, 90), 1),
    }

def normalize_outcome_probs(rates: Dict[str, float]) -> Dict[str, float]:
    keys = ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]
    out = {k: clamp(float(rates.get(k, LEAGUE.get(k, 0.0))), 0.0005, 0.45) for k in keys}
    total = sum(out.values())
    # Keep at least 18% generic balls-in-play outs.
    if total > 0.82:
        scale = 0.82 / total
        out = {k: v * scale for k, v in out.items()}
    out["out_pa"] = 1 - sum(out.values())
    return out


def _pitcher_allowed_rates(stat: Dict[str, Any]) -> Dict[str, float]:
    bf = safe_float(stat.get("battersFaced"), 0) or 0
    h = safe_float(stat.get("hits"), 0) or 0
    d2 = safe_float(stat.get("doubles"), 0) or 0
    d3 = safe_float(stat.get("triples"), 0) or 0
    hr = safe_float(stat.get("homeRuns"), 0) or 0
    return {
        "bf": bf,
        "bb_pa": (safe_float(stat.get("baseOnBalls"), 0) or 0) / bf if bf else LEAGUE["bb_pa"],
        "hbp_pa": (safe_float(stat.get("hitBatsmen"), 0) or 0) / bf if bf else LEAGUE["hbp_pa"],
        "k_pa": (safe_float(stat.get("strikeOuts"), 0) or 0) / bf if bf else LEAGUE["k_pa"],
        "single_pa": max(0.0, h - d2 - d3 - hr) / bf if bf else LEAGUE["single_pa"],
        "double_pa": d2 / bf if bf else LEAGUE["double_pa"],
        "triple_pa": d3 / bf if bf else LEAGUE["triple_pa"],
        "hr_pa": hr / bf if bf else LEAGUE["hr_pa"],
    }


def build_pitcher_profile(player_id: Optional[int], player_name: str, season: int, opening_day: str, today: str, batter_side: str) -> Dict[str, Any]:
    if not player_id:
        return {"rates": normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]}), "available": False, "name": player_name or "TBD", "hand": "R", "expected_bf": 22.0, "data_quality": 25}
    person = get_person(player_id)
    hand = str((person.get("pitchHand") or {}).get("code") or "R").upper()[:1]
    current_stat = player_asof_stat(player_id, "pitching", season, today)
    prior_stat = player_season_stat(player_id, "pitching", season - 1)
    current, prior_fallback = _pitcher_allowed_rates(current_stat), _pitcher_allowed_rates(prior_stat)
    offline = offline_profile_for(historical_pitcher_profiles(), player_id, player_name)
    prior = historical_prior_rates(offline, prior_fallback)
    bf = current.get("bf", 0)
    rates = {}
    for key, strength in {"bb_pa": 100, "hbp_pa": 200, "k_pa": 85, "single_pa": 190, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}.items():
        rates[key] = beta_blend(current.get(key, LEAGUE[key]), bf, prior.get(key, LEAGUE[key]), strength)
    statcast = pitcher_statcast_profile(player_id, opening_day, today, effective_batter_side(batter_side, hand))
    sc_rates = statcast_event_rates(statcast)
    if sc_rates and sc_rates.get("pa", 0) >= 30:
        split_weight = clamp(sc_rates["pa"] / 850, 0.05, 0.18)
        for key in rates:
            rates[key] = rates[key] * (1 - split_weight) + sc_rates.get(key, rates[key]) * split_weight
    if statcast.get("available"):
        xba = safe_float(statcast.get("xba"))
        xwoba = safe_float(statcast.get("xwoba"))
        hard = safe_float(statcast.get("hard_hit_pct"))
        if xba is not None:
            hit_factor = clamp(xba / 0.250, 0.88, 1.14)
            for k in ["single_pa", "double_pa", "triple_pa"]:
                rates[k] *= hit_factor
        if xwoba is not None:
            rates["hr_pa"] *= clamp(xwoba / 0.320, 0.86, 1.16)
        if hard is not None:
            rates["double_pa"] *= clamp(hard / 0.38, 0.90, 1.11)
    logs = _logs_before(player_game_log(player_id, "pitching", season), today)
    starts = logs[pd.to_numeric(logs.get("BF", 0), errors="coerce").fillna(0) >= 10].tail(5) if not logs.empty else pd.DataFrame()
    expected_bf = float(starts["BF"].mean()) if not starts.empty and starts["BF"].sum() > 0 else safe_float(current_stat.get("battersFaced"), 22) / max(safe_float(current_stat.get("gamesStarted"), 1), 1)
    expected_bf = clamp(expected_bf or 22.0, 10.0, 30.0)
    data_quality = clamp(35 + min(35, bf / 12) + (15 if statcast.get("available") else 0) + (10 if len(starts) >= 3 else 0) + (5 if offline else 0), 20, 98)
    return {
        "player_id": player_id, "rates": normalize_outcome_probs(rates), "available": True, "name": person.get("fullName") or player_name,
        "hand": hand, "expected_bf": round(expected_bf, 1), "statcast": statcast, "logs": logs,
        "current_stat": current_stat, "offline_profile": offline, "data_quality": round(data_quality, 1),
    }


def build_compact_pitcher_profile(player_id: int, season: int, as_of_date: str) -> Dict[str, Any]:
    person = get_person(player_id)
    name = str(person.get("fullName") or player_id)
    hand = str((person.get("pitchHand") or {}).get("code") or "R").upper()[:1]
    current_stat = player_asof_stat(player_id, "pitching", season, as_of_date)
    prior_stat = player_season_stat(player_id, "pitching", season - 1)
    current, prior_fallback = _pitcher_allowed_rates(current_stat), _pitcher_allowed_rates(prior_stat)
    offline = offline_profile_for(historical_pitcher_profiles(), player_id, name)
    prior = historical_prior_rates(offline, prior_fallback)
    bf = current.get("bf", 0)
    rates = {}
    for key, strength in {"bb_pa": 100, "hbp_pa": 200, "k_pa": 85, "single_pa": 190, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}.items():
        rates[key] = beta_blend(current.get(key, LEAGUE[key]), bf, prior.get(key, LEAGUE[key]), strength)
    games = safe_float(current_stat.get("gamesPitched"), safe_float(current_stat.get("gamesPlayed"), 0)) or 0
    starts = safe_float(current_stat.get("gamesStarted"), 0) or 0
    return {
        "player_id": player_id, "name": name, "hand": hand, "rates": normalize_outcome_probs(rates),
        "current_stat": current_stat, "bf": bf, "games": games, "starts": starts,
        "saves": safe_float(current_stat.get("saves"), 0) or 0, "holds": safe_float(current_stat.get("holds"), 0) or 0,
        "games_finished": safe_float(current_stat.get("gamesFinished"), 0) or 0,
        "innings": innings_to_float(current_stat.get("inningsPitched")), "offline_profile": offline,
        "data_quality": round(clamp(25 + min(45, bf / 9) + (10 if offline else 0), 20, 92), 1),
    }

def arsenal_match_factor(batter_sc: Dict[str, Any], pitcher_sc: Dict[str, Any]) -> Tuple[float, str]:
    mix = pitcher_sc.get("pitch_mix") or {}
    batter_rows = {str(r.get("pitch_type")): r for r in batter_sc.get("pitch_types") or []}
    if not mix or not batter_rows:
        return 1.0, "Arsenal sample unavailable"
    weighted, used = 0.0, 0.0
    for pitch_type, usage in mix.items():
        row = batter_rows.get(str(pitch_type))
        if not row or usage < 0.03:
            continue
        hit_rate = safe_float(row.get("hit_rate"))
        avg_ev = safe_float(row.get("avg_ev"))
        component = 1.0
        if hit_rate is not None:
            component *= clamp(hit_rate / 0.245, 0.85, 1.18)
        if avg_ev is not None:
            component *= clamp(1 + (avg_ev - 88.5) * 0.012, 0.90, 1.12)
        weighted += usage * component
        used += usage
    if used < 0.20:
        return 1.0, "Arsenal overlap thin"
    raw = weighted / used
    factor = clamp(1 + (raw - 1) * 0.45, 0.93, 1.07)
    return factor, f"Pitch-mix interaction x{factor:.3f}"


def blend_matchup_probs(
    batter: Dict[str, float],
    pitcher: Dict[str, float],
    park: Dict[str, float],
    weather: Any,
    bullpen_quality: float = 1.0,
    arsenal_factor: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    result: Dict[str, float] = {}
    for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
        league = LEAGUE[key]
        b, p = batter.get(key, league), pitcher.get(key, league)
        result[key] = league * (b / league) ** 0.62 * (p / league) ** 0.38
    if isinstance(weather, dict):
        w1 = safe_float(weather.get("1B_factor"), 1.0) or 1.0
        w2 = safe_float(weather.get("2B_factor"), 1.0) or 1.0
        w3 = safe_float(weather.get("3B_factor"), 1.0) or 1.0
        whr = safe_float(weather.get("HR_factor"), safe_float(weather.get("carry_factor"), 1.0)) or 1.0
    else:
        w1, w2, w3, whr = 1.0, 1.0, 1.0, safe_float(weather, 1.0) or 1.0
    result["single_pa"] *= park.get("1B", 1.0) * w1
    result["double_pa"] *= park.get("2B", 1.0) * w2
    result["triple_pa"] *= park.get("3B", 1.0) * w3
    result["hr_pa"] *= park.get("HR", 1.0) * whr * arsenal_factor
    starter = normalize_outcome_probs(result)
    bullpen = {}
    for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
        bullpen[key] = LEAGUE[key] * 0.50 + batter.get(key, LEAGUE[key]) * 0.33 + result.get(key, LEAGUE[key]) * 0.17
    for key in ["single_pa", "double_pa", "triple_pa", "hr_pa", "bb_pa"]:
        bullpen[key] *= clamp(bullpen_quality, 0.88, 1.14)
    return starter, normalize_outcome_probs(bullpen)

def pitcher_vulnerability(profile: Dict[str, Any]) -> Dict[str, Any]:
    r = profile.get("rates") or {}
    sc = profile.get("statcast") or {}
    contact = 50 + (r.get("single_pa", LEAGUE["single_pa"]) - LEAGUE["single_pa"]) * 430 - (r.get("k_pa", LEAGUE["k_pa"]) - LEAGUE["k_pa"]) * 170
    damage = 50 + (r.get("hr_pa", LEAGUE["hr_pa"]) - LEAGUE["hr_pa"]) * 700 + (r.get("double_pa", LEAGUE["double_pa"]) - LEAGUE["double_pa"]) * 350
    traffic = 50 + (r.get("bb_pa", LEAGUE["bb_pa"]) - LEAGUE["bb_pa"]) * 300 + (r.get("single_pa", LEAGUE["single_pa"]) - LEAGUE["single_pa"]) * 250
    if safe_float(sc.get("hard_hit_pct")) is not None:
        damage += (safe_float(sc.get("hard_hit_pct")) - 0.38) * 80
    if safe_float(sc.get("xba")) is not None:
        contact += (safe_float(sc.get("xba")) - 0.25) * 110
    scores = {"Contact Allowed": round(clamp(contact, 10, 95), 1), "Damage Allowed": round(clamp(damage, 10, 95), 1), "Traffic Allowed": round(clamp(traffic, 10, 95), 1)}
    scores["Overall"] = round(np.mean(list(scores.values())), 1)
    return scores



def _bullpen_role(profile: Dict[str, Any]) -> str:
    games = max(safe_float(profile.get("games"), 0) or 0, 1.0)
    saves = safe_float(profile.get("saves"), 0) or 0
    holds = safe_float(profile.get("holds"), 0) or 0
    gf = safe_float(profile.get("games_finished"), 0) or 0
    ip_per = (safe_float(profile.get("innings"), 0) or 0) / games
    if saves >= 4 or saves / games >= 0.18:
        return "CLOSER"
    if holds >= 4 or gf / games >= 0.38:
        return "SETUP"
    if ip_per >= 1.20:
        return "LONG"
    return "MIDDLE"


def _bullpen_offense_factor(rates: Dict[str, float]) -> float:
    hit = rates.get("single_pa", LEAGUE["single_pa"]) + rates.get("double_pa", LEAGUE["double_pa"]) + rates.get("triple_pa", LEAGUE["triple_pa"])
    lg_hit = LEAGUE["single_pa"] + LEAGUE["double_pa"] + LEAGUE["triple_pa"]
    factor = 0.45 * (hit / lg_hit) + 0.25 * (rates.get("hr_pa", LEAGUE["hr_pa"]) / LEAGUE["hr_pa"]) + 0.20 * (rates.get("bb_pa", LEAGUE["bb_pa"]) / LEAGUE["bb_pa"]) + 0.10 * (LEAGUE["k_pa"] / max(rates.get("k_pa", LEAGUE["k_pa"]), 0.05))
    return clamp(factor, 0.84, 1.18)


@st.cache_data(ttl=1200, show_spinner=False)
def build_bullpen_model(team_id: int, probable_starter_id: Optional[int], season: int, opening_day: str, as_of_date: str) -> Dict[str, Any]:
    roster = active_team_roster(team_id, as_of_date)
    usage = recent_reliever_usage(team_id, as_of_date, 4)
    relievers: List[Dict[str, Any]] = []
    for row in roster:
        if str(row.get("position_type")) != "Pitcher":
            continue
        pid = safe_int(row.get("player_id"))
        if not pid or pid == safe_int(probable_starter_id):
            continue
        prof = build_compact_pitcher_profile(pid, season, as_of_date)
        games = safe_float(prof.get("games"), 0) or 0
        starts = safe_float(prof.get("starts"), 0) or 0
        if games <= 0:
            continue
        if starts >= max(3.0, games * 0.45) and (prof.get("saves", 0) + prof.get("holds", 0)) < 2:
            continue
        u = (usage.get("by_pitcher") or {}).get(str(pid), {})
        availability = safe_float(u.get("availability"), 1.0) or 1.0
        role = _bullpen_role(prof)
        base_weight = max(1.0, games + 1.7 * (prof.get("holds", 0) or 0) + 2.2 * (prof.get("saves", 0) or 0))
        expected_bf = clamp(((prof.get("innings", 0) or 0) / max(games, 1)) * 3.0, 2.0, 7.0)
        relievers.append({
            **prof,
            "role": role,
            "availability": round(availability, 3),
            "base_weight": round(base_weight * availability, 3),
            "expected_bf": round(expected_bf, 2),
            "pitches_1d": u.get("pitches_1d", 0.0),
            "pitches_2d": u.get("pitches_2d", 0.0),
            "pitches_3d": u.get("pitches_3d", 0.0),
            "consecutive_days": bool(u.get("consecutive_days")),
        })
    relievers = sorted(relievers, key=lambda r: (r.get("availability", 0), r.get("base_weight", 0)), reverse=True)[:10]
    if relievers:
        weights = np.array([max(0.01, safe_float(r.get("base_weight"), 1.0) or 1.0) for r in relievers], dtype=float)
        weights /= weights.sum()
        aggregate = {}
        for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
            aggregate[key] = float(sum(w * r["rates"].get(key, LEAGUE[key]) for w, r in zip(weights, relievers)))
        aggregate = normalize_outcome_probs(aggregate)
        quality = _bullpen_offense_factor(aggregate)
        dq = float(np.average([r.get("data_quality", 40) for r in relievers], weights=weights))
    else:
        aggregate = normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]})
        quality, dq = 1.0, 25.0
    fatigue = safe_float(usage.get("fatigue_factor"), 1.0) or 1.0
    quality = clamp(quality * (1 + (fatigue - 1) * 0.45), 0.84, 1.20)
    return {
        "team_id": team_id,
        "relievers": relievers,
        "aggregate_rates": aggregate,
        "quality_factor": round(quality, 3),
        "data_quality": round(clamp(dq + min(10, len(relievers)), 20, 98), 1),
        "available_count": sum(1 for r in relievers if r.get("availability", 0) >= 0.35),
        "ids": [r.get("player_id") for r in relievers],
        "pitches_3d": usage.get("pitches_3d", 0.0),
        "innings_3d": usage.get("innings_3d", 0.0),
        "repeat_relievers": usage.get("repeat_relievers", 0),
        "fatigue_factor": round(fatigue, 3),
    }


def reliever_role_multiplier(role: str, inning: int) -> float:
    role = str(role or "MIDDLE")
    if inning <= 5:
        return {"LONG": 2.0, "MIDDLE": 1.25, "SETUP": 0.35, "CLOSER": 0.12}.get(role, 1.0)
    if inning == 6:
        return {"LONG": 1.30, "MIDDLE": 1.35, "SETUP": 0.65, "CLOSER": 0.18}.get(role, 1.0)
    if inning == 7:
        return {"LONG": 0.55, "MIDDLE": 1.00, "SETUP": 1.60, "CLOSER": 0.35}.get(role, 1.0)
    if inning == 8:
        return {"LONG": 0.25, "MIDDLE": 0.75, "SETUP": 1.75, "CLOSER": 0.85}.get(role, 1.0)
    return {"LONG": 0.18, "MIDDLE": 0.55, "SETUP": 1.05, "CLOSER": 2.10}.get(role, 1.0)


def build_full_lineup_context(
    team_id: int,
    target_player_id: int,
    target_slot: int,
    target_profile: Dict[str, Any],
    pitcher_profile: Dict[str, Any],
    bullpen_model: Dict[str, Any],
    game_ctx: Dict[str, Any],
    is_home: bool,
    season: int,
    as_of_date: str,
    park: Dict[str, float],
    weather: Dict[str, Any],
    target_arsenal_factor: float,
) -> Dict[str, Any]:
    lineup_ids, lineup_status = resolve_lineup_ids(game_ctx, team_id, is_home, season, as_of_date, target_player_id, target_slot)
    profiles: List[Dict[str, Any]] = []
    for pid in lineup_ids:
        if safe_int(pid) == safe_int(target_player_id):
            profiles.append(dict(target_profile))
        else:
            profiles.append(build_compact_batter_profile(pid, season, as_of_date, pitcher_profile.get("hand", "R")))
    while len(profiles) < 9:
        profiles.append({
            "player_id": 0, "name": f"Replacement {len(profiles)+1}", "side": "R",
            "rates": normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]}),
            "base_rates": normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]}),
            "obp": LEAGUE["obp"], "sprint_speed": 27.0, "data_quality": 20.0,
        })
    profiles = profiles[:9]
    target_index = next((i for i, p in enumerate(profiles) if safe_int(p.get("player_id")) == safe_int(target_player_id)), int(clamp(target_slot, 1, 9)) - 1)
    starter_probs: List[Dict[str, float]] = []
    fallback_bullpen_probs: List[Dict[str, float]] = []
    aggregate_bp_rates = bullpen_model.get("aggregate_rates") or normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]})
    aggregate_bp_rates = dict(aggregate_bp_rates)
    bp_quality = safe_float(bullpen_model.get("quality_factor"), 1.0) or 1.0
    for key in ["single_pa", "double_pa", "triple_pa", "hr_pa", "bb_pa"]:
        aggregate_bp_rates[key] = aggregate_bp_rates.get(key, LEAGUE[key]) * bp_quality
    aggregate_bp_rates["k_pa"] = aggregate_bp_rates.get("k_pa", LEAGUE["k_pa"]) / max(bp_quality ** 0.35, 0.85)
    aggregate_bp_rates = normalize_outcome_probs(aggregate_bp_rates)
    for i, prof in enumerate(profiles):
        arsenal = target_arsenal_factor if i == target_index else 1.0
        sp, _ = blend_matchup_probs(prof.get("rates") or prof.get("base_rates") or {}, pitcher_profile.get("rates") or {}, park, weather, 1.0, arsenal)
        bp_batter = prof.get("base_rates") or prof.get("rates") or {}
        bp, _ = blend_matchup_probs(bp_batter, aggregate_bp_rates, park, weather, 1.0, 1.0)
        starter_probs.append(sp)
        fallback_bullpen_probs.append(bp)
    relievers = bullpen_model.get("relievers") or []
    individual: List[List[Dict[str, float]]] = []
    for rel in relievers:
        row_probs: List[Dict[str, float]] = []
        for prof in profiles:
            batter_rates = apply_platoon_shape(prof.get("base_rates") or prof.get("rates") or {}, prof.get("side", "R"), rel.get("hand", "R"), strength=0.80)
            p, _ = blend_matchup_probs(batter_rates, rel.get("rates") or aggregate_bp_rates, park, weather, 1.0, 1.0)
            row_probs.append(p)
        individual.append(row_probs)
    return {
        "profiles": profiles,
        "lineup_ids": [p.get("player_id") for p in profiles],
        "lineup_names": [p.get("name") for p in profiles],
        "lineup_status": lineup_status,
        "target_index": target_index,
        "starter_probs": starter_probs,
        "fallback_bullpen_probs": fallback_bullpen_probs,
        "individual_bullpen_probs": individual,
        "relievers": relievers,
        "speed_boosts": [clamp(((safe_float(p.get("sprint_speed"), 27.0) or 27.0) - 27.0) * 0.025, -0.06, 0.08) for p in profiles],
        "lineup_obp": float(np.mean([safe_float(p.get("obp"), LEAGUE["obp"]) or LEAGUE["obp"] for p in profiles])),
        "lineup_data_quality": float(np.mean([safe_float(p.get("data_quality"), 30) or 30 for p in profiles])),
    }


# ============================================================
# CORRELATED FULL BASE/OUT-STATE SIMULATION
# ============================================================
OUTCOMES = ["BB", "HBP", "K", "1B", "2B", "3B", "HR", "OUT"]


def probs_array(probs: Dict[str, float]) -> np.ndarray:
    vals = np.array([
        probs.get("bb_pa", 0), probs.get("hbp_pa", 0), probs.get("k_pa", 0),
        probs.get("single_pa", 0), probs.get("double_pa", 0), probs.get("triple_pa", 0),
        probs.get("hr_pa", 0), probs.get("out_pa", 0),
    ], dtype=float)
    vals = np.clip(vals, 1e-7, None)
    return vals / vals.sum()


def generic_team_probs(team_stat: Dict[str, Any]) -> Dict[str, float]:
    rates = _batter_rates_from_stat(team_stat)
    return normalize_outcome_probs({k: rates[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]})


def force_walk(bases: List[int], runner: int) -> Tuple[List[int], List[int]]:
    scored: List[int] = []
    first, second, third = bases
    if first:
        if second:
            if third:
                scored.append(third)
            third = second
        second = first
    first = runner
    return [first, second, third], scored


def _runner_speed(code: int, speed_boosts: Sequence[float]) -> float:
    idx = int(code) - 1
    if 0 <= idx < len(speed_boosts):
        return safe_float(speed_boosts[idx], 0.0) or 0.0
    return 0.0


def advance_hit(bases: List[int], runner: int, hit_type: str, rng: np.random.Generator, speed_boosts: Sequence[float]) -> Tuple[List[int], List[int]]:
    first, second, third = bases
    scored: List[int] = []
    if hit_type == "HR":
        scored.extend([x for x in bases if x])
        scored.append(runner)
        return [0, 0, 0], scored
    if hit_type == "3B":
        scored.extend([x for x in bases if x])
        return [0, 0, runner], scored
    if hit_type == "2B":
        if third:
            scored.append(third)
        if second:
            scored.append(second)
        new_third = 0
        if first:
            speed = _runner_speed(first, speed_boosts)
            if rng.random() < clamp(0.53 + speed, 0.35, 0.80):
                scored.append(first)
            else:
                new_third = first
        return [0, runner, new_third], scored
    if third:
        scored.append(third)
    new_third, new_second = 0, 0
    if second:
        speed = _runner_speed(second, speed_boosts)
        if rng.random() < clamp(0.64 + speed, 0.46, 0.88):
            scored.append(second)
        else:
            new_third = second
    if first:
        speed = _runner_speed(first, speed_boosts)
        if new_third == 0 and rng.random() < clamp(0.29 + speed * 0.55, 0.14, 0.50):
            new_third = first
        else:
            new_second = first
    return [runner, new_second, new_third], scored


def apply_out(bases: List[int], outs: int, rng: np.random.Generator) -> Tuple[List[int], int, List[int]]:
    first, second, third = bases
    scored: List[int] = []
    if outs < 2 and third and rng.random() < 0.19:
        scored.append(third)
        third = 0
        return [first, second, third], outs + 1, scored
    if outs < 2 and first and rng.random() < 0.105:
        first = 0
        return [first, second, third], min(3, outs + 2), scored
    return [first, second, third], outs + 1, scored


@dataclass
class SimResult:
    hits: np.ndarray
    runs: np.ndarray
    rbi: np.ndarray
    hrr: np.ndarray
    pa: np.ndarray


def _choose_reliever(rng: np.random.Generator, inning: int, relievers: Sequence[Dict[str, Any]], used: set) -> Optional[int]:
    if not relievers:
        return None
    weights = []
    for idx, rel in enumerate(relievers):
        base = safe_float(rel.get("base_weight"), 1.0) or 1.0
        avail = safe_float(rel.get("availability"), 1.0) or 1.0
        role = reliever_role_multiplier(str(rel.get("role") or "MIDDLE"), inning)
        repeat_penalty = 0.22 if idx in used else 1.0
        weights.append(max(1e-6, base * avail * role * repeat_penalty))
    arr = np.array(weights, dtype=float)
    if arr.sum() <= 0:
        return None
    arr /= arr.sum()
    return int(rng.choice(len(relievers), p=arr))


def simulate_player_games(
    target_index: int,
    is_home: bool,
    lineup_starter_probs: Sequence[Dict[str, float]],
    lineup_fallback_bullpen_probs: Sequence[Dict[str, float]],
    individual_bullpen_probs: Sequence[Sequence[Dict[str, float]]],
    relievers: Sequence[Dict[str, Any]],
    expected_starter_bf: float,
    opponent_runs_mean: float,
    simulations: int,
    seed: int,
    speed_boosts: Sequence[float],
    uncertainty_strength: float = 120.0,
) -> SimResult:
    rng = np.random.default_rng(seed)
    target_index = int(clamp(target_index, 0, 8))
    target_code = target_index + 1
    sp_matrix = np.stack([probs_array(p) for p in lineup_starter_probs], axis=0)
    fb_matrix = np.stack([probs_array(p) for p in lineup_fallback_bullpen_probs], axis=0)
    if individual_bullpen_probs:
        bp_matrix = np.stack([[probs_array(p) for p in row] for row in individual_bullpen_probs], axis=0)
    else:
        bp_matrix = np.empty((0, 9, len(OUTCOMES)), dtype=float)
    hits = np.zeros(simulations, dtype=np.int16)
    runs = np.zeros(simulations, dtype=np.int16)
    rbi = np.zeros(simulations, dtype=np.int16)
    hrr = np.zeros(simulations, dtype=np.int16)
    pas = np.zeros(simulations, dtype=np.int16)
    alpha_scale = max(25.0, uncertainty_strength)

    for sim in range(simulations):
        sp_draw = sp_matrix.copy()
        sp_draw[target_index] = rng.dirichlet(np.clip(sp_matrix[target_index] * alpha_scale, 0.25, None))
        bp_draw = bp_matrix.copy()
        if len(bp_draw):
            for ridx in range(len(bp_draw)):
                bp_draw[ridx, target_index] = rng.dirichlet(np.clip(bp_matrix[ridx, target_index] * alpha_scale, 0.25, None))
        fallback_draw = fb_matrix.copy()
        fallback_draw[target_index] = rng.dirichlet(np.clip(fb_matrix[target_index] * alpha_scale, 0.25, None))

        starter_bf_limit = int(clamp(rng.normal(expected_starter_bf, 2.4), 9, 32))
        opponent_runs = int(rng.poisson(max(1.3, opponent_runs_mean)))
        team_runs = 0
        batter_index = 0
        starter_bf = 0
        current_reliever: Optional[int] = None
        reliever_bf_left = 0
        used_relievers: set = set()
        target_h = target_r = target_rbi = target_pa = 0

        for inning in range(1, 10):
            if is_home and inning == 9 and team_runs > opponent_runs:
                break
            outs = 0
            bases = [0, 0, 0]
            while outs < 3:
                is_target = batter_index == target_index
                pitcher_is_starter = starter_bf < starter_bf_limit
                if pitcher_is_starter:
                    p = sp_draw[batter_index]
                    starter_bf += 1
                else:
                    if current_reliever is None or reliever_bf_left <= 0:
                        current_reliever = _choose_reliever(rng, inning, relievers, used_relievers)
                        if current_reliever is not None:
                            used_relievers.add(current_reliever)
                            mean_bf = safe_float(relievers[current_reliever].get("expected_bf"), 3.5) or 3.5
                            reliever_bf_left = int(clamp(round(rng.normal(mean_bf, 1.0)), 1, 8))
                    if current_reliever is not None and len(bp_draw) > current_reliever:
                        p = bp_draw[current_reliever, batter_index]
                    else:
                        p = fallback_draw[batter_index]
                    reliever_bf_left -= 1
                if is_target:
                    target_pa += 1
                runner_code = batter_index + 1
                outcome = OUTCOMES[int(rng.choice(len(OUTCOMES), p=p))]
                scored: List[int] = []
                if outcome in {"BB", "HBP"}:
                    bases, scored = force_walk(bases, runner_code)
                    if is_target and scored:
                        target_rbi += len(scored)  # bases-loaded walk/HBP RBI
                elif outcome in {"1B", "2B", "3B", "HR"}:
                    bases, scored = advance_hit(bases, runner_code, outcome, rng, speed_boosts)
                    if is_target:
                        target_h += 1
                        target_rbi += sum(1 for x in scored if x != target_code)
                        if outcome == "HR":
                            target_rbi += 1
                elif outcome == "K":
                    outs += 1
                else:
                    bases, outs, scored = apply_out(bases, outs, rng)
                    if is_target:
                        target_rbi += sum(1 for x in scored if x != target_code)
                if scored:
                    team_runs += len(scored)
                    target_r += sum(1 for x in scored if x == target_code)
                batter_index = (batter_index + 1) % 9
        hits[sim], runs[sim], rbi[sim], pas[sim] = target_h, target_r, target_rbi, target_pa
        hrr[sim] = target_h + target_r + target_rbi
    return SimResult(hits=hits, runs=runs, rbi=rbi, hrr=hrr, pa=pas)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    return float(v[min(len(v) - 1, int(np.searchsorted(cdf, q, side="left")))])


def pa_poststratification_weights(pa_values: np.ndarray, target_dist: Optional[Dict[int, float]]) -> Tuple[np.ndarray, float]:
    n = len(pa_values)
    if n == 0 or not target_dist:
        return np.ones(n, dtype=float) / max(n, 1), float(n)
    clean = {int(k): max(0.0, float(v)) for k, v in target_dist.items()}
    total = sum(clean.values()) or 1.0
    clean = {k: v / total * 0.995 for k, v in clean.items()}
    observed = Counter(int(v) for v in pa_values.tolist())
    raw_prob = {k: c / n for k, c in observed.items()}
    weights = np.zeros(n, dtype=float)
    for i, pa in enumerate(pa_values.astype(int)):
        if pa <= 2:
            target = 0.0025
        elif pa >= 7:
            target = 0.0025
        else:
            target = clean.get(pa, 0.0025)
        obs = max(raw_prob.get(pa, 1 / n), 1 / n)
        weights[i] = clamp(target / obs, 0.02, 25.0)
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights ** 2)) if np.sum(weights ** 2) > 0 else 0.0
    return weights, ess


def summarize_sim(result: SimResult, line: float, pa_target_dist: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    vals = result.hrr.astype(float)
    weights, ess = pa_poststratification_weights(result.pa, pa_target_dist)
    over = float(np.sum(weights * (vals > line)))
    under = float(np.sum(weights * (vals < line)))
    push = float(np.sum(weights * (vals == line)))
    unique = sorted(set(result.hrr.tolist()))
    mode = max(unique, key=lambda x: float(np.sum(weights[result.hrr == x]))) if unique else None
    return {
        "projection": round(float(np.sum(weights * vals)), 2),
        "median": round(_weighted_quantile(vals, weights, 0.50), 2), "mode": mode,
        "over_prob": over, "under_prob": under, "push_prob": push,
        "hits": round(float(np.sum(weights * result.hits)), 2),
        "runs": round(float(np.sum(weights * result.runs)), 2),
        "rbi": round(float(np.sum(weights * result.rbi)), 2),
        "pa": round(float(np.sum(weights * result.pa)), 2),
        "raw_pa": round(float(result.pa.mean()), 2),
        "std": round(float(np.sqrt(np.sum(weights * (vals - np.sum(weights * vals)) ** 2))), 2),
        "p10": _weighted_quantile(vals, weights, 0.10), "p90": _weighted_quantile(vals, weights, 0.90),
        "pa_calibration_ess": round(ess, 1),
    }


def empirical_probability(logs: pd.DataFrame, line: float, side: str, scale: float = 1.0) -> Optional[float]:
    if logs.empty or "HRR" not in logs or len(logs) < 5:
        return None
    values = logs["HRR"].astype(float).to_numpy()
    weights = np.linspace(0.70, 1.30, len(values))
    adjusted = values * clamp(scale, 0.85, 1.15)
    wins = adjusted > line if side == "OVER" else adjusted < line
    weighted_wins = float(np.sum(weights * wins))
    total = float(np.sum(weights))
    return (weighted_wins + 3.0) / (total + 6.0)

# ============================================================
# BOARD ENGINE
# ============================================================
def build_team_environment(
    team_id: int,
    opp_team_id: int,
    season: int,
    opening_day: str,
    as_of_date: str,
    park: Dict[str, float],
    weather: Dict[str, Any],
    bullpen: Dict[str, Any],
) -> Dict[str, Any]:
    team_hit = team_asof_stat(team_id, "hitting", season, opening_day, as_of_date)
    opp_pitch = team_asof_stat(opp_team_id, "pitching", season, opening_day, as_of_date)
    opp_hit = team_asof_stat(opp_team_id, "hitting", season, opening_day, as_of_date)
    team_pitch = team_asof_stat(team_id, "pitching", season, opening_day, as_of_date)
    games = safe_float(team_hit.get("gamesPlayed"), 0) or 0
    runs_pg = (safe_float(team_hit.get("runs"), 0) or 0) / games if games else LEAGUE["runs_per_game"]
    ops = safe_float(team_hit.get("ops"), LEAGUE["ops"]) or LEAGUE["ops"]
    opp_games = safe_float(opp_pitch.get("gamesPlayed"), safe_float(opp_pitch.get("gamesPitched"), 0)) or 0
    opp_ra = (safe_float(opp_pitch.get("runs"), 0) or 0) / opp_games if opp_games else LEAGUE["runs_per_game"]
    implied = 0.48 * runs_pg + 0.30 * opp_ra + 0.22 * LEAGUE["runs_per_game"]
    implied *= park.get("R", 1.0)
    implied *= clamp((weather.get("carry_factor", 1.0) or 1.0) ** 0.35, 0.97, 1.03)
    implied *= clamp(1 + (safe_float(bullpen.get("quality_factor"), 1.0) - 1) * 0.30, 0.94, 1.08)
    implied += clamp((ops - LEAGUE["ops"]) * 2.4, -0.35, 0.35)

    opp_games_hit = safe_float(opp_hit.get("gamesPlayed"), 0) or 0
    opp_runs_pg = (safe_float(opp_hit.get("runs"), 0) or 0) / opp_games_hit if opp_games_hit else LEAGUE["runs_per_game"]
    team_pitch_games = safe_float(team_pitch.get("gamesPlayed"), safe_float(team_pitch.get("gamesPitched"), 0)) or 0
    team_ra = (safe_float(team_pitch.get("runs"), 0) or 0) / team_pitch_games if team_pitch_games else LEAGUE["runs_per_game"]
    opp_mean = clamp(0.58 * opp_runs_pg + 0.42 * team_ra, 2.4, 7.0)
    return {
        "team_stat": team_hit, "opp_pitch_stat": opp_pitch,
        "implied_runs": round(clamp(implied, 2.5, 7.1), 2),
        "runs_pg": round(runs_pg, 2), "ops": round(ops, 3), "opp_ra_pg": round(opp_ra, 2),
        "opponent_runs_mean": round(opp_mean, 2),
        "as_of_date": as_of_date, "cutoff_exclusive": as_of_date,
    }


def data_quality_score(row: Dict[str, Any]) -> Tuple[float, str]:
    score = 12.0
    score += 10 if row.get("Player ID") else 0
    score += 8 if row.get("GamePk") else 0
    score += 12 if row.get("Pitcher ID") else 2
    score += 10 if row.get("Batter PA", 0) >= 100 else 6 if row.get("Batter PA", 0) >= 40 else 2
    score += 10 if row.get("Batter Statcast") else 0
    score += 8 if row.get("Pitcher Statcast") else 0
    score += 10 if row.get("Lineup Status") == "CONFIRMED" else 4
    score += 8 if row.get("Full Lineup Count", 0) >= 9 else 3
    score += 8 if row.get("Bullpen Available Count", 0) >= 5 else 4 if row.get("Bullpen Available Count", 0) >= 3 else 0
    score += 5 if row.get("Weather Available") else 2
    score += 5 if row.get("Historical Batter Profile") else 0
    score += 4 if row.get("Historical Pitcher Profile") else 0
    score += clamp((safe_float(row.get("Lineup Data Quality"), 30) or 30) / 20, 1, 5)
    score = clamp(score, 10, 100)
    label = "ELITE" if score >= 90 else "STRONG" if score >= 78 else "OK" if score >= 65 else "THIN" if score >= 48 else "POOR"
    return round(score, 1), label

def select_grade(side: str, pick_prob: float, edge: float, data_score: float, lineup_status: str, disagreement: float, volatility: float, role_risk: float) -> Tuple[str, str]:
    if data_score < 48 or role_risk >= 70:
        return "🚫 PASS", "Insufficient/unstable pregame data"
    if disagreement >= 0.11:
        return "🚫 PASS", "Simulation and independent baseline disagree"
    if pick_prob >= 0.635 and abs(edge) >= 0.55 and data_score >= 82 and volatility <= 2.25 and lineup_status == "CONFIRMED":
        return "🔥 ATTACK", "Strong calibrated probability, confirmed lineup and model agreement"
    if pick_prob >= 0.595 and abs(edge) >= 0.35 and data_score >= 72 and role_risk < 45:
        return "✅ OFFICIAL", "Clear edge with acceptable uncertainty"
    if pick_prob >= 0.555 and abs(edge) >= 0.20 and data_score >= 60:
        return "⚠️ PLAYABLE", "Smaller edge or one manageable uncertainty"
    if pick_prob >= 0.52:
        return "👀 TRACK ONLY", "Direction exists but edge is not strong enough"
    return "🚫 PASS", "No reliable edge"


def saved_odds_lookup(player: str, game_date: str, line: float) -> Dict[str, Any]:
    data = read_json(SAVED_ODDS_FILE, {})
    return data.get(f"{game_date}|{normalize_name(player)}|{line}", {})


def historical_probability_calibration(prob: float) -> Tuple[float, float, str]:
    learning = read_json(LEARNING_FILE, {})
    calibration = learning.get("calibration") or {}
    for label, row in calibration.items():
        m = re.match(r"(\d+)-(\d+)%", str(label))
        if not m:
            continue
        low, high = int(m.group(1)) / 100.0, int(m.group(2)) / 100.0
        if low <= prob < high or (high >= 0.78 and low <= prob <= high):
            n = safe_int(row.get("n"), 0) or 0
            actual = safe_float(row.get("win_rate"))
            avg_p = safe_float(row.get("avg_probability"))
            if n >= 20 and actual is not None and avg_p is not None:
                weight = clamp(n / 180.0, 0.08, 0.45)
                adjusted = clamp(prob + (actual - avg_p) * weight, 0.34, 0.78)
                return adjusted, adjusted - prob, f"Historical calibration {label} (n={n})"
    return prob, 0.0, "No mature historical calibration bucket"


def build_one_projection(ud: Dict[str, Any], game_date: str, season: int, opening_day: str, screen_sims: int = 5000) -> Dict[str, Any]:
    player_id, player_name, line = safe_int(ud.get("Player ID")), str(ud.get("Player")), float(ud.get("Line"))
    person = get_person(player_id) if player_id else {}
    team = person.get("currentTeam") or {}
    team_id = safe_int(team.get("id"))
    schedule = get_schedule_for_date(game_date)
    game = match_player_game(team_id, schedule)
    game_ctx = get_live_game_context(game.get("game_pk")) if game else {}
    slot, lineup_status = confirmed_lineup_slot(game_ctx, bool(game.get("is_home")), player_id)
    if slot is None:
        slots = recent_lineup_slots(player_id, season, 5, game_date)
        slot = int(round(float(np.median(slots)))) if slots else 5
        lineup_status = "PROJECTED"
    pitcher_id = safe_int(game.get("opp_pitcher_id"))
    pitcher_person = get_person(pitcher_id) if pitcher_id else {}
    pitcher_hand = str((pitcher_person.get("pitchHand") or {}).get("code") or "R").upper()[:1]
    batter_side = str((person.get("batSide") or {}).get("code") or "R").upper()[:1]

    batter = build_batter_profile(player_id, player_name, season, opening_day, game_date, pitcher_hand, line)
    pitcher = build_pitcher_profile(pitcher_id, game.get("opp_pitcher") or "TBD", season, opening_day, game_date, batter_side)
    venue = str(game.get("venue") or "")
    park = get_park_factors(venue)
    weather_raw = game_ctx.get("weather") or {}
    weather = parse_weather_factor(weather_raw, venue, game.get("start_time"), game_date)
    opponent_team_id = safe_int(game.get("opponent_team_id"))
    bullpen = build_bullpen_model(opponent_team_id, pitcher_id, season, opening_day, game_date) if opponent_team_id else {
        "relievers": [], "aggregate_rates": normalize_outcome_probs({k: LEAGUE[k] for k in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]}),
        "quality_factor": 1.0, "data_quality": 20, "available_count": 0, "ids": [], "fatigue_factor": 1.0,
    }
    env = build_team_environment(team_id, opponent_team_id, season, opening_day, game_date, park, weather, bullpen) if team_id and opponent_team_id else {
        "team_stat": {}, "implied_runs": LEAGUE["runs_per_game"], "runs_pg": LEAGUE["runs_per_game"], "ops": LEAGUE["ops"],
        "opp_ra_pg": LEAGUE["runs_per_game"], "opponent_runs_mean": LEAGUE["runs_per_game"], "cutoff_exclusive": game_date,
    }
    arsenal_factor, arsenal_note = arsenal_match_factor(batter.get("statcast") or {}, pitcher.get("statcast") or {})
    lineup = build_full_lineup_context(
        team_id=team_id, target_player_id=player_id, target_slot=slot, target_profile=batter,
        pitcher_profile=pitcher, bullpen_model=bullpen, game_ctx=game_ctx,
        is_home=bool(game.get("is_home")), season=season, as_of_date=game_date,
        park=park, weather=weather, target_arsenal_factor=arsenal_factor,
    )
    slot = int(lineup.get("target_index", slot - 1)) + 1
    lineup_status = str(lineup.get("lineup_status") or lineup_status)
    pa_mean, pa_dist = projected_pa(slot, bool(game.get("is_home")), env.get("implied_runs", LEAGUE["runs_per_game"]), lineup.get("lineup_obp", LEAGUE["obp"]))

    current_pa = safe_float(batter.get("current_pa"), 0) or 0
    pitcher_q = safe_float(pitcher.get("data_quality"), 25) or 25
    lineup_q = safe_float(lineup.get("lineup_data_quality"), 30) or 30
    bullpen_q = safe_float(bullpen.get("data_quality"), 25) or 25
    uncertainty = clamp(42 + current_pa * 0.24 + pitcher_q * 0.42 + lineup_q * 0.16 + bullpen_q * 0.12, 40, 260)
    seed = stable_seed(game_date, player_id, line, MODEL_VERSION, slot, pitcher_id, tuple(lineup.get("lineup_ids") or []))
    sim = simulate_player_games(
        target_index=int(lineup.get("target_index", slot - 1)),
        is_home=bool(game.get("is_home")),
        lineup_starter_probs=lineup.get("starter_probs") or [],
        lineup_fallback_bullpen_probs=lineup.get("fallback_bullpen_probs") or [],
        individual_bullpen_probs=lineup.get("individual_bullpen_probs") or [],
        relievers=lineup.get("relievers") or [],
        expected_starter_bf=pitcher.get("expected_bf", 22.0),
        opponent_runs_mean=env.get("opponent_runs_mean", LEAGUE["runs_per_game"]),
        simulations=screen_sims,
        seed=seed,
        speed_boosts=lineup.get("speed_boosts") or [0.0] * 9,
        uncertainty_strength=uncertainty,
    )
    summary = summarize_sim(sim, line, pa_dist)
    side = "OVER" if summary["over_prob"] >= summary["under_prob"] else "UNDER"
    sim_pick_prob = summary["over_prob"] if side == "OVER" else summary["under_prob"]
    matchup_scale = clamp(summary["projection"] / max(batter["recent"].get("season_hrr_avg") or summary["projection"], 0.6), 0.85, 1.15)
    batter_logs = batter.get("logs") if isinstance(batter.get("logs"), pd.DataFrame) else pd.DataFrame()
    empirical = empirical_probability(batter_logs, line, side, matchup_scale)
    if empirical is None:
        empirical = 0.5
    disagreement = abs(sim_pick_prob - empirical)
    preliminary = 0.76 * sim_pick_prob + 0.24 * empirical
    edge = summary["projection"] - line
    row_base = {
        "Player ID": player_id, "GamePk": game.get("game_pk"), "Pitcher ID": pitcher_id,
        "Batter PA": current_pa, "Batter Statcast": bool((batter.get("statcast") or {}).get("available")),
        "Pitcher Statcast": bool((pitcher.get("statcast") or {}).get("available")), "Lineup Status": lineup_status,
        "Full Lineup Count": len(lineup.get("profiles") or []), "Lineup Data Quality": lineup_q,
        "Bullpen Available Count": bullpen.get("available_count", 0),
        "Weather Available": bool(weather_raw) or bool(weather.get("source")),
        "Historical Batter Profile": bool(batter.get("offline_profile")),
        "Historical Batter PA": round(safe_float(batter.get("historical_pa"), 0) or 0, 0),
        "Historical Batter Source": batter.get("historical_source") or "—",
        "Historical Pitcher Profile": bool(pitcher.get("offline_profile")),
    }
    dq, dq_label = data_quality_score(row_base)
    calibrated_raw = clamp(0.5 + (preliminary - 0.5) * clamp(0.56 + dq / 175, 0.62, 1.08), 0.34, 0.77)
    calibrated, historical_cal_adj, historical_cal_note = historical_probability_calibration(calibrated_raw)
    role_risk = 0.0
    role_notes = []
    if lineup_status != "CONFIRMED":
        role_risk += 15; role_notes.append("lineup projected")
    if not pitcher_id:
        role_risk += 25; role_notes.append("probable pitcher TBD")
    if slot >= 7:
        role_risk += 9; role_notes.append("lower batting order")
    if pa_mean < 4.0:
        role_risk += 10; role_notes.append("limited PA projection")
    if current_pa < 45:
        role_risk += 14; role_notes.append("thin current-season sample")
    if bullpen.get("available_count", 0) < 3:
        role_risk += 9; role_notes.append("bullpen depth uncertain")
    if len(lineup.get("profiles") or []) < 9:
        role_risk += 12; role_notes.append("incomplete lineup context")
    if abs(summary.get("pa", pa_mean) - pa_mean) > 0.16:
        role_risk += 5; role_notes.append("PA calibration gap")
    grade, grade_note = select_grade(side, calibrated, edge, dq, lineup_status, disagreement, summary["std"], role_risk)
    odds = saved_odds_lookup(player_name, game_date, line)
    market_prob = no_vig_probability(odds.get("Over Odds"), odds.get("Under Odds"), side)
    market_edge = calibrated - market_prob if market_prob is not None else None
    market_agreement = "NO ODDS"
    if market_prob is not None:
        market_agreement = "AGREE" if market_prob >= 0.515 else "DISAGREE"
        if market_prob < 0.48 and grade in {"🔥 ATTACK", "✅ OFFICIAL"}:
            grade, grade_note = "⚠️ PLAYABLE", "Downgraded: sportsbook no-vig market disagrees"

    vuln = pitcher_vulnerability(pitcher)
    recent = batter.get("recent") or {}
    expected_sp_pa = clamp(1 + (safe_float(pitcher.get("expected_bf"), 22) - slot) / 9.0, 0.0, pa_mean)
    starter_exposure = expected_sp_pa / max(pa_mean, 1e-6)
    factors = [
        f"Slot {slot} ({lineup_status.lower()})",
        f"PA model {pa_mean:.2f} → sim {summary['pa']:.2f}",
        f"Nine-hitter lineup OBP {lineup.get('lineup_obp', LEAGUE['obp']):.3f}",
        f"Team runs {env.get('implied_runs')}",
        f"Pitcher vulnerability {vuln.get('Overall')}",
        arsenal_note,
        f"Bullpen {bullpen.get('available_count',0)} available / x{bullpen.get('quality_factor',1.0):.3f}",
        f"Park HR x{park.get('HR',1.0):.2f}",
        f"Weather carry x{weather.get('carry_factor',1.0):.3f}",
    ]

    return {
        "Date": game_date, "Player": player_name, "Player ID": player_id,
        "Team": game.get("team") or team.get("name") or "—", "Opponent": game.get("opponent") or "—",
        "Matchup": f"{game.get('away','—')} @ {game.get('home','—')}" if game else "No MLB game match",
        "GamePk": game.get("game_pk"), "Start Time": game.get("start_time"), "Venue": venue or "—",
        "Source": ud.get("Source"), "Market": "Hits + Runs + RBIs", "Line": line,
        "Projection": summary["projection"], "Median": summary["median"], "Mode": summary["mode"],
        "Expected H": summary["hits"], "Expected R": summary["runs"], "Expected RBI": summary["rbi"],
        "Projected PA": summary["pa"], "Raw Sim PA": summary.get("raw_pa"), "PA Model Mean": pa_mean, "PA Distribution": pa_dist,
        "PA Calibration ESS": summary.get("pa_calibration_ess"), "PA Model Connected": True,
        "P10": summary["p10"], "P90": summary["p90"], "Volatility": summary["std"],
        "Over Probability %": round(summary["over_prob"] * 100, 1), "Under Probability %": round(summary["under_prob"] * 100, 1), "Push Probability %": round(summary["push_prob"] * 100, 1),
        "Pick": side, "Pick Probability %": round(calibrated * 100, 1), "Raw Calibrated Probability %": round(calibrated_raw * 100, 1),
        "Historical Calibration Adjustment %": round(historical_cal_adj * 100, 2), "Historical Calibration Note": historical_cal_note,
        "Fair Odds": fair_american(calibrated),
        "Edge": round(edge, 2), "Grade": grade, "Grade Note": grade_note,
        "Data Quality": dq, "Data Quality Label": dq_label, "Model Agreement Gap": round(disagreement * 100, 1),
        "Simulation Probability %": round(sim_pick_prob * 100, 1), "Independent Baseline %": round(empirical * 100, 1),
        "Lineup Slot": slot, "Lineup Status": lineup_status, "Full Lineup Count": len(lineup.get("profiles") or []),
        "Lineup Names": " | ".join(lineup.get("lineup_names") or []), "Lineup OBP": round(lineup.get("lineup_obp", LEAGUE["obp"]), 3),
        "Lineup Data Quality": round(lineup_q, 1), "Role Risk": round(role_risk, 1), "Role Risk Note": "; ".join(role_notes) or "Low",
        "Pitcher": pitcher.get("name") or "TBD", "Pitcher ID": pitcher_id, "Pitcher Hand": pitcher.get("hand") or pitcher_hand,
        "Starter Expected BF": pitcher.get("expected_bf"), "Starter Exposure %": round(starter_exposure * 100, 1),
        "Pitcher Vulnerability": vuln.get("Overall"), "Contact Allowed Score": vuln.get("Contact Allowed"), "Damage Allowed Score": vuln.get("Damage Allowed"), "Traffic Allowed Score": vuln.get("Traffic Allowed"),
        "Bullpen Available Count": bullpen.get("available_count"), "Bullpen IDs": json.dumps(bullpen.get("ids") or []),
        "Bullpen Pitches 3D": bullpen.get("pitches_3d"), "Bullpen Fatigue Factor": bullpen.get("fatigue_factor"),
        "Bullpen Quality Factor": bullpen.get("quality_factor"), "Bullpen Data Quality": bullpen.get("data_quality"),
        "Team Implied Runs": env.get("implied_runs"), "Team OPS": env.get("ops"), "Team Runs/G": env.get("runs_pg"),
        "Temperature": weather.get("temp"), "Humidity": weather.get("humidity"), "Pressure": weather.get("pressure"),
        "Wind": weather.get("wind"), "Wind Alignment": weather.get("wind_alignment"), "Weather": weather.get("condition"),
        "Roof Status": weather.get("roof_status"), "Weather Source": weather.get("source"), "Weather Carry": weather.get("carry_factor"),
        "Park 1B": park.get("1B", 1.0), "Park 2B": park.get("2B", 1.0), "Park 3B": park.get("3B", 1.0), "Park HR": park.get("HR", 1.0),
        "Batter PA": current_pa, "Batter Statcast Rows": (batter.get("statcast") or {}).get("rows"), "Pitcher Statcast Rows": (pitcher.get("statcast") or {}).get("rows"),
        "Historical Batter Profile": bool(batter.get("offline_profile")), "Historical Pitcher Profile": bool(pitcher.get("offline_profile")),
        "xBA": round(safe_float((batter.get("statcast") or {}).get("xba"), 0), 3) if safe_float((batter.get("statcast") or {}).get("xba")) is not None else None,
        "xwOBA": round(safe_float((batter.get("statcast") or {}).get("xwoba"), 0), 3) if safe_float((batter.get("statcast") or {}).get("xwoba")) is not None else None,
        "Hard Hit %": round((safe_float((batter.get("statcast") or {}).get("hard_hit_pct"), 0) or 0) * 100, 1) if (batter.get("statcast") or {}).get("hard_hit_pct") is not None else None,
        "Barrel %": round((safe_float((batter.get("statcast") or {}).get("barrel_pct"), 0) or 0) * 100, 1) if (batter.get("statcast") or {}).get("barrel_pct") is not None else None,
        "Contact %": round((safe_float((batter.get("statcast") or {}).get("contact_pct"), 0) or 0) * 100, 1) if (batter.get("statcast") or {}).get("contact_pct") is not None else None,
        "L5 HRR Avg": recent.get("l5_avg"), "L10 HRR Avg": recent.get("l10_avg"), "L20 HRR Avg": recent.get("l20_avg"),
        "L5 Over %": round((recent.get("l5_over") or 0) * 100, 1) if recent.get("l5_over") is not None else None,
        "L10 Over %": round((recent.get("l10_over") or 0) * 100, 1) if recent.get("l10_over") is not None else None,
        "Season HRR Avg": recent.get("season_hrr_avg"), "Season Over %": round((recent.get("season_over") or 0) * 100, 1) if recent.get("season_over") is not None else None,
        "Over Odds": odds.get("Over Odds"), "Under Odds": odds.get("Under Odds"), "Market No-Vig %": round(market_prob * 100, 1) if market_prob is not None else None,
        "Market Edge %": round(market_edge * 100, 1) if market_edge is not None else None, "Market Agreement": market_agreement,
        "Backtest Cutoff": f"Before {game_date}", "Leakage Safe": True,
        "Main Factors": " • ".join(factors), "Simulations": screen_sims, "Model Version": MODEL_VERSION,
        "Timestamp": now_iso(),
    }

def apply_learning_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Expose learned projection bias without changing an already classified row."""
    if df.empty:
        return df
    learning = read_json(LEARNING_FILE, {})
    global_cal = learning.get("global") or {}
    n = safe_float(global_cal.get("n"), 0) or 0
    bias = safe_float(global_cal.get("projection_bias"), 0) or 0
    suggested = clamp(-bias * min(n / 150, 1.0), -0.22, 0.22) if n >= 25 else 0.0
    df = df.copy()
    df["Learning Adjustment"] = 0.0
    df["Suggested Projection Bias Correction"] = round(suggested, 3)
    return df

def build_board(lines: Sequence[Dict[str, Any]], game_date: str, screen_sims: int = 5000) -> pd.DataFrame:
    season = pd.Timestamp(game_date).year
    opening_day = discover_opening_day(season)
    rows: List[Dict[str, Any]] = []
    progress = st.progress(0, text="Building H+R+RBI profiles…") if lines else None
    for i, ud in enumerate(lines):
        try:
            rows.append(build_one_projection(dict(ud), game_date, season, opening_day, screen_sims))
        except Exception as exc:
            rows.append({"Date": game_date, "Player": ud.get("Player"), "Player ID": ud.get("Player ID"), "Line": ud.get("Line"), "Source": ud.get("Source"), "Grade": "🚫 PASS", "Grade Note": f"Projection error: {exc}", "Data Quality": 0, "Timestamp": now_iso()})
        if progress:
            progress.progress((i + 1) / max(len(lines), 1), text=f"Projecting {i+1}/{len(lines)} — {ud.get('Player')}")
    if progress:
        progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = apply_learning_adjustment(df)
    grade_order = {"🔥 ATTACK": 0, "✅ OFFICIAL": 1, "⚠️ PLAYABLE": 2, "👀 TRACK ONLY": 3, "🚫 PASS": 4}
    df["_grade_order"] = df["Grade"].map(grade_order).fillna(9)
    df = df.sort_values(["_grade_order", "Pick Probability %", "Data Quality"], ascending=[True, False, False]).drop(columns=["_grade_order"]).reset_index(drop=True)
    # Persist compact active-board profiles for later debugging/learning.
    try:
        df.to_csv(CACHE_DIR / f"active_hrr_board_{game_date}.csv", index=False)
    except Exception:
        pass
    return df

# ============================================================
# SAVE / GRADE / LEARN / GITHUB
# ============================================================
def row_snapshot(row: pd.Series) -> Dict[str, Any]:
    keep = [
        "Date", "Player", "Player ID", "Team", "Opponent", "Matchup", "GamePk", "Start Time", "Venue",
        "Source", "Market", "Line", "Projection", "Expected H", "Expected R", "Expected RBI", "Projected PA",
        "Raw Sim PA", "PA Model Mean", "PA Distribution", "PA Calibration ESS", "PA Model Connected",
        "Over Probability %", "Under Probability %", "Push Probability %", "Pick", "Pick Probability %", "Fair Odds",
        "Edge", "Grade", "Grade Note", "Data Quality", "Data Quality Label", "Model Agreement Gap", "Simulation Probability %", "Independent Baseline %", "Lineup Slot",
        "Lineup Status", "Full Lineup Count", "Lineup Names", "Lineup OBP", "Lineup Data Quality", "Role Risk", "Pitcher", "Pitcher ID", "Pitcher Hand", "Pitcher Vulnerability",
        "Bullpen Available Count", "Bullpen IDs", "Bullpen Quality Factor", "Bullpen Data Quality",
        "Team Implied Runs", "Temperature", "Humidity", "Pressure", "Wind", "Weather", "Roof Status", "Weather Source", "Park HR", "L5 HRR Avg", "L10 HRR Avg",
        "Season HRR Avg", "Over Odds", "Under Odds", "Market No-Vig %", "Market Edge %", "Market Agreement",
        "Historical Batter Profile", "Historical Pitcher Profile", "Backtest Cutoff", "Leakage Safe",
        "Main Factors", "Simulations", "Model Version", "Timestamp",
    ]
    result = {k: row.get(k) for k in keep if k in row.index}
    result["Snapshot ID"] = hashlib.sha1(f"{result.get('Date')}|{normalize_name(result.get('Player'))}|{result.get('Line')}|{result.get('Pick')}|{result.get('Model Version')}".encode()).hexdigest()[:16]
    result["Saved At"] = now_iso()
    return result


def save_official_board(df: pd.DataFrame, include_playable: bool = False) -> Dict[str, int]:
    if df.empty:
        return {"added": 0, "duplicates": 0}
    allowed = {"🔥 ATTACK", "✅ OFFICIAL"}
    if include_playable:
        allowed.add("⚠️ PLAYABLE")
    existing = read_json(PICK_LOG, [])
    keys = {r.get("Snapshot ID") for r in existing}
    added = duplicates = 0
    for _, row in df[df["Grade"].isin(allowed)].iterrows():
        snap = row_snapshot(row)
        if snap["Snapshot ID"] in keys:
            duplicates += 1
            continue
        existing.append(snap)
        keys.add(snap["Snapshot ID"])
        added += 1
    write_json(PICK_LOG, existing)
    github_backup_files([PICK_LOG])
    return {"added": added, "duplicates": duplicates}


def actual_hrr_for_date(player_id: int, game_date: str, game_pk: Optional[int] = None) -> Optional[Dict[str, Any]]:
    season = pd.Timestamp(game_date).year
    logs = player_game_log(player_id, "hitting", season)
    if logs.empty:
        return None
    target = pd.Timestamp(game_date).date()
    sub = logs[logs["Date"].dt.date.eq(target)]
    if game_pk and "GamePk" in sub:
        exact = sub[pd.to_numeric(sub["GamePk"], errors="coerce").eq(int(game_pk))]
        if not exact.empty:
            sub = exact
    if sub.empty:
        return None
    r = sub.iloc[-1]
    return {"H": int(r["H"]), "R": int(r["R"]), "RBI": int(r["RBI"]), "HRR": int(r["HRR"]), "PA": int(r["PA"]), "GamePk": safe_int(r.get("GamePk"))}

def grade_saved_picks(force: bool = False) -> Dict[str, int]:
    picks = read_json(PICK_LOG, [])
    results = read_json(RESULT_LOG, [])
    result_ids = {r.get("Snapshot ID") for r in results}
    graded = pushes = pending = skipped = 0
    for pick in picks:
        sid = pick.get("Snapshot ID")
        if sid in result_ids and not force:
            skipped += 1
            continue
        actual = actual_hrr_for_date(safe_int(pick.get("Player ID")), str(pick.get("Date")), safe_int(pick.get("GamePk")))
        if actual is None:
            pending += 1
            continue
        line, side, value = float(pick.get("Line")), str(pick.get("Pick")), actual["HRR"]
        outcome = "PUSH" if value == line else ("WIN" if (value > line if side == "OVER" else value < line) else "LOSS")
        row = dict(pick)
        row.update({"Actual H": actual["H"], "Actual R": actual["R"], "Actual RBI": actual["RBI"], "Actual HRR": actual["HRR"], "Actual PA": actual["PA"], "Result": outcome, "Projection Error": round(value - safe_float(pick.get("Projection"), 0), 2), "Graded At": now_iso()})
        if sid in result_ids:
            results = [r for r in results if r.get("Snapshot ID") != sid]
        results.append(row)
        result_ids.add(sid)
        graded += 1
        pushes += outcome == "PUSH"
    write_json(RESULT_LOG, results)
    update_learning(results)
    sync_graded_csv(results)
    github_backup_files([RESULT_LOG, LEARNING_FILE, GRADED_CSV])
    return {"graded": graded, "pushes": pushes, "pending": pending, "skipped": skipped}


def _record_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    non_push = [r for r in rows if r.get("Result") in {"WIN", "LOSS"}]
    if not non_push:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": None}
    wins = sum(r.get("Result") == "WIN" for r in non_push)
    return {"n": len(non_push), "wins": wins, "losses": len(non_push) - wins, "win_rate": round(wins / len(non_push), 4)}


def _bootstrap_win_rate_ci(rows: Sequence[Dict[str, Any]], samples: int = 2000) -> Optional[List[float]]:
    ys = np.array([1.0 if r.get("Result") == "WIN" else 0.0 for r in rows if r.get("Result") in {"WIN", "LOSS"}], dtype=float)
    if len(ys) < 8:
        return None
    rng = np.random.default_rng(stable_seed("hrr_bootstrap", len(ys), float(ys.sum()), MODEL_VERSION))
    draws = rng.choice(ys, size=(samples, len(ys)), replace=True).mean(axis=1)
    return [round(float(np.quantile(draws, 0.025)), 4), round(float(np.quantile(draws, 0.975)), 4)]


def calculate_validation_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    finished = [r for r in results if r.get("Result") in {"WIN", "LOSS", "PUSH"}]
    non_push = [r for r in finished if r.get("Result") in {"WIN", "LOSS"}]
    errors = [safe_float(r.get("Projection Error"), 0) or 0 for r in finished]
    probs, ys = [], []
    for r in non_push:
        p = clamp((safe_float(r.get("Pick Probability %"), 50) or 50) / 100.0, 0.001, 0.999)
        probs.append(p)
        ys.append(1.0 if r.get("Result") == "WIN" else 0.0)
    probs_arr, ys_arr = np.array(probs, dtype=float), np.array(ys, dtype=float)
    brier = float(np.mean((probs_arr - ys_arr) ** 2)) if len(probs_arr) else None
    log_loss = float(-np.mean(ys_arr * np.log(probs_arr) + (1 - ys_arr) * np.log(1 - probs_arr))) if len(probs_arr) else None
    calibration: Dict[str, Any] = {}
    ece = 0.0
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.78)]
    for low, high in bins:
        idx = [i for i, p in enumerate(probs) if low <= p < high or (high == 0.78 and low <= p <= high)]
        if not idx:
            continue
        avg_p = float(np.mean([probs[i] for i in idx]))
        actual = float(np.mean([ys[i] for i in idx]))
        key = f"{int(low*100)}-{int(high*100)}%"
        calibration[key] = {"n": len(idx), "avg_probability": round(avg_p, 4), "win_rate": round(actual, 4), "gap": round(actual - avg_p, 4)}
        ece += len(idx) / max(len(probs), 1) * abs(actual - avg_p)

    def grouped(field: str, transform=None) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in non_push:
            value = r.get(field)
            if transform:
                value = transform(value, r)
            key = str(value if value not in {None, "", "nan"} else "Unknown")
            groups[key].append(r)
        return {k: _record_summary(v) for k, v in sorted(groups.items()) if v}

    line_group = lambda value, _: f"Line {safe_float(value, 0):g}" if safe_float(value) is not None else "Unknown"
    slot_group = lambda value, _: f"Slot {safe_int(value)}" if safe_int(value) else "Unknown"
    pitcher_tier = lambda value, row: "High vulnerability" if (safe_float(row.get("Pitcher Vulnerability"), 50) or 50) >= 62 else "Low vulnerability" if (safe_float(row.get("Pitcher Vulnerability"), 50) or 50) <= 42 else "Average vulnerability"
    global_row = _record_summary(non_push)
    global_row.update({
        "total_with_pushes": len(finished),
        "pushes": sum(r.get("Result") == "PUSH" for r in finished),
        "win_rate_ci_95": _bootstrap_win_rate_ci(non_push),
        "projection_bias": round(float(np.mean(errors)), 4) if errors else 0.0,
        "mae": round(float(np.mean(np.abs(errors))), 4) if errors else None,
        "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 4) if errors else None,
        "brier_score": round(brier, 5) if brier is not None else None,
        "log_loss": round(log_loss, 5) if log_loss is not None else None,
        "expected_calibration_error": round(float(ece), 5) if probs else None,
    })
    return {
        "updated": now_iso(), "global": global_row, "calibration": calibration,
        "by_grade": grouped("Grade"), "by_pick": grouped("Pick"),
        "by_line": grouped("Line", line_group), "by_lineup_slot": grouped("Lineup Slot", slot_group),
        "by_pitcher_tier": grouped("Pitcher Vulnerability", pitcher_tier),
        "by_data_quality": grouped("Data Quality Label"), "by_lineup_status": grouped("Lineup Status"),
        "by_model_version": grouped("Model Version"),
    }


def update_learning(results: Sequence[Dict[str, Any]]) -> None:
    metrics = calculate_validation_metrics(results)
    # Keep backward-compatible grade key for older UI/workflows.
    metrics["grades"] = metrics.get("by_grade", {})
    write_json(LEARNING_FILE, metrics, protect=False)

def sync_graded_csv(results: Sequence[Dict[str, Any]]) -> None:
    try:
        pd.DataFrame(results).to_csv(GRADED_CSV, index=False)
    except Exception:
        pass


def github_backup_files(paths: Sequence[Path]) -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", "main")
    base_path = os.getenv("GITHUB_DATA_PATH", "learning_data")
    if not token or not repo:
        return {str(p): "GitHub not configured" for p in paths}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    status: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            status[str(path)] = "missing"
            continue
        remote = f"{base_path.strip('/')}/{path.name}"
        url = f"https://api.github.com/repos/{repo}/contents/{remote}"
        sha = None
        try:
            get = requests.get(url, headers=headers, params={"ref": branch}, timeout=20)
            if get.status_code == 200:
                sha = get.json().get("sha")
            payload = {"message": f"Update {path.name} from {APP_VERSION}", "content": base64.b64encode(path.read_bytes()).decode(), "branch": branch}
            if sha:
                payload["sha"] = sha
            put = requests.put(url, headers=headers, json=payload, timeout=30)
            status[str(path)] = "uploaded" if put.status_code in {200, 201} else f"HTTP {put.status_code}: {put.text[:100]}"
        except Exception as exc:
            status[str(path)] = f"error: {exc}"
    return status

# ============================================================
# UI HELPERS
# ============================================================
def display_card(row: pd.Series) -> None:
    side = str(row.get("Pick") or "PASS")
    css = "pick-over" if side == "OVER" else "pick-under" if side == "UNDER" else "pick-pass"
    player = html.escape(str(row.get("Player") or ""))
    grade = html.escape(str(row.get("Grade") or ""))
    matchup = html.escape(str(row.get("Matchup") or ""))
    factors = html.escape(str(row.get("Main Factors") or ""))
    st.markdown(
        f"""
<div class="hrr-card">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
    <div><div class="player">{player}</div><div class="sub">{matchup} · {html.escape(str(row.get('Pitcher','TBD')))} ({row.get('Pitcher Hand','—')}) · Line {row.get('Line')}</div></div>
    <div style="text-align:right"><div class="{css}">{side} {row.get('Line')}</div><div class="sub">{grade} · {row.get('Pick Probability %','—')}% · Fair {row.get('Fair Odds','—')}</div></div>
  </div>
  <div class="metric-grid">
    <div class="metric"><div class="k">Projection</div><div class="v">{row.get('Projection','—')}</div></div>
    <div class="metric"><div class="k">H / R / RBI</div><div class="v">{row.get('Expected H','—')} / {row.get('Expected R','—')} / {row.get('Expected RBI','—')}</div></div>
    <div class="metric"><div class="k">Projected PA</div><div class="v">{row.get('Projected PA','—')}</div></div>
    <div class="metric"><div class="k">Edge</div><div class="v">{row.get('Edge','—')}</div></div>
    <div class="metric"><div class="k">Data Quality</div><div class="v">{row.get('Data Quality','—')}</div></div>
    <div class="metric"><div class="k">Volatility</div><div class="v">{row.get('Volatility','—')}</div></div>
  </div>
  <div class="small-note" style="margin-top:11px">{factors}</div>
  <div class="small-note" style="margin-top:6px">L5 {row.get('L5 HRR Avg','—')} · L10 {row.get('L10 HRR Avg','—')} · Season {row.get('Season HRR Avg','—')} · Lineup {row.get('Lineup Slot','—')} ({row.get('Lineup Status','—')}) · Market {row.get('Market Agreement','NO ODDS')}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def board_table_columns(df: pd.DataFrame) -> List[str]:
    desired = [
        "Player", "Matchup", "Pitcher", "Line", "Pick", "Grade", "Projection", "Pick Probability %", "Fair Odds", "Edge",
        "Expected H", "Expected R", "Expected RBI", "Projected PA", "PA Model Mean", "Lineup Slot", "Lineup Status", "Full Lineup Count", "Team Implied Runs",
        "Starter Exposure %", "Pitcher Vulnerability", "Bullpen Available Count", "Bullpen Quality Factor", "Data Quality", "Volatility", "L5 HRR Avg", "L10 HRR Avg", "Season HRR Avg",
        "Market No-Vig %", "Market Edge %", "Market Agreement", "Role Risk", "Grade Note",
    ]
    return [c for c in desired if c in df.columns]

# ============================================================
# APP STATE / SIDEBAR
# ============================================================
st.markdown(
    f"""
<div class="hero"><h1>⚾ ONE WAY PICKZ — HITS + RUNS + RBI</h1><p>{APP_VERSION} · Underdog players only · connected PA model + real lineups + role-weighted bullpens + leakage-safe Monte Carlo</p></div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Daily Slate")
    selected_date = st.date_input("Game date", value=la_now().date()).isoformat()
    sims = st.select_slider("Screening simulations per player", options=[2048, 4096, 5000, 8192, 10000, 16384], value=5000)
    use_manual_when_empty = st.checkbox("Use saved manual lines when Underdog is empty", value=True)
    refresh = st.button("🔄 Pull lines + build projections", use_container_width=True, type="primary")
    st.caption("The engine starts with active Underdog H+R+RBI players, then loads only their real nine-hitter lineups, opposing starter and likely available relievers.")
    st.divider()
    st.subheader("Model controls")
    show_passes = st.checkbox("Show PASS rows", value=False)
    include_playable_save = st.checkbox("Include Playable when saving", value=False)
    st.caption("Pregame cutoff is exclusive: historical slates use only information available before the selected game date. L5/L10 remain supporting context.")

if "hrr_board" not in st.session_state:
    st.session_state["hrr_board"] = pd.DataFrame()
if "hrr_lines" not in st.session_state:
    st.session_state["hrr_lines"] = []
if "hrr_last_refresh" not in st.session_state:
    st.session_state["hrr_last_refresh"] = None

if refresh:
    with st.spinner("Pulling active Underdog H+R+RBI lines…"):
        lines = fetch_underdog_hrr_rows()
        if not lines and use_manual_when_empty:
            lines = load_manual_lines()
        st.session_state["hrr_lines"] = lines
    if lines:
        st.session_state["hrr_board"] = build_board(lines, selected_date, int(sims))
        st.session_state["hrr_last_refresh"] = now_iso()
    else:
        st.session_state["hrr_board"] = pd.DataFrame()
        st.session_state["hrr_last_refresh"] = now_iso()

board = st.session_state.get("hrr_board", pd.DataFrame())
lines = st.session_state.get("hrr_lines", [])

# ============================================================
# TABS
# ============================================================
tab_board, tab_official, tab_grade, tab_data, tab_debug = st.tabs([
    "🎯 H+R+RBI BOARD", "✅ OFFICIAL PICKS", "📊 AFTER GAMES / LEARNING", "🗂️ DATA MANAGER", "🧪 DEBUG / SETTINGS"
])

with tab_board:
    st.markdown('<div class="section">Daily H+R+RBI Board</div>', unsafe_allow_html=True)
    st.caption(f"Last refresh: {st.session_state.get('hrr_last_refresh') or 'Not refreshed'} · Opening Day source auto-detected · Active line rows: {len(lines)}")
    if board.empty:
        st.warning("No board loaded. Pull the slate from the sidebar. If Underdog is unavailable, add real H+R+RBI lines in Data Manager—no synthetic lines will be created.")
    else:
        display = board if show_passes else board[~board["Grade"].astype(str).eq("🚫 PASS")]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("UD Players", len(board))
        c2.metric("Attack", int((board["Grade"] == "🔥 ATTACK").sum()))
        c3.metric("Official", int((board["Grade"] == "✅ OFFICIAL").sum()))
        c4.metric("Playable", int((board["Grade"] == "⚠️ PLAYABLE").sum()))
        c5.metric("Median Data", f"{board['Data Quality'].median():.0f}" if "Data Quality" in board else "—")
        for _, row in display.head(35).iterrows():
            display_card(row)
        st.markdown("#### Full table")
        st.dataframe(display[board_table_columns(display)], use_container_width=True, hide_index=True)
        st.download_button("Download current H+R+RBI board CSV", data=board.to_csv(index=False).encode(), file_name=f"hrr_board_{selected_date}.csv", mime="text/csv", use_container_width=True)

with tab_official:
    st.markdown('<div class="section">Official Selection Gate</div>', unsafe_allow_html=True)
    if board.empty:
        st.info("Build the board first.")
    else:
        allowed = ["🔥 ATTACK", "✅ OFFICIAL"] + (["⚠️ PLAYABLE"] if include_playable_save else [])
        official = board[board["Grade"].isin(allowed)].copy()
        st.dataframe(official[board_table_columns(official)], use_container_width=True, hide_index=True)
        if st.button("💾 Save official pregame snapshot", type="primary", use_container_width=True):
            status = save_official_board(board, include_playable=include_playable_save)
            st.success(f"Saved {status['added']} new picks; skipped {status['duplicates']} duplicates.")
        saved = read_json(PICK_LOG, [])
        if saved:
            st.markdown("#### Saved snapshots")
            st.dataframe(pd.DataFrame(saved).tail(100), use_container_width=True, hide_index=True)

with tab_grade:
    st.markdown('<div class="section">Automatic Grading, Calibration and Validation</div>', unsafe_allow_html=True)
    if st.button("🏁 Grade finished H+R+RBI picks", type="primary", use_container_width=True):
        status = grade_saved_picks(force=False)
        st.success(f"Graded {status['graded']} · Pushes {status['pushes']} · Pending {status['pending']} · Already graded {status['skipped']}")
    results = read_json(RESULT_LOG, [])
    if results:
        rdf = pd.DataFrame(results)
        metrics = calculate_validation_metrics(results)
        global_m = metrics.get("global") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Graded", global_m.get("total_with_pushes", len(rdf)))
        c2.metric("Record", f"{global_m.get('wins',0)}-{global_m.get('losses',0)}")
        c3.metric("Win Rate", f"{(global_m.get('win_rate') or 0)*100:.1f}%" if global_m.get("win_rate") is not None else "—")
        ci = global_m.get("win_rate_ci_95")
        c4.metric("95% Win-Rate CI", f"{ci[0]*100:.1f}%–{ci[1]*100:.1f}%" if ci else "Need 8+ picks")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Brier Score", f"{global_m.get('brier_score'):.4f}" if global_m.get("brier_score") is not None else "—")
        c6.metric("Log Loss", f"{global_m.get('log_loss'):.4f}" if global_m.get("log_loss") is not None else "—")
        c7.metric("Calibration Error", f"{global_m.get('expected_calibration_error'):.4f}" if global_m.get("expected_calibration_error") is not None else "—")
        c8.metric("Projection MAE", f"{global_m.get('mae'):.2f}" if global_m.get("mae") is not None else "—")

        st.markdown("#### Probability calibration")
        calibration = metrics.get("calibration") or {}
        if calibration:
            cal_rows = [{"Probability Bucket": k, **v} for k, v in calibration.items()]
            cal_df = pd.DataFrame(cal_rows).rename(columns={"n": "Picks", "avg_probability": "Average Predicted", "win_rate": "Actual Win Rate", "gap": "Actual - Predicted"})
            for col in ["Average Predicted", "Actual Win Rate", "Actual - Predicted"]:
                if col in cal_df:
                    cal_df[col] = (pd.to_numeric(cal_df[col], errors="coerce") * 100).round(1)
            st.dataframe(cal_df, use_container_width=True, hide_index=True)
        else:
            st.info("Calibration buckets will appear after graded non-push selections are available.")

        st.markdown("#### Performance breakdowns")
        breakdown_map = {
            "Grade": "by_grade", "Over vs Under": "by_pick", "Line": "by_line",
            "Lineup Slot": "by_lineup_slot", "Pitcher Vulnerability": "by_pitcher_tier",
            "Data Quality": "by_data_quality", "Lineup Status": "by_lineup_status", "Model Version": "by_model_version",
        }
        selected_breakdown = st.selectbox("Breakdown", list(breakdown_map.keys()))
        breakdown = metrics.get(breakdown_map[selected_breakdown]) or {}
        if breakdown:
            bdf = pd.DataFrame([{"Group": k, **v} for k, v in breakdown.items()])
            if "win_rate" in bdf:
                bdf["win_rate"] = (pd.to_numeric(bdf["win_rate"], errors="coerce") * 100).round(1)
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        st.markdown("#### Graded selections")
        st.dataframe(rdf.sort_values("Graded At", ascending=False) if "Graded At" in rdf else rdf, use_container_width=True, hide_index=True)
        st.download_button("Download graded history CSV", data=rdf.to_csv(index=False).encode(), file_name="hrr_graded_history.csv", mime="text/csv", use_container_width=True)
        with st.expander("Raw learning JSON"):
            st.json(read_json(LEARNING_FILE, metrics))
    else:
        st.info("No graded H+R+RBI picks yet. Save an official snapshot before the games, then grade after final results post.")

with tab_data:
    st.markdown('<div class="section">Manual Lines, Odds and Profile Data</div>', unsafe_allow_html=True)
    st.caption("Manual fallback accepts only real H+R+RBI lines. Format: Player,Line")
    manual_default = MANUAL_LINES_FILE.read_text() if MANUAL_LINES_FILE.exists() else "Player,Line\n"
    manual_text = st.text_area("Manual Underdog H+R+RBI lines", value=manual_default, height=180)
    if st.button("Save manual lines", use_container_width=True):
        parsed = parse_manual_hrr_lines(manual_text)
        if parsed:
            MANUAL_LINES_FILE.write_text(manual_text)
            st.success(f"Saved {len(parsed)} valid MLB player lines.")
        else:
            st.error("No valid MLB H+R+RBI rows were detected.")

    st.markdown("#### Saved sportsbook odds / no-vig market")
    if board.empty:
        st.info("Build a board to edit market odds.")
    else:
        odds_existing = read_json(SAVED_ODDS_FILE, {})
        edit_rows = []
        for _, r in board.iterrows():
            key = f"{selected_date}|{normalize_name(r['Player'])}|{r['Line']}"
            saved = odds_existing.get(key, {})
            edit_rows.append({"Player": r["Player"], "Line": r["Line"], "Over Odds": saved.get("Over Odds"), "Under Odds": saved.get("Under Odds")})
        edited = st.data_editor(pd.DataFrame(edit_rows), use_container_width=True, hide_index=True, num_rows="fixed")
        if st.button("Save odds", use_container_width=True):
            for _, r in edited.iterrows():
                key = f"{selected_date}|{normalize_name(r['Player'])}|{r['Line']}"
                odds_existing[key] = {"Over Odds": safe_float(r.get("Over Odds")), "Under Odds": safe_float(r.get("Under Odds")), "Saved At": now_iso()}
            write_json(SAVED_ODDS_FILE, odds_existing, protect=False)
            github_backup_files([SAVED_ODDS_FILE])
            st.success("Saved market odds. Refresh the board to apply no-vig agreement and market edge.")

    st.markdown("#### Historical profile availability")
    hb, hp = historical_batter_profiles(), historical_pitcher_profiles()
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical batter profiles", len(hb))
    c2.metric("Historical pitcher profiles", len(hp))
    c3.metric("Current opening day", discover_opening_day(pd.Timestamp(selected_date).year))
    bundled_rows = int(pd.to_numeric(hb.get("historical_pa", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not hb.empty else 0
    st.caption(f"Bundled 2015-2024 batter priors matched by name: {bundled_rows}. The daily board still uses pregame 2026 data, 2025 priors, and targeted 2021-2025 Statcast profiles when available.")
    profile_state = read_json(PROFILE_BUILD_STATE, {})
    if profile_state:
        st.caption(f"Last targeted profile build: {profile_state.get('updated','—')} · Batters added {profile_state.get('batters_added',0)} · Pitchers added {profile_state.get('pitchers_added',0)}")
    if st.button("🧱 Build missing 2021-2025 profiles for this slate", use_container_width=True, disabled=board.empty):
        with st.spinner("Building only missing Underdog batter, starter and likely bullpen profiles…"):
            status = build_targeted_historical_profiles(board, include_bullpen=True)
        if status.get("errors"):
            st.warning(f"Built {status.get('batters',0)} batter and {status.get('pitchers',0)} pitcher profiles with {len(status.get('errors',[]))} source errors. See Debug.")
        else:
            st.success(f"Built {status.get('batters',0)} batter and {status.get('pitchers',0)} pitcher profiles. Refresh the board to apply them.")

    if board is not None and not board.empty:
        st.markdown("#### Active-player data audit")
        audit_cols = [c for c in ["Player", "Batter PA", "Batter Statcast Rows", "Historical Batter Profile", "Historical Batter PA", "Historical Batter Source", "Pitcher", "Pitcher Statcast Rows", "Historical Pitcher Profile", "Lineup Status", "Full Lineup Count", "Bullpen Available Count", "Weather Source", "Leakage Safe", "Data Quality", "Role Risk Note"] if c in board]
        st.dataframe(board[audit_cols], use_container_width=True, hide_index=True)

with tab_debug:
    st.markdown('<div class="section">Underdog Parser / Source Debug</div>', unsafe_allow_html=True)
    debug = st.session_state.get("hrr_ud_debug", {})
    st.json(debug)
    if debug.get("samples"):
        with st.expander("Underdog H+R+RBI market text samples"):
            for sample in debug.get("samples")[:12]:
                st.code(sample)
    st.markdown("#### Recent source requests/errors")
    requests_log = read_json(REQUEST_LOG_FILE, [])
    if requests_log:
        st.dataframe(pd.DataFrame(requests_log).tail(100), use_container_width=True, hide_index=True)
    else:
        st.info("No source errors recorded.")
    st.markdown("#### Storage / GitHub")
    st.code(str(LOCAL_DIR.resolve()))
    st.write({"Picks": str(PICK_LOG), "Results": str(RESULT_LOG), "Learning": str(LEARNING_FILE), "Odds": str(SAVED_ODDS_FILE)})
    st.caption("GitHub backup uses Railway secrets GITHUB_TOKEN (or GH_TOKEN), GITHUB_REPO, optional GITHUB_BRANCH and GITHUB_DATA_PATH.")
    if st.button("Back up current HRR files to GitHub", use_container_width=True):
        status = github_backup_files([PICK_LOG, RESULT_LOG, LEARNING_FILE, GRADED_CSV, SAVED_ODDS_FILE])
        st.json(status)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Streamlit data caches", use_container_width=True):
            st.cache_data.clear()
            st.success("Caches cleared.")
    with c2:
        if st.button("Clear current in-session board", use_container_width=True):
            st.session_state["hrr_board"] = pd.DataFrame()
            st.session_state["hrr_lines"] = []
            st.success("Current session board cleared; saved history was not deleted.")

st.caption("Educational projection tool only. Baseball outcomes are highly variable; no projection guarantees a result.")
