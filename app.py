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
4. Build Bayesian/regressed batter and pitcher profiles with optional Statcast.
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
APP_VERSION = "ONE WAY PICKZ MLB H+R+RBI v1.0"
MODEL_VERSION = "HRR_BAYES_GAMESTATE_MC_2026_07_12"
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
LINE_HISTORY_FILE = LOCAL_DIR / "hrr_line_history.json"
SAVED_ODDS_FILE = LOCAL_DIR / "hrr_saved_odds.json"
REQUEST_LOG_FILE = LOCAL_DIR / "hrr_request_log.json"
GRADED_CSV = LOCAL_DIR / "hrr_graded_history.csv"
MANUAL_LINES_FILE = LOCAL_DIR / "hrr_manual_lines.csv"

# Optional offline profiles produced from the user's planned 2021-2025 builder.
# The daily app runs without these files, but will use them automatically when present.
HIST_BATTER_PROFILE_CANDIDATES = [
    Path("data/batter_profiles.parquet"),
    Path("data/batter_profiles.csv"),
    Path("learning_data/batter_profiles.parquet"),
    Path("learning_data/batter_profiles.csv"),
    CACHE_DIR / "batter_profiles_2021_2025.parquet",
    CACHE_DIR / "batter_profiles_2021_2025.csv",
]
HIST_PITCHER_PROFILE_CANDIDATES = [
    Path("data/pitcher_profiles.parquet"),
    Path("data/pitcher_profiles.csv"),
    Path("learning_data/pitcher_profiles.parquet"),
    Path("learning_data/pitcher_profiles.csv"),
    CACHE_DIR / "pitcher_profiles_2021_2025.parquet",
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

# Conservative outcome-specific venue fallback factors. Values are deliberately
# capped in the projection engine. Unknown parks remain neutral.
PARK_FACTORS: Dict[str, Dict[str, float]] = {
    "Coors Field": {"1B": 1.05, "2B": 1.10, "3B": 1.08, "HR": 1.12, "R": 1.10},
    "Great American Ball Park": {"1B": 1.00, "2B": 1.01, "3B": 0.96, "HR": 1.10, "R": 1.05},
    "Fenway Park": {"1B": 1.02, "2B": 1.12, "3B": 0.96, "HR": 1.01, "R": 1.05},
    "Yankee Stadium": {"1B": 0.99, "2B": 0.95, "3B": 0.88, "HR": 1.08, "R": 1.02},
    "Citizens Bank Park": {"1B": 1.00, "2B": 1.00, "3B": 0.96, "HR": 1.07, "R": 1.04},
    "Dodger Stadium": {"1B": 0.98, "2B": 1.00, "3B": 0.95, "HR": 1.03, "R": 1.00},
    "Oracle Park": {"1B": 1.01, "2B": 1.05, "3B": 1.12, "HR": 0.90, "R": 0.97},
    "T-Mobile Park": {"1B": 0.98, "2B": 0.97, "3B": 0.98, "HR": 0.92, "R": 0.94},
    "Petco Park": {"1B": 0.99, "2B": 0.98, "3B": 0.98, "HR": 0.94, "R": 0.96},
    "loanDepot park": {"1B": 0.98, "2B": 0.98, "3B": 1.00, "HR": 0.93, "R": 0.95},
    "Globe Life Field": {"1B": 1.00, "2B": 1.02, "3B": 0.98, "HR": 1.02, "R": 1.01},
    "Truist Park": {"1B": 1.00, "2B": 1.01, "3B": 0.97, "HR": 1.04, "R": 1.02},
    "Chase Field": {"1B": 1.01, "2B": 1.04, "3B": 1.02, "HR": 1.01, "R": 1.02},
    "Wrigley Field": {"1B": 1.00, "2B": 1.02, "3B": 1.00, "HR": 1.00, "R": 1.00},
}

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
    for p in paths:
        if p.exists():
            df = safe_read_table(p)
            if not df.empty:
                return df
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def historical_batter_profiles() -> pd.DataFrame:
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
                    "HR": safe_float(stat.get("homeRuns"), 0) or 0,
                    "BB": safe_float(stat.get("baseOnBalls"), 0) or 0,
                    "HBP": safe_float(stat.get("hitBatsmen"), 0) or 0,
                    "SO": safe_float(stat.get("strikeOuts"), 0) or 0,
                    "R": safe_float(stat.get("runs"), 0) or 0,
                    "ER": safe_float(stat.get("earnedRuns"), 0) or 0,
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
    params = {
        "all": "true", "player_type": "batter", "batters_lookup[]": str(player_id),
        "game_date_gt": start_date, "game_date_lt": end_date, "type": "details",
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
    params = {
        "all": "true", "player_type": "pitcher", "pitchers_lookup[]": str(player_id),
        "game_date_gt": start_date, "game_date_lt": end_date, "type": "details",
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


@st.cache_data(ttl=300, show_spinner=False)
def bullpen_recent_workload(team_id: int, as_of: str, days: int = 3) -> Dict[str, Any]:
    end = pd.Timestamp(as_of).date()
    start = end - timedelta(days=days)
    data = safe_get_json(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "teamId": team_id, "startDate": start.isoformat(), "endDate": end.isoformat(), "gameTypes": "R"},
        timeout=20,
    ) or {}
    game_pks = [g.get("gamePk") for d in data.get("dates") or [] for g in d.get("games") or [] if g.get("gamePk")]
    pitches, innings, appearances = 0.0, 0.0, 0
    back_to_back = Counter()
    for game_pk in game_pks[-4:]:
        box = safe_get_json(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15) or {}
        for side in ["away", "home"]:
            team = ((box.get("teams") or {}).get(side) or {})
            if safe_int((team.get("team") or {}).get("id")) != safe_int(team_id):
                continue
            pitcher_ids = team.get("pitchers") or []
            for idx, pid in enumerate(pitcher_ids):
                player = (team.get("players") or {}).get(f"ID{pid}") or {}
                stat = ((player.get("stats") or {}).get("pitching") or {})
                # Treat all after the first listed pitcher as relief.
                if idx == 0:
                    continue
                appearances += 1
                pitches += safe_float(stat.get("pitchesThrown"), safe_float(stat.get("numberOfPitches"), 0)) or 0
                innings += innings_to_float(stat.get("inningsPitched"))
                back_to_back[str(pid)] += 1
    fatigue = clamp((pitches - 105) / 150, -0.10, 0.18)
    return {
        "pitches_3d": round(pitches, 1), "innings_3d": round(innings, 2), "appearances": appearances,
        "repeat_relievers": sum(1 for v in back_to_back.values() if v >= 2),
        "fatigue_factor": round(1 + fatigue, 3),
    }


def parse_weather_factor(weather: Dict[str, Any], venue: str) -> Dict[str, Any]:
    temp = safe_float(weather.get("temp"), 72) or 72
    wind_text = str(weather.get("wind") or "")
    condition = str(weather.get("condition") or "")
    mph_match = re.search(r"(\d+(?:\.\d+)?)\s*mph", wind_text, re.I)
    mph = safe_float(mph_match.group(1), 0) if mph_match else 0
    low = wind_text.lower()
    carry = 1.0 + clamp((temp - 72) * 0.0016, -0.025, 0.035)
    if "out" in low:
        carry += clamp(mph * 0.004, 0, 0.06)
    elif "in" in low:
        carry -= clamp(mph * 0.004, 0, 0.06)
    if any(x in condition.lower() for x in ["dome", "roof closed", "indoor"]):
        carry = 1.0
    return {"temp": temp, "wind": wind_text or "Unknown", "condition": condition or "Unknown", "carry_factor": round(clamp(carry, 0.93, 1.08), 3)}


@st.cache_data(ttl=21600, show_spinner=False)
def recent_lineup_slots(player_id: int, season: int, limit: int = 5) -> List[int]:
    logs = player_game_log(player_id, "hitting", season)
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


def projected_pa(slot: int, is_home: bool, team_runs: float) -> Tuple[float, Dict[int, float]]:
    base = {1: 4.72, 2: 4.62, 3: 4.53, 4: 4.43, 5: 4.31, 6: 4.18, 7: 4.06, 8: 3.94, 9: 3.82}.get(int(slot), 4.20)
    base += clamp((team_runs - LEAGUE["runs_per_game"]) * 0.12, -0.25, 0.30)
    if is_home:
        base -= 0.08  # possible skipped bottom ninth
    mean = clamp(base, 3.35, 5.15)
    values = np.array([3, 4, 5, 6], dtype=float)
    sigma = 0.70
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
    if df.empty:
        return {}
    match = pd.DataFrame()
    for c in ["player_id", "Player ID", "mlbam_id", "batter", "pitcher"]:
        if c in df.columns:
            match = df[pd.to_numeric(df[c], errors="coerce").eq(player_id)]
            if not match.empty:
                break
    if match.empty:
        for c in ["player_name", "Player", "Name", "full_name"]:
            if c in df.columns:
                match = df[df[c].astype(str).map(normalize_name).eq(normalize_name(player_name))]
                if not match.empty:
                    break
    return match.iloc[-1].to_dict() if not match.empty else {}


def historical_prior_rates(profile: Dict[str, Any], fallback_2025: Dict[str, float]) -> Dict[str, float]:
    aliases = {
        "bb_pa": ["bb_pa", "BB%", "bb_rate"], "hbp_pa": ["hbp_pa", "HBP%"],
        "k_pa": ["k_pa", "K%", "k_rate"], "single_pa": ["single_pa", "1B/PA", "single_rate"],
        "double_pa": ["double_pa", "2B/PA", "double_rate"], "triple_pa": ["triple_pa", "3B/PA", "triple_rate"],
        "hr_pa": ["hr_pa", "HR/PA", "hr_rate"],
    }
    out = dict(fallback_2025)
    for key, cols in aliases.items():
        for c in cols:
            value = safe_float(profile.get(c))
            if value is not None:
                if value > 1:
                    value /= 100.0
                out[key] = value
                break
    return out


def build_batter_profile(player_id: int, player_name: str, season: int, opening_day: str, today: str, pitcher_hand: str, line: float) -> Dict[str, Any]:
    current_stat = player_season_stat(player_id, "hitting", season)
    prior_stat = player_season_stat(player_id, "hitting", season - 1)
    split_stat = player_hand_split(player_id, "hitting", season, pitcher_hand)
    current = _batter_rates_from_stat(current_stat)
    prior_2025 = _batter_rates_from_stat(prior_stat)
    split = _batter_rates_from_stat(split_stat) if split_stat else {}
    offline = offline_profile_for(historical_batter_profiles(), player_id, player_name)
    prior = historical_prior_rates(offline, prior_2025)
    pa = current.get("pa", 0)
    split_pa = split.get("pa", 0) if split else 0
    # Feature-specific shrinkage: discipline stabilizes sooner than extra-base outcomes.
    strengths = {"bb_pa": 90, "hbp_pa": 170, "k_pa": 75, "single_pa": 180, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}
    rates: Dict[str, float] = {}
    for key, strength in strengths.items():
        base = beta_blend(current.get(key, LEAGUE[key]), pa, prior.get(key, LEAGUE[key]), strength)
        if split and split_pa >= 20:
            split_weight = clamp(split_pa / 350, 0.08, 0.28)
            base = base * (1 - split_weight) + split.get(key, base) * split_weight
        rates[key] = float(base)

    statcast = batter_statcast_profile(player_id, opening_day, today, pitcher_hand)
    # Physics-based expected-stat adjustment, capped to avoid double counting.
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

    logs = player_game_log(player_id, "hitting", season)
    if not logs.empty:
        start_ts = pd.Timestamp(opening_day)
        end_ts = pd.Timestamp(today) + pd.Timedelta(days=1)
        logs = logs[(logs["Date"] >= start_ts) & (logs["Date"] < end_ts)].copy()
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
    return {
        "rates": normalize_outcome_probs(rates), "current_pa": pa, "split_pa": split_pa,
        "current_stat": current_stat, "prior_stat": prior_stat, "split_stat": split_stat,
        "statcast": statcast, "logs": logs, "recent": recent, "offline_profile": offline,
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
    hand = str(person.get("pitchHand", {}).get("code") or "R").upper()[:1]
    current_stat = player_season_stat(player_id, "pitching", season)
    prior_stat = player_season_stat(player_id, "pitching", season - 1)
    current, prior = _pitcher_allowed_rates(current_stat), _pitcher_allowed_rates(prior_stat)
    offline = offline_profile_for(historical_pitcher_profiles(), player_id, player_name)
    prior = historical_prior_rates(offline, prior)
    bf = current.get("bf", 0)
    rates = {}
    for key, strength in {"bb_pa": 100, "hbp_pa": 200, "k_pa": 85, "single_pa": 190, "double_pa": 260, "triple_pa": 500, "hr_pa": 300}.items():
        rates[key] = beta_blend(current.get(key, LEAGUE[key]), bf, prior.get(key, LEAGUE[key]), strength)
    statcast = pitcher_statcast_profile(player_id, opening_day, today, batter_side)
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
    logs = player_game_log(player_id, "pitching", season)
    recent = logs.tail(5) if not logs.empty else pd.DataFrame()
    expected_bf = float(recent["BF"].mean()) if not recent.empty and recent["BF"].sum() > 0 else safe_float(current_stat.get("battersFaced"), 22) / max(safe_float(current_stat.get("gamesStarted"), 1), 1)
    expected_bf = clamp(expected_bf or 22.0, 10.0, 30.0)
    data_quality = clamp(35 + min(35, bf / 12) + (15 if statcast.get("available") else 0) + (10 if len(logs) >= 5 else 0), 20, 95)
    return {
        "rates": normalize_outcome_probs(rates), "available": True, "name": person.get("fullName") or player_name,
        "hand": hand, "expected_bf": round(expected_bf, 1), "statcast": statcast, "logs": logs,
        "current_stat": current_stat, "offline_profile": offline, "data_quality": round(data_quality, 1),
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


def blend_matchup_probs(batter: Dict[str, float], pitcher: Dict[str, float], park: Dict[str, float], weather_carry: float, bullpen_quality: float, arsenal_factor: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    result: Dict[str, float] = {}
    # Multiplicative batter/pitcher interaction around league rate.
    for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
        league = LEAGUE[key]
        b, p = batter.get(key, league), pitcher.get(key, league)
        raw = league * (b / league) ** 0.62 * (p / league) ** 0.38
        result[key] = raw
    result["single_pa"] *= park.get("1B", 1.0)
    result["double_pa"] *= park.get("2B", 1.0)
    result["triple_pa"] *= park.get("3B", 1.0)
    result["hr_pa"] *= park.get("HR", 1.0) * weather_carry * arsenal_factor
    # Bullpen factor is applied mildly because only a portion of PA face relievers.
    for key in ["single_pa", "double_pa", "triple_pa", "hr_pa", "bb_pa"]:
        result[key] *= clamp(1 + (bullpen_quality - 1) * 0.35, 0.94, 1.08)
    starter = normalize_outcome_probs(result)
    # Aggregate bullpen distribution regresses much more toward league.
    bullpen = {}
    for key in ["bb_pa", "hbp_pa", "k_pa", "single_pa", "double_pa", "triple_pa", "hr_pa"]:
        bullpen[key] = LEAGUE[key] * 0.55 + batter.get(key, LEAGUE[key]) * 0.30 + result.get(key, LEAGUE[key]) * 0.15
    for key in ["single_pa", "double_pa", "triple_pa", "hr_pa", "bb_pa"]:
        bullpen[key] *= bullpen_quality
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

# ============================================================
# CORRELATED FULL BASE/OUT-STATE SIMULATION
# ============================================================
OUTCOMES = ["BB", "HBP", "K", "1B", "2B", "3B", "HR", "OUT"]


def probs_array(probs: Dict[str, float]) -> np.ndarray:
    vals = np.array([probs.get("bb_pa", 0), probs.get("hbp_pa", 0), probs.get("k_pa", 0), probs.get("single_pa", 0), probs.get("double_pa", 0), probs.get("triple_pa", 0), probs.get("hr_pa", 0), probs.get("out_pa", 0)], dtype=float)
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


def advance_hit(bases: List[int], runner: int, hit_type: str, rng: np.random.Generator, speed_boost: float = 0.0) -> Tuple[List[int], List[int]]:
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
            if rng.random() < clamp(0.53 + speed_boost, 0.35, 0.78):
                scored.append(first)
            else:
                new_third = first
        return [0, runner, new_third], scored
    # Single
    if third:
        scored.append(third)
    new_third, new_second = 0, 0
    if second:
        if rng.random() < clamp(0.64 + speed_boost, 0.48, 0.86):
            scored.append(second)
        else:
            new_third = second
    if first:
        if new_third == 0 and rng.random() < clamp(0.29 + speed_boost * 0.5, 0.15, 0.48):
            new_third = first
        else:
            new_second = first
    return [runner, new_second, new_third], scored


def apply_out(bases: List[int], outs: int, rng: np.random.Generator) -> Tuple[List[int], int, List[int]]:
    first, second, third = bases
    scored: List[int] = []
    # Sacrifice fly / productive out.
    if outs < 2 and third and rng.random() < 0.19:
        scored.append(third)
        third = 0
        return [first, second, third], outs + 1, scored
    # Ground-ball double play.
    if outs < 2 and first and rng.random() < 0.105:
        first = 0
        return [first, second, third], outs + 2, scored
    return [first, second, third], outs + 1, scored


@dataclass
class SimResult:
    hits: np.ndarray
    runs: np.ndarray
    rbi: np.ndarray
    hrr: np.ndarray
    pa: np.ndarray


def simulate_player_games(
    player_name: str,
    line: float,
    slot: int,
    is_home: bool,
    target_starter_probs: Dict[str, float],
    target_bullpen_probs: Dict[str, float],
    generic_starter_probs: Dict[str, float],
    generic_bullpen_probs: Dict[str, float],
    expected_starter_bf: float,
    opponent_runs_mean: float,
    simulations: int,
    seed: int,
    speed_boost: float = 0.0,
    uncertainty_strength: float = 120.0,
) -> SimResult:
    rng = np.random.default_rng(seed)
    target_idx = int(clamp(slot, 1, 9)) - 1
    t_sp = probs_array(target_starter_probs)
    t_bp = probs_array(target_bullpen_probs)
    g_sp = probs_array(generic_starter_probs)
    g_bp = probs_array(generic_bullpen_probs)
    hits = np.zeros(simulations, dtype=np.int16)
    runs = np.zeros(simulations, dtype=np.int16)
    rbi = np.zeros(simulations, dtype=np.int16)
    hrr = np.zeros(simulations, dtype=np.int16)
    pas = np.zeros(simulations, dtype=np.int16)

    # Dirichlet outer uncertainty: wider when data is thin.
    alpha_scale = max(25.0, uncertainty_strength)
    for sim in range(simulations):
        t_sp_draw = rng.dirichlet(np.clip(t_sp * alpha_scale, 0.25, None))
        t_bp_draw = rng.dirichlet(np.clip(t_bp * alpha_scale, 0.25, None))
        starter_bf_limit = int(clamp(rng.normal(expected_starter_bf, 2.4), 10, 31))
        opponent_runs = int(rng.poisson(max(1.5, opponent_runs_mean)))
        team_runs = 0
        batter_index = 0
        starter_bf = 0
        target_h = target_r = target_rbi = target_pa = 0

        for inning in range(1, 10):
            if is_home and inning == 9 and team_runs > opponent_runs:
                break
            outs = 0
            bases = [0, 0, 0]  # 0 empty, 1 generic runner, 2 target player
            while outs < 3:
                is_target = batter_index == target_idx
                pitcher_is_starter = starter_bf < starter_bf_limit
                if is_target:
                    p = t_sp_draw if pitcher_is_starter else t_bp_draw
                    runner_code = 2
                    target_pa += 1
                else:
                    p = g_sp if pitcher_is_starter else g_bp
                    runner_code = 1
                outcome = OUTCOMES[int(rng.choice(len(OUTCOMES), p=p))]
                starter_bf += 1 if pitcher_is_starter else 0
                scored: List[int] = []
                if outcome in {"BB", "HBP"}:
                    bases, scored = force_walk(bases, runner_code)
                elif outcome in {"1B", "2B", "3B", "HR"}:
                    bases, scored = advance_hit(bases, runner_code, outcome, rng, speed_boost if is_target else 0.0)
                    if is_target:
                        target_h += 1
                        target_rbi += len([x for x in scored if x != 2]) + (1 if outcome == "HR" else 0)
                elif outcome == "K":
                    outs += 1
                else:
                    bases, outs, scored = apply_out(bases, outs, rng)
                    if is_target:
                        target_rbi += len([x for x in scored if x != 2])
                if scored:
                    team_runs += len(scored)
                    target_r += sum(1 for x in scored if x == 2)
                batter_index = (batter_index + 1) % 9
                if outs >= 3:
                    break
        hits[sim], runs[sim], rbi[sim], pas[sim] = target_h, target_r, target_rbi, target_pa
        hrr[sim] = target_h + target_r + target_rbi
    return SimResult(hits=hits, runs=runs, rbi=rbi, hrr=hrr, pa=pas)


def summarize_sim(result: SimResult, line: float) -> Dict[str, Any]:
    vals = result.hrr.astype(float)
    counts = Counter(result.hrr.tolist())
    mode = counts.most_common(1)[0][0] if counts else None
    over = float(np.mean(vals > line))
    under = float(np.mean(vals < line))
    push = float(np.mean(vals == line))
    return {
        "projection": round(float(vals.mean()), 2), "median": round(float(np.median(vals)), 2), "mode": mode,
        "over_prob": over, "under_prob": under, "push_prob": push,
        "hits": round(float(result.hits.mean()), 2), "runs": round(float(result.runs.mean()), 2),
        "rbi": round(float(result.rbi.mean()), 2), "pa": round(float(result.pa.mean()), 2),
        "std": round(float(vals.std()), 2), "p10": float(np.quantile(vals, 0.10)), "p90": float(np.quantile(vals, 0.90)),
    }


def empirical_probability(logs: pd.DataFrame, line: float, side: str, scale: float = 1.0) -> Optional[float]:
    if logs.empty or "HRR" not in logs or len(logs) < 5:
        return None
    values = logs["HRR"].astype(float).to_numpy()
    # Recency weights are controlled and cannot overpower the season sample.
    weights = np.linspace(0.70, 1.30, len(values))
    adjusted = values * clamp(scale, 0.85, 1.15)
    wins = adjusted > line if side == "OVER" else adjusted < line
    # Beta prior keeps tiny samples from looking certain.
    weighted_wins = float(np.sum(weights * wins))
    total = float(np.sum(weights))
    return (weighted_wins + 3.0) / (total + 6.0)

# ============================================================
# BOARD ENGINE
# ============================================================
def build_team_environment(team_id: int, opp_team_id: int, season: int, park: Dict[str, float], weather: Dict[str, Any], bullpen: Dict[str, Any]) -> Dict[str, Any]:
    team_hit = team_season_stat(team_id, "hitting", season)
    opp_pitch = team_season_stat(opp_team_id, "pitching", season)
    games = safe_float(team_hit.get("gamesPlayed"), 0) or 0
    runs_pg = (safe_float(team_hit.get("runs"), 0) or 0) / games if games else LEAGUE["runs_per_game"]
    ops = safe_float(team_hit.get("ops"), LEAGUE["ops"]) or LEAGUE["ops"]
    opp_games = safe_float(opp_pitch.get("gamesPlayed"), 0) or 0
    opp_ra = (safe_float(opp_pitch.get("runs"), 0) or 0) / opp_games if opp_games else LEAGUE["runs_per_game"]
    implied = 0.48 * runs_pg + 0.30 * opp_ra + 0.22 * LEAGUE["runs_per_game"]
    implied *= park.get("R", 1.0)
    implied *= clamp(weather.get("carry_factor", 1.0) ** 0.35, 0.97, 1.03)
    implied *= clamp(bullpen.get("fatigue_factor", 1.0), 0.95, 1.07)
    implied += clamp((ops - LEAGUE["ops"]) * 2.4, -0.35, 0.35)
    return {"team_stat": team_hit, "opp_pitch_stat": opp_pitch, "implied_runs": round(clamp(implied, 2.6, 6.9), 2), "runs_pg": round(runs_pg, 2), "ops": round(ops, 3), "opp_ra_pg": round(opp_ra, 2)}


def data_quality_score(row: Dict[str, Any]) -> Tuple[float, str]:
    score = 20.0
    score += 12 if row.get("Player ID") else 0
    score += 12 if row.get("GamePk") else 0
    score += 14 if row.get("Pitcher ID") else 3
    score += 12 if row.get("Batter PA", 0) >= 100 else 7 if row.get("Batter PA", 0) >= 40 else 2
    score += 12 if row.get("Batter Statcast") else 0
    score += 10 if row.get("Pitcher Statcast") else 0
    score += 8 if row.get("Lineup Status") == "CONFIRMED" else 3
    score += 5 if row.get("Bullpen Available") else 0
    score += 5 if row.get("Weather Available") else 0
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
        slots = recent_lineup_slots(player_id, season, 5)
        slot = int(round(float(np.median(slots)))) if slots else 5
        lineup_status = "PROJECTED"
    pitcher_id = safe_int(game.get("opp_pitcher_id"))
    pitcher_person = get_person(pitcher_id) if pitcher_id else {}
    pitcher_hand = str((pitcher_person.get("pitchHand") or {}).get("code") or "R").upper()[:1]
    batter_side = str((person.get("batSide") or {}).get("code") or "R").upper()[:1]

    batter = build_batter_profile(player_id, player_name, season, opening_day, game_date, pitcher_hand, line)
    pitcher = build_pitcher_profile(pitcher_id, game.get("opp_pitcher") or "TBD", season, opening_day, game_date, batter_side)
    park = PARK_FACTORS.get(str(game.get("venue") or ""), {"1B": 1.0, "2B": 1.0, "3B": 1.0, "HR": 1.0, "R": 1.0})
    weather_raw = game_ctx.get("weather") or {}
    weather = parse_weather_factor(weather_raw, str(game.get("venue") or ""))
    bullpen = bullpen_recent_workload(safe_int(game.get("opponent_team_id")), game_date) if game.get("opponent_team_id") else {}
    bullpen_quality = clamp(safe_float(bullpen.get("fatigue_factor"), 1.0) or 1.0, 0.93, 1.10)
    env = build_team_environment(team_id, safe_int(game.get("opponent_team_id")), season, park, weather, bullpen) if team_id and game.get("opponent_team_id") else {"team_stat": {}, "implied_runs": LEAGUE["runs_per_game"], "runs_pg": LEAGUE["runs_per_game"], "ops": LEAGUE["ops"], "opp_ra_pg": LEAGUE["runs_per_game"]}
    pa_mean, pa_dist = projected_pa(slot, bool(game.get("is_home")), env.get("implied_runs", LEAGUE["runs_per_game"]))
    arsenal_factor, arsenal_note = arsenal_match_factor(batter.get("statcast") or {}, pitcher.get("statcast") or {})
    starter_probs, bullpen_probs = blend_matchup_probs(batter["rates"], pitcher["rates"], park, weather.get("carry_factor", 1.0), bullpen_quality, arsenal_factor)
    generic = generic_team_probs(env.get("team_stat") or {})
    generic_starter, generic_bullpen = blend_matchup_probs(generic, pitcher["rates"], park, weather.get("carry_factor", 1.0), bullpen_quality, 1.0)
    current_pa = safe_float(batter.get("current_pa"), 0) or 0
    pitcher_q = safe_float(pitcher.get("data_quality"), 25) or 25
    uncertainty = clamp(55 + current_pa * 0.25 + pitcher_q * 0.55, 45, 220)
    speed = safe_float((batter.get("offline_profile") or {}).get("sprint_speed"), 27.0) or 27.0
    speed_boost = clamp((speed - 27.0) * 0.025, -0.06, 0.08)
    seed = stable_seed(game_date, player_id, line, MODEL_VERSION, slot, pitcher_id)
    sim = simulate_player_games(
        player_name, line, slot, bool(game.get("is_home")), starter_probs, bullpen_probs,
        generic_starter, generic_bullpen, pitcher.get("expected_bf", 22.0),
        opponent_runs_mean=LEAGUE["runs_per_game"], simulations=screen_sims, seed=seed,
        speed_boost=speed_boost, uncertainty_strength=uncertainty,
    )
    summary = summarize_sim(sim, line)
    side = "OVER" if summary["over_prob"] >= summary["under_prob"] else "UNDER"
    sim_pick_prob = summary["over_prob"] if side == "OVER" else summary["under_prob"]
    matchup_scale = clamp(summary["projection"] / max(batter["recent"].get("season_hrr_avg") or summary["projection"], 0.6), 0.85, 1.15)
    batter_logs = batter.get("logs") if isinstance(batter.get("logs"), pd.DataFrame) else pd.DataFrame()
    empirical = empirical_probability(batter_logs, line, side, matchup_scale)
    if empirical is None:
        empirical = 0.5
    disagreement = abs(sim_pick_prob - empirical)
    # Shrink probabilities based on uncertainty and data quality; calibration improves with grades.
    preliminary = 0.74 * sim_pick_prob + 0.26 * empirical
    edge = summary["projection"] - line
    row_base = {
        "Player ID": player_id, "GamePk": game.get("game_pk"), "Pitcher ID": pitcher_id,
        "Batter PA": current_pa, "Batter Statcast": bool((batter.get("statcast") or {}).get("available")),
        "Pitcher Statcast": bool((pitcher.get("statcast") or {}).get("available")), "Lineup Status": lineup_status,
        "Bullpen Available": bool(bullpen), "Weather Available": bool(weather_raw),
    }
    dq, dq_label = data_quality_score(row_base)
    calibrated = 0.5 + (preliminary - 0.5) * clamp(0.58 + dq / 180, 0.64, 1.08)
    calibrated = clamp(calibrated, 0.35, 0.76)
    role_risk = 0.0
    role_notes = []
    if lineup_status != "CONFIRMED":
        role_risk += 17; role_notes.append("lineup projected")
    if not pitcher_id:
        role_risk += 25; role_notes.append("probable pitcher TBD")
    if slot >= 7:
        role_risk += 10; role_notes.append("lower batting order")
    if pa_mean < 4.0:
        role_risk += 10; role_notes.append("limited PA projection")
    if current_pa < 45:
        role_risk += 15; role_notes.append("thin current-season sample")
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
    factors = []
    factors.append(f"Slot {slot} ({lineup_status.lower()})")
    factors.append(f"Team runs {env.get('implied_runs')}")
    factors.append(f"Pitcher vulnerability {vuln.get('Overall')}")
    factors.append(arsenal_note)
    if bullpen.get("fatigue_factor", 1.0) > 1.03:
        factors.append("Bullpen workload favorable")
    elif bullpen.get("fatigue_factor", 1.0) < 0.98:
        factors.append("Fresh bullpen")
    factors.append(f"Park HR x{park.get('HR',1.0):.2f}")
    factors.append(f"Weather carry x{weather.get('carry_factor',1.0):.3f}")

    return {
        "Date": game_date, "Player": player_name, "Player ID": player_id,
        "Team": game.get("team") or team.get("name") or "—", "Opponent": game.get("opponent") or "—",
        "Matchup": f"{game.get('away','—')} @ {game.get('home','—')}" if game else "No MLB game match",
        "GamePk": game.get("game_pk"), "Start Time": game.get("start_time"), "Venue": game.get("venue") or "—",
        "Source": ud.get("Source"), "Market": "Hits + Runs + RBIs", "Line": line,
        "Projection": summary["projection"], "Median": summary["median"], "Mode": summary["mode"],
        "Expected H": summary["hits"], "Expected R": summary["runs"], "Expected RBI": summary["rbi"],
        "Projected PA": summary["pa"], "PA Model Mean": pa_mean, "PA Distribution": pa_dist,
        "P10": summary["p10"], "P90": summary["p90"], "Volatility": summary["std"],
        "Over Probability %": round(summary["over_prob"] * 100, 1), "Under Probability %": round(summary["under_prob"] * 100, 1), "Push Probability %": round(summary["push_prob"] * 100, 1),
        "Pick": side, "Pick Probability %": round(calibrated * 100, 1), "Fair Odds": fair_american(calibrated),
        "Edge": round(edge, 2), "Grade": grade, "Grade Note": grade_note,
        "Data Quality": dq, "Data Quality Label": dq_label, "Model Agreement Gap": round(disagreement * 100, 1),
        "Simulation Probability %": round(sim_pick_prob * 100, 1), "Independent Baseline %": round(empirical * 100, 1),
        "Lineup Slot": slot, "Lineup Status": lineup_status, "Role Risk": round(role_risk, 1), "Role Risk Note": "; ".join(role_notes) or "Low",
        "Pitcher": pitcher.get("name") or "TBD", "Pitcher Hand": pitcher.get("hand") or pitcher_hand,
        "Starter Expected BF": pitcher.get("expected_bf"), "Starter Exposure %": round(min(1.0, (pitcher.get("expected_bf",22) / 9) / max(pa_mean,1)) * 100, 1),
        "Pitcher Vulnerability": vuln.get("Overall"), "Contact Allowed Score": vuln.get("Contact Allowed"), "Damage Allowed Score": vuln.get("Damage Allowed"), "Traffic Allowed Score": vuln.get("Traffic Allowed"),
        "Bullpen Pitches 3D": bullpen.get("pitches_3d"), "Bullpen Fatigue Factor": bullpen.get("fatigue_factor"),
        "Team Implied Runs": env.get("implied_runs"), "Team OPS": env.get("ops"), "Team Runs/G": env.get("runs_pg"),
        "Temperature": weather.get("temp"), "Wind": weather.get("wind"), "Weather": weather.get("condition"), "Weather Carry": weather.get("carry_factor"),
        "Park 1B": park.get("1B", 1.0), "Park 2B": park.get("2B", 1.0), "Park 3B": park.get("3B", 1.0), "Park HR": park.get("HR", 1.0),
        "Batter PA": current_pa, "Batter Statcast Rows": (batter.get("statcast") or {}).get("rows"), "Pitcher Statcast Rows": (pitcher.get("statcast") or {}).get("rows"),
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
        "Main Factors": " • ".join(factors), "Simulations": screen_sims, "Model Version": MODEL_VERSION,
        "Timestamp": now_iso(),
    }


def apply_learning_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    learning = read_json(LEARNING_FILE, {})
    global_cal = learning.get("global") or {}
    n = safe_float(global_cal.get("n"), 0) or 0
    bias = safe_float(global_cal.get("projection_bias"), 0) or 0
    if n < 25 or abs(bias) < 0.05:
        df["Learning Adjustment"] = 0.0
        return df
    adj = clamp(-bias * min(n / 150, 1.0), -0.22, 0.22)
    df = df.copy()
    df["Pre-Learning Projection"] = df["Projection"]
    df["Projection"] = (df["Projection"] + adj).clip(lower=0).round(2)
    df["Edge"] = (df["Projection"] - df["Line"]).round(2)
    df["Learning Adjustment"] = round(adj, 3)
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
        "Over Probability %", "Under Probability %", "Push Probability %", "Pick", "Pick Probability %", "Fair Odds",
        "Edge", "Grade", "Grade Note", "Data Quality", "Data Quality Label", "Model Agreement Gap", "Lineup Slot",
        "Lineup Status", "Role Risk", "Pitcher", "Pitcher ID", "Pitcher Hand", "Pitcher Vulnerability",
        "Team Implied Runs", "Temperature", "Wind", "Weather", "Park HR", "L5 HRR Avg", "L10 HRR Avg",
        "Season HRR Avg", "Over Odds", "Under Odds", "Market No-Vig %", "Market Edge %", "Market Agreement",
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


def actual_hrr_for_date(player_id: int, game_date: str) -> Optional[Dict[str, Any]]:
    season = pd.Timestamp(game_date).year
    logs = player_game_log(player_id, "hitting", season)
    if logs.empty:
        return None
    target = pd.Timestamp(game_date).date()
    sub = logs[logs["Date"].dt.date.eq(target)]
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
        actual = actual_hrr_for_date(safe_int(pick.get("Player ID")), str(pick.get("Date")))
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


def update_learning(results: Sequence[Dict[str, Any]]) -> None:
    finished = [r for r in results if r.get("Result") in {"WIN", "LOSS", "PUSH"}]
    non_push = [r for r in finished if r.get("Result") in {"WIN", "LOSS"}]
    errors = [safe_float(r.get("Projection Error"), 0) or 0 for r in finished]
    global_row = {
        "n": len(finished), "non_push_n": len(non_push),
        "wins": sum(r.get("Result") == "WIN" for r in non_push), "losses": sum(r.get("Result") == "LOSS" for r in non_push),
        "win_rate": round(sum(r.get("Result") == "WIN" for r in non_push) / len(non_push), 4) if non_push else None,
        "projection_bias": round(float(np.mean(errors)), 4) if errors else 0.0,
        "mae": round(float(np.mean(np.abs(errors))), 4) if errors else None,
    }
    buckets: Dict[str, Any] = {}
    for grade in ["🔥 ATTACK", "✅ OFFICIAL", "⚠️ PLAYABLE", "👀 TRACK ONLY", "🚫 PASS"]:
        subset = [r for r in non_push if r.get("Grade") == grade]
        if subset:
            buckets[grade] = {"n": len(subset), "win_rate": round(sum(r.get("Result") == "WIN" for r in subset) / len(subset), 4)}
    write_json(LEARNING_FILE, {"updated": now_iso(), "global": global_row, "grades": buckets}, protect=False)


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
        "Expected H", "Expected R", "Expected RBI", "Projected PA", "Lineup Slot", "Lineup Status", "Team Implied Runs",
        "Pitcher Vulnerability", "Data Quality", "Volatility", "L5 HRR Avg", "L10 HRR Avg", "Season HRR Avg",
        "Market No-Vig %", "Market Edge %", "Market Agreement", "Role Risk", "Grade Note",
    ]
    return [c for c in desired if c in df.columns]

# ============================================================
# APP STATE / SIDEBAR
# ============================================================
st.markdown(
    f"""
<div class="hero"><h1>⚾ ONE WAY PICKZ — HITS + RUNS + RBI</h1><p>{APP_VERSION} · Underdog players only · Bayesian profiles + pitcher/bullpen context + correlated game-state Monte Carlo</p></div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Daily Slate")
    selected_date = st.date_input("Game date", value=la_now().date()).isoformat()
    sims = st.select_slider("Screening simulations per player", options=[2048, 4096, 5000, 8192, 10000], value=5000)
    use_manual_when_empty = st.checkbox("Use saved manual lines when Underdog is empty", value=True)
    refresh = st.button("🔄 Pull lines + build projections", use_container_width=True, type="primary")
    st.caption("The engine pulls data only for active Underdog H+R+RBI players and their opposing probable pitchers.")
    st.divider()
    st.subheader("Model controls")
    show_passes = st.checkbox("Show PASS rows", value=False)
    include_playable_save = st.checkbox("Include Playable when saving", value=False)
    st.caption("Official thresholds are intentionally strict. L5/L10 are supporting context, not the core projection.")

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
    st.markdown('<div class="section">Automatic Grading and Learning</div>', unsafe_allow_html=True)
    if st.button("🏁 Grade finished H+R+RBI picks", type="primary", use_container_width=True):
        status = grade_saved_picks(force=False)
        st.success(f"Graded {status['graded']} · Pushes {status['pushes']} · Pending {status['pending']} · Already graded {status['skipped']}")
    results = read_json(RESULT_LOG, [])
    if results:
        rdf = pd.DataFrame(results)
        finished = rdf[rdf["Result"].isin(["WIN", "LOSS"])].copy() if "Result" in rdf else pd.DataFrame()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Graded", len(rdf))
        c2.metric("Record", f"{int((finished['Result']=='WIN').sum())}-{int((finished['Result']=='LOSS').sum())}" if not finished.empty else "—")
        c3.metric("Win Rate", f"{(finished['Result']=='WIN').mean()*100:.1f}%" if not finished.empty else "—")
        c4.metric("MAE", f"{rdf['Projection Error'].abs().mean():.2f}" if "Projection Error" in rdf else "—")
        st.dataframe(rdf.sort_values("Graded At", ascending=False) if "Graded At" in rdf else rdf, use_container_width=True, hide_index=True)
        st.download_button("Download graded history CSV", data=rdf.to_csv(index=False).encode(), file_name="hrr_graded_history.csv", mime="text/csv", use_container_width=True)
        learning = read_json(LEARNING_FILE, {})
        st.markdown("#### Learning summary")
        st.json(learning)
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
    st.caption("The app is fully operational with official MLB current/prior-season data. Add the planned 2021-2025 offline Statcast profile files to improve priors without making Railway rebuild millions of pitches.")

    if board is not None and not board.empty:
        st.markdown("#### Active-player data audit")
        audit_cols = [c for c in ["Player", "Batter PA", "Batter Statcast Rows", "Pitcher", "Pitcher Statcast Rows", "Lineup Status", "Data Quality", "Role Risk Note"] if c in board]
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
