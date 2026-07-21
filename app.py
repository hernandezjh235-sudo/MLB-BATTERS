# -*- coding: utf-8 -*-
"""
ONE WAY PICKZ — MLB BATTER PROJECTION ENGINE v5

Batter-only Streamlit application.
Markets shown:
  • Hits + Runs + RBIs (H+R+RBI)
  • Home Runs

The opposing pitcher and bullpen are used only as matchup inputs. There are no
pitcher-prop, pitching-outs, strikeout-prop, or moneyline tabs in this app.

Runtime data sources:
  • Underdog Fantasy public over/under feed (posted batter lines only)
  • MLB Stats API (schedule, lineups, player/team statistics, game results)
  • Baseball Savant CSV leaderboards/search (best effort; neutral fallbacks)

Model architecture:
  • Bayesian true-talent baseline with season-progressive and learned weights
  • Plate-appearance predictor with 3/4/5/6 PA probabilities
  • Hierarchical PA-outcome model; optional LightGBM artifact ensemble
  • Batter contact/discipline and quality-of-contact feature engine
  • Six-part opposing-pitcher vulnerability engine
  • Starter/bullpen exposure model
  • PA-by-PA Monte Carlo base-advancement simulation
  • Persistent grading, calibration, and backtested weight learning
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================================================================
# APP / STORAGE CONFIG
# =============================================================================

APP_VERSION = "ONE WAY PICKZ — BATTER v5.0 TRUE TALENT + PA SIM"
APP_BUILD = "2026-07-15"
MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"

UNDERDOG_URLS = [
    "https://api.underdogfantasy.com/beta/v7/over_under_lines",
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
    "https://api.underdogfantasy.com/beta/v2/over_under_lines",
    "https://api.underdogfantasy.com/v1/over_under_lines",
]

STORAGE_DIR = Path(os.getenv("BATTER_STORAGE_DIR", "batter_engine"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_FILE = STORAGE_DIR / "batter_pick_snapshots.json"
GRADE_FILE = STORAGE_DIR / "batter_graded_history.json"
WEIGHT_FILE = STORAGE_DIR / "batter_model_weights.json"
LINE_HISTORY_FILE = STORAGE_DIR / "batter_line_history.json"
CACHE_DIR = STORAGE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SIMULATIONS = int(os.getenv("BATTER_SIMULATIONS", "7000"))
MAX_BOARD_WORKERS = int(os.getenv("BATTER_MAX_WORKERS", "8"))
HTTP_TIMEOUT = int(os.getenv("BATTER_HTTP_TIMEOUT", "18"))

st.set_page_config(
    page_title="OneWayPickz Batter Projections",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# VISUAL STYLE — WHITE / RED / BLACK
# =============================================================================

st.markdown(
    """
<style>
:root {
  --ow-red:#e10600;
  --ow-red2:#ff3b30;
  --ow-black:#080808;
  --ow-panel:#111111;
  --ow-panel2:#171717;
  --ow-white:#ffffff;
  --ow-muted:#a7a7a7;
  --ow-green:#29d17d;
  --ow-yellow:#ffcf33;
}
.stApp { background: radial-gradient(circle at top, #211010 0%, #0a0a0a 38%, #050505 100%); color:white; }
.block-container { padding-top:1.1rem; max-width:1550px; }
[data-testid="stSidebar"] { background:#090909; border-right:1px solid rgba(225,6,0,.45); }
.ow-hero {
  border:1px solid rgba(225,6,0,.55); border-radius:18px; padding:20px 22px;
  background:linear-gradient(135deg,rgba(225,6,0,.22),rgba(12,12,12,.96) 48%,rgba(255,255,255,.04));
  box-shadow:0 0 28px rgba(225,6,0,.14); margin-bottom:14px;
}
.ow-title { font-size:34px; font-weight:950; letter-spacing:.02em; }
.ow-sub { color:#c9c9c9; font-size:14px; margin-top:4px; }
.ow-card {
  background:linear-gradient(145deg,#161616,#0d0d0d); border:1px solid rgba(255,255,255,.11);
  border-left:4px solid var(--ow-red); border-radius:16px; padding:16px; margin:10px 0;
  box-shadow:0 8px 25px rgba(0,0,0,.30);
}
.ow-player { font-size:24px; font-weight:900; }
.ow-muted { color:var(--ow-muted); font-size:12px; }
.ow-over { color:var(--ow-green); font-weight:900; }
.ow-under { color:#ff5b57; font-weight:900; }
.ow-pass { color:var(--ow-yellow); font-weight:900; }
.ow-badge { display:inline-block; border-radius:999px; padding:4px 9px; margin-right:5px; font-size:11px; font-weight:850; border:1px solid rgba(255,255,255,.14); }
.ow-good { background:rgba(41,209,125,.14); color:#69eca7; }
.ow-warn { background:rgba(255,207,51,.13); color:#ffdc67; }
.ow-bad { background:rgba(255,65,60,.13); color:#ff7772; }
div[data-testid="stMetric"] { background:#111; border:1px solid rgba(255,255,255,.09); padding:12px; border-radius:12px; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#111; border-radius:10px 10px 0 0; border:1px solid rgba(255,255,255,.08); }
.stTabs [aria-selected="true"] { background:rgba(225,6,0,.24) !important; border-color:rgba(225,6,0,.55) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        out = float(str(value).replace("%", "").replace(",", "").strip())
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    v = safe_float(value, None)
    return int(round(v)) if v is not None else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct(value: Optional[float], digits: int = 1) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100.0, digits)


def norm_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def today_local() -> date:
    return datetime.now().astimezone().date()


def season_year(day: Optional[date] = None) -> int:
    d = day or today_local()
    return d.year


def iso_day(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_json_rows(path: Path, rows: Sequence[Mapping[str, Any]], unique_keys: Sequence[str]) -> int:
    current = read_json(path, [])
    if not isinstance(current, list):
        current = []
    index: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in current:
        if isinstance(row, dict):
            key = tuple(row.get(k) for k in unique_keys)
            index[key] = row
    added = 0
    for row in rows:
        item = dict(row)
        key = tuple(item.get(k) for k in unique_keys)
        if key not in index:
            added += 1
        index[key] = item
    write_json(path, list(index.values()))
    return added


def stable_seed(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16) % (2**32 - 1)


def format_value(value: Any, digits: int = 2, missing: str = "—") -> str:
    v = safe_float(value, None)
    return missing if v is None else f"{v:.{digits}f}"


def implied_grade(probability: float, edge: float, data_quality: float, market: str) -> str:
    edge_need = 0.42 if market == "HRR" else 0.08
    if probability >= 0.64 and abs(edge) >= edge_need * 1.5 and data_quality >= 82:
        return "A+"
    if probability >= 0.60 and abs(edge) >= edge_need and data_quality >= 75:
        return "A"
    if probability >= 0.575 and abs(edge) >= edge_need * 0.75 and data_quality >= 68:
        return "B+"
    if probability >= 0.555 and data_quality >= 62:
        return "B"
    return "PASS"


# =============================================================================
# HTTP CLIENT
# =============================================================================


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def json(self, url: str, params: Optional[Mapping[str, Any]] = None, timeout: int = HTTP_TIMEOUT) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(0.45 * (attempt + 1))
        return None

    def text(self, url: str, params: Optional[Mapping[str, Any]] = None, timeout: int = HTTP_TIMEOUT) -> Optional[str]:
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.text
            except Exception:
                time.sleep(0.45 * (attempt + 1))
        return None


HTTP = HttpClient()


# =============================================================================
# UNDERDOG — BATTER LINES ONLY
# =============================================================================


HRR_TERMS = (
    "hits + runs + rbis",
    "hits+runs+rbis",
    "hits+runs+rbis o/u",
    "hits runs rbis",
    "hits runs and rbis",
    "hits, runs, and rbis",
    "hits runs rbi",
    "hits + runs + rbi",
    "hits+runs+rbi",
    "hits+runs+rbi o/u",
    "batter hits + runs + rbis",
    "batter hits+runs+rbis",
    "h+r+rbi",
    "h + r + rbi",
    "h + r + rbi o/u",
    "h+r+r",
    "h + r + r",
)
HR_TERMS = (
    "home runs",
    "home run",
    "home run o/u",
    "home runs o/u",
    "batter home runs",
    "batter home run",
    "homeruns",
)
NON_BATTER_TERMS = (
    "pitcher",
    "strikeouts allowed",
    "hits allowed",
    "earned runs",
    "pitching outs",
    "outs recorded",
    "moneyline",
    "run line",
    "team total",
    "soccer",
    "basketball",
    "football",
    "hockey",
    "golf",
    "tennis",
    "esports",
    "cs2",
    "league of legends",
)
OTHER_BATTER_MARKETS = (
    "total bases",
    "batter hits",
    "runs o/u",
    "rbis o/u",
    "singles",
    "doubles",
    "stolen bases",
    "batter strikeouts",
    "fantasy points",
    "fantasy score",
)


@dataclass
class UnderdogLine:
    player: str
    market: str
    line: float
    source: str = "Underdog"
    status: str = "active"
    evidence: str = ""
    event_title: str = ""
    line_id: str = ""


def _ud_attrs(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    attrs = obj.get("attributes")
    out = dict(attrs) if isinstance(attrs, dict) else {}
    for key, value in obj.items():
        if key not in {"attributes", "relationships", "included", "data"} and key not in out:
            out[key] = value
    return out


def _ud_collect(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("type", "attributes", "relationships", "stat_value", "line", "title")):
                found.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return found


def _ud_object_maps(objects: Sequence[Mapping[str, Any]]) -> Tuple[Dict[Tuple[str, str], Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    by_id: Dict[str, Mapping[str, Any]] = {}
    for obj in objects:
        oid = str(obj.get("id", ""))
        typ = str(obj.get("type", ""))
        if oid:
            by_id[oid] = obj
            by_key[(typ, oid)] = obj
    return by_key, by_id


def _ud_related(
    obj: Optional[Mapping[str, Any]],
    names: Sequence[str],
    by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if not isinstance(obj, Mapping):
        return None
    rels = obj.get("relationships")
    if not isinstance(rels, Mapping):
        return None
    normalized = {str(k).lower().replace("-", "_"): v for k, v in rels.items()}
    for name in names:
        rel = normalized.get(name.lower().replace("-", "_"))
        if not isinstance(rel, Mapping):
            continue
        data = rel.get("data")
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, Mapping):
            continue
        oid = str(data.get("id", ""))
        typ = str(data.get("type", ""))
        if (typ, oid) in by_key:
            return by_key[(typ, oid)]
        if oid in by_id:
            return by_id[oid]
    return None


def _ud_blob(*objects: Optional[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    keys = (
        "title",
        "display_title",
        "name",
        "display_name",
        "full_name",
        "first_name",
        "last_name",
        "player_name",
        "stat",
        "stat_type",
        "appearance_stat",
        "appearance_stat_type",
        "market",
        "market_name",
        "market_display_name",
        "stat_name",
        "display_stat",
        "display_stat_name",
        "stat_type",
        "stat_type_name",
        "over_under_title",
        "label",
        "description",
        "sport",
        "sport_name",
        "league",
        "league_name",
    )
    for obj in objects:
        attrs = _ud_attrs(obj)
        for key in keys:
            value = attrs.get(key)
            if value not in (None, "") and not isinstance(value, (dict, list)):
                parts.append(str(value))
    return " | ".join(parts)


def _ud_market(blob: str) -> Optional[str]:
    low = blob.lower()
    if any(term in low for term in NON_BATTER_TERMS):
        return None
    compact = re.sub(r"[^a-z0-9]+", "", low)
    if any(term in low for term in HRR_TERMS) or any(
        token in compact for token in ("hitsrunsrbis", "hitsrunsrbi", "hrbi", "hrrbi", "hrr")
    ):
        return "HRR"
    if any(term in low for term in HR_TERMS) or any(
        token in compact for token in ("homeruns", "homerun", "batterhomeruns", "batterhomerun")
    ):
        return "HR"
    return None


def _ud_line_value(*objects: Optional[Mapping[str, Any]]) -> Optional[float]:
    strict_keys = (
        "stat_value",
        "statValue",
        "line",
        "over_under_line",
        "overUnderLine",
        "over_under_value",
        "overUnderValue",
        "target_value",
        "targetValue",
        "line_score",
        "lineScore",
        "display_stat_value",
        "displayStatValue",
        "value",
        "points",
    )
    for obj in objects:
        attrs = _ud_attrs(obj)
        for key in strict_keys:
            value = safe_float(attrs.get(key), None)
            if value is not None and 0.5 <= value <= 8.5 and abs(value * 2 - round(value * 2)) < 1e-7:
                return float(value)
    return None


def _clean_player_name(value: str) -> str:
    text = str(value or "").strip()
    patterns = [
        r"\s+Hits\s*\+\s*Runs\s*\+\s*RBIs?.*$",
        r"\s+H\s*\+\s*R\s*\+\s*R(?:BI)?.*$",
        r"\s+(?:Batter\s+)?Home\s+Runs?.*$",
        r"\s+Over/Under.*$",
        r"\s+O/U.*$",
        r"\s+(?:Higher|Lower|Over|Under).*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.I).strip()
    return text.strip(" -|•:")


def _ud_player_name(*objects: Optional[Mapping[str, Any]]) -> str:
    values: List[str] = []
    for obj in objects:
        attrs = _ud_attrs(obj)
        first = str(attrs.get("first_name", "")).strip()
        last = str(attrs.get("last_name", "")).strip()
        if first and last:
            values.append(f"{first} {last}")
        for key in ("player_name", "full_name", "display_name", "name", "appearance_name", "title"):
            value = attrs.get(key)
            if isinstance(value, str) and 2 <= len(value) <= 120:
                values.append(value)
    cleaned = [_clean_player_name(v) for v in values]
    cleaned = [v for v in cleaned if len(norm_name(v).split()) >= 2 and len(v) <= 65]
    if not cleaned:
        return ""
    return sorted(cleaned, key=lambda x: (len(norm_name(x).split()), -len(x)), reverse=True)[0]


def _ud_active(*objects: Optional[Mapping[str, Any]]) -> bool:
    blob = _ud_blob(*objects).lower()
    return not any(term in blob for term in ("suspended", "inactive", "closed", "disabled", "hidden true", "removed"))


@st.cache_data(ttl=180, show_spinner=False)
def fetch_underdog_batter_lines() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch and parse active MLB batter HRR/HR lines only.

    The parser follows JSON:API relationships when present and also includes a
    strict flattened-title fallback. It never manufactures lines from unrelated
    numbers inside the response.
    """
    headers = {
        "Origin": "https://underdogfantasy.com",
        "Referer": "https://underdogfantasy.com/",
    }
    HTTP.session.headers.update(headers)
    parsed: List[UnderdogLine] = []
    debug: Dict[str, Any] = {
        "endpoint": None,
        "status": "NO_RESPONSE",
        "objects": 0,
        "candidate_lines": 0,
        "parsed_rows": 0,
        "sample_market_text": [],
    }

    for url in UNDERDOG_URLS:
        payload = HTTP.json(url, timeout=HTTP_TIMEOUT)
        if not payload:
            continue
        debug["endpoint"] = url
        debug["status"] = "RESPONSE_RECEIVED"
        objects = _ud_collect(payload)
        debug["objects"] = len(objects)
        by_key, by_id = _ud_object_maps(objects)

        candidates: List[Mapping[str, Any]] = []
        for obj in objects:
            attrs = _ud_attrs(obj)
            typ = str(obj.get("type", "")).lower()
            if "over_under_line" in typ or any(
                attrs.get(key) not in (None, "")
                for key in ("stat_value", "line", "over_under_line", "target_value", "line_score")
            ):
                candidates.append(obj)
        debug["candidate_lines"] = len(candidates)

        for line_obj in candidates:
            ou_obj = _ud_related(line_obj, ("over_under", "over_unders"), by_key, by_id)
            app_obj = _ud_related(ou_obj, ("appearance", "appearances"), by_key, by_id) or _ud_related(
                line_obj, ("appearance", "appearances"), by_key, by_id
            )
            player_obj = (
                _ud_related(app_obj, ("player", "players"), by_key, by_id)
                or _ud_related(ou_obj, ("player", "players"), by_key, by_id)
                or _ud_related(line_obj, ("player", "players"), by_key, by_id)
            )
            blob = _ud_blob(line_obj, ou_obj, app_obj, player_obj)
            if len(debug["sample_market_text"]) < 10 and any(x in blob.lower() for x in ("hits", "rbi", "home run")):
                debug["sample_market_text"].append(blob[:250])
            market = _ud_market(blob)
            if market is None or not _ud_active(line_obj, ou_obj, app_obj, player_obj):
                continue
            line = _ud_line_value(line_obj, ou_obj, app_obj)
            if line is None:
                continue
            if market == "HR" and not (0.5 <= line <= 2.5):
                continue
            if market == "HRR" and not (0.5 <= line <= 6.5):
                continue
            player = _ud_player_name(player_obj, app_obj, ou_obj, line_obj)
            if not player:
                continue
            event_title = _ud_blob(app_obj, ou_obj)[:180]
            parsed.append(
                UnderdogLine(
                    player=player,
                    market=market,
                    line=line,
                    event_title=event_title,
                    evidence=blob[:400],
                    line_id=str(line_obj.get("id", "")),
                )
            )

        # Strict flattened object fallback.
        for obj in objects:
            blob = _ud_blob(obj)
            market = _ud_market(blob)
            if market is None:
                continue
            line = _ud_line_value(obj)
            if line is None:
                continue
            player = _ud_player_name(obj)
            if not player:
                # title-regex fallback, only around explicit market names
                match = re.search(
                    r"([A-Z][A-Za-zÀ-ÿ.'’\-]+(?:\s+(?:[A-Z][A-Za-zÀ-ÿ.'’\-]+|Jr\.?|Sr\.?|II|III|IV)){1,5})"
                    r".{0,40}(?:Hits\s*\+\s*Runs\s*\+\s*RBIs?|Home\s+Runs?)",
                    blob,
                    flags=re.I,
                )
                player = _clean_player_name(match.group(1)) if match else ""
            if player:
                parsed.append(
                    UnderdogLine(
                        player=player,
                        market=market,
                        line=line,
                        evidence="flattened fallback: " + blob[:350],
                        line_id=str(obj.get("id", "")),
                    )
                )
        break

    dedup: Dict[Tuple[str, str], UnderdogLine] = {}
    for row in parsed:
        key = (norm_name(row.player), row.market)
        if key not in dedup or bool(row.line_id):
            dedup[key] = row
    rows = [asdict(row) for row in dedup.values()]
    rows.sort(key=lambda r: (r["market"], r["player"]))
    debug["parsed_rows"] = len(rows)
    debug["status"] = "OK" if rows else debug["status"]
    return rows, debug


def record_line_history(lines: Sequence[Mapping[str, Any]]) -> None:
    history = read_json(LINE_HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    now = datetime.now(timezone.utc).isoformat()
    existing = {(x.get("date"), x.get("player"), x.get("market"), x.get("line")) for x in history if isinstance(x, dict)}
    for row in lines:
        item = {
            "date": today_local().isoformat(),
            "captured_at": now,
            "player": row.get("player"),
            "market": row.get("market"),
            "line": row.get("line"),
            "line_id": row.get("line_id", ""),
        }
        key = (item["date"], item["player"], item["market"], item["line"])
        if key not in existing:
            history.append(item)
            existing.add(key)
    write_json(LINE_HISTORY_FILE, history[-5000:])


# =============================================================================
# MLB DATA ACCESS
# =============================================================================


TEAM_ABBR_ALIASES = {
    "AZ": "ARI",
    "CHW": "CWS",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
    "ATH": "OAK",
}


def team_abbr(value: Any) -> str:
    raw = str(value or "").upper().strip()
    return TEAM_ABBR_ALIASES.get(raw, raw)


def _extract_splits(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for block in payload.get("stats", []) or []:
        if isinstance(block, dict):
            rows.extend(x for x in block.get("splits", []) or [] if isinstance(x, dict))
    return rows


@st.cache_data(ttl=21600, show_spinner=False)
def mlb_bulk_stats(
    group: str,
    stat_type: str,
    year: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sit_code: Optional[str] = None,
    sport_ids: str = "1",
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "stats": stat_type,
        "group": group,
        "playerPool": "ALL",
        "limit": 5000,
        "sportIds": sport_ids,
        "hydrate": "person,team",
    }
    if year:
        params["season"] = year
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if sit_code:
        params["sitCodes"] = sit_code
    payload = HTTP.json(f"{MLB_BASE}/stats", params=params)
    return _extract_splits(payload)


@st.cache_data(ttl=86400, show_spinner=False)
def mlb_person_search(player_name: str) -> Dict[str, Any]:
    payload = HTTP.json(f"{MLB_BASE}/sports/1/players", params={"season": season_year()})
    people = (payload or {}).get("people", []) if isinstance(payload, dict) else []
    target = norm_name(player_name)
    exact = [p for p in people if norm_name(p.get("fullName")) == target]
    if exact:
        return exact[0]
    # Search API fallback.
    payload = HTTP.json(f"{MLB_BASE}/people/search", params={"names": player_name, "hydrate": "currentTeam"})
    candidates = (payload or {}).get("people", []) if isinstance(payload, dict) else []
    if not candidates:
        return {}
    candidates.sort(key=lambda p: _name_similarity(target, norm_name(p.get("fullName"))), reverse=True)
    return candidates[0]


def _name_similarity(a: str, b: str) -> float:
    aset, bset = set(a.split()), set(b.split())
    if not aset or not bset:
        return 0.0
    token = len(aset & bset) / len(aset | bset)
    prefix = 1.0 if a[:4] == b[:4] else 0.0
    return token * 0.85 + prefix * 0.15


@st.cache_data(ttl=300, show_spinner=False)
def mlb_schedule(game_date: str) -> List[Dict[str, Any]]:
    payload = HTTP.json(
        f"{MLB_BASE}/schedule",
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,venue,linescore,flags",
        },
    )
    games: List[Dict[str, Any]] = []
    for date_block in (payload or {}).get("dates", []) if isinstance(payload, dict) else []:
        games.extend(x for x in date_block.get("games", []) or [] if isinstance(x, dict))
    return games


@st.cache_data(ttl=180, show_spinner=False)
def mlb_live_feed(game_pk: int) -> Dict[str, Any]:
    return HTTP.json(f"{MLB_LIVE}/game/{game_pk}/feed/live") or {}


@st.cache_data(ttl=900, show_spinner=False)
def mlb_game_contexts(game_date: str) -> Dict[int, Dict[str, Any]]:
    contexts: Dict[int, Dict[str, Any]] = {}
    games = mlb_schedule(game_date)

    def load(game: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
        game_pk = safe_int(game.get("gamePk"), 0) or 0
        teams = game.get("teams", {}) or {}
        away = (teams.get("away", {}) or {}).get("team", {}) or {}
        home = (teams.get("home", {}) or {}).get("team", {}) or {}
        away_id = safe_int(away.get("id"), 0) or 0
        home_id = safe_int(home.get("id"), 0) or 0
        away_prob = (teams.get("away", {}) or {}).get("probablePitcher", {}) or {}
        home_prob = (teams.get("home", {}) or {}).get("probablePitcher", {}) or {}
        feed = mlb_live_feed(game_pk) if game_pk else {}
        game_data = feed.get("gameData", {}) if isinstance(feed, dict) else {}
        live_data = feed.get("liveData", {}) if isinstance(feed, dict) else {}
        players = game_data.get("players", {}) if isinstance(game_data, dict) else {}
        box = live_data.get("boxscore", {}) if isinstance(live_data, dict) else {}
        box_teams = box.get("teams", {}) if isinstance(box, dict) else {}
        lineups: Dict[int, List[Dict[str, Any]]] = {away_id: [], home_id: []}
        for side, team_id in (("away", away_id), ("home", home_id)):
            team_box = box_teams.get(side, {}) if isinstance(box_teams, dict) else {}
            order = team_box.get("battingOrder", []) if isinstance(team_box, dict) else []
            for idx, pid in enumerate(order, start=1):
                pdata = players.get(f"ID{pid}", {}) if isinstance(players, dict) else {}
                lineups[team_id].append(
                    {
                        "player_id": safe_int(pid, 0),
                        "player": pdata.get("fullName", ""),
                        "slot": idx,
                        "confirmed": True,
                    }
                )
        status = ((game.get("status") or {}).get("detailedState") or "")
        venue = game.get("venue", {}) or {}
        return game_pk, {
            "game_pk": game_pk,
            "date": game_date,
            "away_id": away_id,
            "home_id": home_id,
            "away": team_abbr(away.get("abbreviation") or away.get("teamCode")),
            "home": team_abbr(home.get("abbreviation") or home.get("teamCode")),
            "away_pitcher_id": safe_int(away_prob.get("id"), None),
            "away_pitcher": away_prob.get("fullName", ""),
            "home_pitcher_id": safe_int(home_prob.get("id"), None),
            "home_pitcher": home_prob.get("fullName", ""),
            "lineups": lineups,
            "status": status,
            "venue": venue.get("name", ""),
            "day_night": game_data.get("datetime", {}).get("dayNight", "") if isinstance(game_data, dict) else "",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(games)))) as executor:
        for game_pk, ctx in executor.map(load, games):
            if game_pk:
                contexts[game_pk] = ctx
    return contexts


@st.cache_data(ttl=3600, show_spinner=False)
def recent_lineup_usage(team_id: int, before_date: str, games_back: int = 10) -> Dict[int, Dict[str, Any]]:
    """Estimate normal batting slots and platoon risk from recent completed games."""
    end = datetime.fromisoformat(before_date).date() - timedelta(days=1)
    start = end - timedelta(days=22)
    payload = HTTP.json(
        f"{MLB_BASE}/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "gameType": "R",
        },
    )
    games: List[Dict[str, Any]] = []
    for block in (payload or {}).get("dates", []) if isinstance(payload, dict) else []:
        games.extend(block.get("games", []) or [])
    games = [g for g in games if "Final" in str((g.get("status") or {}).get("detailedState", ""))][-games_back:]
    usage: Dict[int, Dict[str, Any]] = {}
    for game in games:
        game_pk = safe_int(game.get("gamePk"), 0) or 0
        if not game_pk:
            continue
        feed = mlb_live_feed(game_pk)
        gd = feed.get("gameData", {}) if isinstance(feed, dict) else {}
        ld = feed.get("liveData", {}) if isinstance(feed, dict) else {}
        players = gd.get("players", {}) if isinstance(gd, dict) else {}
        box_teams = (ld.get("boxscore", {}) or {}).get("teams", {}) if isinstance(ld, dict) else {}
        for side in ("away", "home"):
            box_team = box_teams.get(side, {}) if isinstance(box_teams, dict) else {}
            tid = safe_int(((box_team.get("team") or {}).get("id")), 0) or 0
            if tid != team_id:
                continue
            order = box_team.get("battingOrder", []) or []
            for slot, pid in enumerate(order, start=1):
                pid = safe_int(pid, 0) or 0
                if not pid:
                    continue
                record = usage.setdefault(pid, {"starts": 0, "slots": [], "name": ""})
                record["starts"] += 1
                record["slots"].append(slot)
                record["name"] = (players.get(f"ID{pid}", {}) or {}).get("fullName", "")
    for record in usage.values():
        slots = record.get("slots", [])
        record["avg_slot"] = float(np.mean(slots)) if slots else None
        record["start_rate"] = record.get("starts", 0) / max(1, len(games))
    return usage


@st.cache_data(ttl=21600, show_spinner=False)
def individual_stats(
    player_id: int,
    group: str,
    stat_type: str,
    year: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    league_list_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"stats": stat_type, "group": group}
    if year:
        params["season"] = year
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if league_list_id:
        params["leagueListId"] = league_list_id
    payload = HTTP.json(f"{MLB_BASE}/people/{player_id}/stats", params=params)
    return _extract_splits(payload)


# =============================================================================
# STATCAST LEADERBOARDS / PITCH-TYPE DATA (BEST EFFORT)
# =============================================================================


SAVANT_BATTER_SELECTIONS = [
    "pa",
    "k_percent",
    "bb_percent",
    "batting_avg",
    "on_base_percent",
    "slg_percent",
    "isolated_power",
    "woba",
    "xba",
    "xslg",
    "xwoba",
    "barrel_batted_rate",
    "hard_hit_percent",
    "avg_hit_speed",
    "max_hit_speed",
    "launch_angle",
    "sweet_spot_percent",
    "squared_up_rate",
    "pull_percent",
    "oppo_percent",
    "sprint_speed",
    "whiff_percent",
    "z_swing_miss_percent",
    "chase_percent",
]
SAVANT_PITCHER_SELECTIONS = [
    "pa",
    "k_percent",
    "bb_percent",
    "batting_avg",
    "woba",
    "xba",
    "xslg",
    "xwoba",
    "barrel_batted_rate",
    "hard_hit_percent",
    "avg_hit_speed",
    "max_hit_speed",
    "launch_angle",
    "whiff_percent",
    "chase_percent",
]


@st.cache_data(ttl=21600, show_spinner=False)
def savant_custom_leaderboard(year: int, player_type: str) -> List[Dict[str, Any]]:
    selections = SAVANT_BATTER_SELECTIONS if player_type == "batter" else SAVANT_PITCHER_SELECTIONS
    params = {
        "year": year,
        "type": player_type,
        "filter": "",
        "sort": "pa",
        "sortDir": "desc",
        "min": 1,
        "selections": ",".join(selections),
        "chart": "false",
        "x": "pa",
        "y": "pa",
        "r": "no",
        "csv": "true",
    }
    text = HTTP.text("https://baseballsavant.mlb.com/leaderboard/custom", params=params, timeout=25)
    if not text or len(text) < 30:
        return []
    try:
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame.replace({np.nan: None}).to_dict("records")
    except Exception:
        return []


def savant_row_map(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    by_id: Dict[int, Mapping[str, Any]] = {}
    by_name: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pid = safe_int(
            row.get("player_id") or row.get("playerid") or row.get("player_id_lookup") or row.get("id"),
            None,
        )
        name = row.get("last_name, first_name") or row.get("player_name") or row.get("name")
        if isinstance(name, str) and "," in name:
            last, first = [x.strip() for x in name.split(",", 1)]
            name = f"{first} {last}"
        if pid:
            by_id[pid] = row
        if name:
            by_name[norm_name(name)] = row
    return by_id, by_name


@st.cache_data(ttl=21600, show_spinner=False)
def statcast_pitch_rows(player_id: int, role: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch pitch-level Statcast rows for optional arsenal matching.

    This endpoint is best effort. A failure produces a neutral arsenal score and
    lowers data quality instead of crashing or inventing data.
    """
    params = {
        "type": role,
        "player_type": role,
        "player_lookup[]": player_id,
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "hfGT": "R|",
        "group_by": "name-date",
    }
    text = HTTP.text("https://baseballsavant.mlb.com/statcast_search/csv", params=params, timeout=30)
    if not text or len(text) < 50:
        return []
    try:
        frame = pd.read_csv(io.StringIO(text), low_memory=False)
        keep = [
            c
            for c in [
                "game_date",
                "pitch_type",
                "pitch_name",
                "release_speed",
                "p_throws",
                "events",
                "description",
                "launch_speed",
                "launch_angle",
                "estimated_woba_using_speedangle",
                "woba_value",
                "delta_run_exp",
            ]
            if c in frame.columns
        ]
        return frame[keep].replace({np.nan: None}).to_dict("records") if keep else []
    except Exception:
        return []


# =============================================================================
# STAT SAMPLES / FEATURE ENGINE
# =============================================================================


LEAGUE_OUTCOME_PRIORS = {
    "K": 0.225,
    "BB": 0.082,
    "HBP": 0.011,
    "1B": 0.145,
    "2B": 0.047,
    "3B": 0.004,
    "HR": 0.031,
    "ROE": 0.010,
    "PRODUCTIVE_OUT": 0.055,
    "DOUBLE_PLAY": 0.021,
}


@dataclass
class HittingSample:
    label: str
    pa: float = 0.0
    ab: float = 0.0
    hits: float = 0.0
    doubles: float = 0.0
    triples: float = 0.0
    homers: float = 0.0
    walks: float = 0.0
    hbp: float = 0.0
    strikeouts: float = 0.0
    runs: float = 0.0
    rbi: float = 0.0
    sac_flies: float = 0.0
    gidp: float = 0.0
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None
    iso: Optional[float] = None
    woba: Optional[float] = None
    gb_rate: Optional[float] = None
    fb_rate: Optional[float] = None
    ld_rate: Optional[float] = None
    home_away: str = ""
    source: str = "MLB Stats API"

    @property
    def singles(self) -> float:
        return max(0.0, self.hits - self.doubles - self.triples - self.homers)

    def event_rate(self, event: str) -> Optional[float]:
        if self.pa <= 0:
            return None
        counts = {
            "K": self.strikeouts,
            "BB": self.walks,
            "HBP": self.hbp,
            "1B": self.singles,
            "2B": self.doubles,
            "3B": self.triples,
            "HR": self.homers,
            "DOUBLE_PLAY": self.gidp,
        }
        if event in counts:
            return clamp(counts[event] / self.pa, 0.0, 0.95)
        return None

    def expected_hrr_per_pa(self) -> Optional[float]:
        if self.pa <= 0:
            return None
        return clamp((self.hits + self.runs + self.rbi) / self.pa, 0.0, 2.5)


def sample_from_split(split: Optional[Mapping[str, Any]], label: str, source: str = "MLB Stats API") -> HittingSample:
    stat = (split or {}).get("stat", {}) if isinstance(split, Mapping) else {}
    pa = safe_float(stat.get("plateAppearances"), 0.0) or 0.0
    ab = safe_float(stat.get("atBats"), 0.0) or 0.0
    hits = safe_float(stat.get("hits"), 0.0) or 0.0
    doubles = safe_float(stat.get("doubles"), 0.0) or 0.0
    triples = safe_float(stat.get("triples"), 0.0) or 0.0
    homers = safe_float(stat.get("homeRuns"), 0.0) or 0.0
    walks = safe_float(stat.get("baseOnBalls"), 0.0) or 0.0
    hbp = safe_float(stat.get("hitByPitch"), 0.0) or 0.0
    strikeouts = safe_float(stat.get("strikeOuts"), 0.0) or 0.0
    runs = safe_float(stat.get("runs"), 0.0) or 0.0
    rbi = safe_float(stat.get("rbi"), 0.0) or 0.0
    sf = safe_float(stat.get("sacFlies"), 0.0) or 0.0
    gidp = safe_float(stat.get("groundIntoDoublePlay"), 0.0) or 0.0
    if pa <= 0:
        pa = ab + walks + hbp + sf
    avg = safe_float(stat.get("avg"), hits / ab if ab > 0 else None)
    obp = safe_float(stat.get("obp"), (hits + walks + hbp) / pa if pa > 0 else None)
    slg = safe_float(stat.get("slg"), None)
    if slg is None and ab > 0:
        total_bases = (hits - doubles - triples - homers) + 2 * doubles + 3 * triples + 4 * homers
        slg = total_bases / ab
    iso = safe_float(stat.get("iso"), (slg - avg) if slg is not None and avg is not None else None)
    ground = safe_float(stat.get("groundOuts"), None)
    air = safe_float(stat.get("airOuts"), None)
    gb_rate = ground / (ground + air) if ground is not None and air is not None and ground + air > 0 else None
    fb_rate = 1.0 - gb_rate if gb_rate is not None else None
    return HittingSample(
        label=label,
        pa=pa,
        ab=ab,
        hits=hits,
        doubles=doubles,
        triples=triples,
        homers=homers,
        walks=walks,
        hbp=hbp,
        strikeouts=strikeouts,
        runs=runs,
        rbi=rbi,
        sac_flies=sf,
        gidp=gidp,
        avg=avg,
        obp=obp,
        slg=slg,
        ops=safe_float(stat.get("ops"), (obp + slg) if obp is not None and slg is not None else None),
        iso=iso,
        woba=safe_float(stat.get("woba"), None),
        gb_rate=gb_rate,
        fb_rate=fb_rate,
        ld_rate=safe_float(stat.get("lineDriveRate"), None),
        source=source,
    )


def indexed_splits(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    by_id: Dict[int, Mapping[str, Any]] = {}
    by_name: Dict[str, Mapping[str, Any]] = {}
    for split in rows:
        player = split.get("player", {}) if isinstance(split, Mapping) else {}
        pid = safe_int(player.get("id"), None) if isinstance(player, Mapping) else None
        name = player.get("fullName") if isinstance(player, Mapping) else None
        if pid:
            by_id[pid] = split
        if name:
            by_name[norm_name(name)] = split
    return by_id, by_name


def first_split(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    return rows[0] if rows else None


def savant_value(row: Mapping[str, Any], aliases: Sequence[str], percent: bool = False) -> Optional[float]:
    lower = {str(k).lower().strip(): v for k, v in row.items()}
    for alias in aliases:
        value = safe_float(lower.get(alias.lower()), None)
        if value is not None:
            if percent and value > 1.5:
                value /= 100.0
            return value
    return None


@dataclass
class BatterFeatures:
    player_id: int
    player: str
    bats: str = ""
    age: Optional[float] = None
    current: HittingSample = field(default_factory=lambda: HittingSample("Current"))
    previous: HittingSample = field(default_factory=lambda: HittingSample("Previous"))
    l30: HittingSample = field(default_factory=lambda: HittingSample("L30"))
    l15: HittingSample = field(default_factory=lambda: HittingSample("L15"))
    l5: HittingSample = field(default_factory=lambda: HittingSample("L5 Display Only"))
    career: HittingSample = field(default_factory=lambda: HittingSample("Career"))
    split: HittingSample = field(default_factory=lambda: HittingSample("Split"))
    home_away: HittingSample = field(default_factory=lambda: HittingSample("Home/Away"))
    minors: HittingSample = field(default_factory=lambda: HittingSample("Minors"))
    xba: Optional[float] = None
    xwoba: Optional[float] = None
    xslg: Optional[float] = None
    whiff_rate: Optional[float] = None
    contact_rate: Optional[float] = None
    zone_contact_rate: Optional[float] = None
    chase_rate: Optional[float] = None
    csw_rate: Optional[float] = None
    avg_ev: Optional[float] = None
    max_ev: Optional[float] = None
    ev50: Optional[float] = None
    hard_hit_rate: Optional[float] = None
    barrel_rate: Optional[float] = None
    sweet_spot_rate: Optional[float] = None
    launch_angle: Optional[float] = None
    squared_up_rate: Optional[float] = None
    pull_rate: Optional[float] = None
    oppo_rate: Optional[float] = None
    sprint_speed: Optional[float] = None
    data_notes: List[str] = field(default_factory=list)

    def display_dict(self) -> Dict[str, Any]:
        c = self.current
        return {
            "AVG": c.avg,
            "OBP": c.obp,
            "SLG": c.slg,
            "ISO": c.iso,
            "wOBA": self.xwoba if self.xwoba is not None else c.woba,
            "xBA": self.xba,
            "xwOBA": self.xwoba,
            "xSLG": self.xslg,
            "K%": pct(c.event_rate("K")),
            "BB%": pct(c.event_rate("BB")),
            "Whiff%": pct(self.whiff_rate),
            "Contact%": pct(self.contact_rate),
            "Zone Contact%": pct(self.zone_contact_rate),
            "Chase%": pct(self.chase_rate),
            "CSW%": pct(self.csw_rate),
            "GB%": pct(c.gb_rate),
            "FB%": pct(c.fb_rate),
            "LD%": pct(c.ld_rate),
            "Avg EV": self.avg_ev,
            "Max EV": self.max_ev,
            "EV50": self.ev50,
            "Hard-Hit%": pct(self.hard_hit_rate),
            "Barrel%": pct(self.barrel_rate),
            "Sweet Spot%": pct(self.sweet_spot_rate),
            "Launch Angle": self.launch_angle,
            "Squared-Up%": pct(self.squared_up_rate),
            "Pull%": pct(self.pull_rate),
            "Oppo%": pct(self.oppo_rate),
            "Sprint Speed": self.sprint_speed,
            "L5 H+R+RBI / Game (display only)": ((self.l5.hits + self.l5.runs + self.l5.rbi) / max(self.l5.pa / 4.25, 1.0)) if self.l5.pa > 0 else None,
        }


# =============================================================================
# LEARNED TRUE-TALENT WEIGHTS
# =============================================================================


BASE_WEIGHT_PRIORS = {
    "current": 0.42,
    "previous": 0.29,
    "career": 0.10,
    "l30": 0.09,
    "l15": 0.035,
    "split": 0.065,
    "home_away": 0.02,
    "minors": 0.035,
}


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, safe_float(v, 0.0) or 0.0) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return dict(BASE_WEIGHT_PRIORS)
    return {k: v / total for k, v in clean.items()}


def load_learned_weights() -> Dict[str, Any]:
    payload = read_json(WEIGHT_FILE, {})
    if not isinstance(payload, dict):
        payload = {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else BASE_WEIGHT_PRIORS
    return {
        "weights": normalize_weights(weights),
        "learned": bool(payload.get("learned", False)),
        "rows": safe_int(payload.get("rows"), 0) or 0,
        "trained_at": payload.get("trained_at"),
        "validation_mse": safe_float(payload.get("validation_mse"), None),
        "source": payload.get("source", "prior"),
    }


def season_progress_weights(base: Mapping[str, float], game_date: date) -> Dict[str, float]:
    """Progressively move weight from previous/career to current season.

    Learned weights remain the center of the blend; season progress supplies a
    bounded transition rather than replacing learned coefficients.
    """
    opening = date(game_date.year, 3, 20)
    progress = clamp((game_date - opening).days / 190.0, 0.0, 1.0)
    w = dict(base)
    # Early season protects against unstable current samples; late season allows
    # current performance to become the primary foundation.
    current_mult = 0.55 + 0.85 * progress
    previous_mult = 1.35 - 0.75 * progress
    career_mult = 1.20 - 0.45 * progress
    w["current"] = w.get("current", 0.0) * current_mult
    w["previous"] = w.get("previous", 0.0) * previous_mult
    w["career"] = w.get("career", 0.0) * career_mult
    # Recent form remains intentionally capped.
    w["l30"] = min(w.get("l30", 0.0), 0.14)
    w["l15"] = min(w.get("l15", 0.0), 0.055)
    w["home_away"] = min(w.get("home_away", 0.0), 0.035)
    return normalize_weights(w)


def learn_weights_from_history(history: pd.DataFrame, trials: int = 2500) -> Dict[str, Any]:
    features = ["Comp_current", "Comp_previous", "Comp_career", "Comp_l30", "Comp_l15", "Comp_split", "Comp_home_away", "Comp_minors"]
    if history.empty or "Actual" not in history.columns:
        raise ValueError("No graded history with Actual values is available.")
    frame = history.copy()
    for col in features + ["Actual"]:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame = frame.dropna(subset=["Actual"])
    if len(frame) < 40:
        raise ValueError("At least 40 graded batter rows are required before learning weights.")
    # Median imputation uses only the training history; absent components then
    # contribute near the learned population center rather than zero.
    x = frame[features].copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0).to_numpy(float)
    y = frame["Actual"].to_numpy(float)
    rng = np.random.default_rng(20260715)
    base = np.array([BASE_WEIGHT_PRIORS[k] for k in ["current", "previous", "career", "l30", "l15", "split", "home_away", "minors"]])
    base = base / base.sum()
    best_w = base.copy()
    best_loss = float(np.mean((x @ best_w - y) ** 2))
    # Nonnegative constrained random search. Recent weights are capped and
    # current+previous+career must remain the primary foundation.
    for _ in range(trials):
        concentration = 60 if rng.random() < 0.65 else 18
        candidate = rng.dirichlet(np.maximum(0.15, base * concentration))
        candidate[3] = min(candidate[3], 0.16)  # L30
        candidate[4] = min(candidate[4], 0.07)  # L15
        candidate[6] = min(candidate[6], 0.04)  # home/away
        if candidate[:3].sum() < 0.52:
            continue
        candidate /= candidate.sum()
        loss = float(np.mean((x @ candidate - y) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_w = candidate
    names = ["current", "previous", "career", "l30", "l15", "split", "home_away", "minors"]
    result = {
        "weights": {name: round(float(weight), 6) for name, weight in zip(names, best_w)},
        "learned": True,
        "rows": int(len(frame)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_mse": round(best_loss, 6),
        "source": "graded historical random-search backtest",
    }
    write_json(WEIGHT_FILE, result)
    return result


# =============================================================================
# DATA BUNDLE
# =============================================================================


@dataclass
class DataBundle:
    year: int
    previous_year: int
    current_hitting: Dict[int, Mapping[str, Any]]
    current_hitting_name: Dict[str, Mapping[str, Any]]
    previous_hitting: Dict[int, Mapping[str, Any]]
    l30_hitting: Dict[int, Mapping[str, Any]]
    l15_hitting: Dict[int, Mapping[str, Any]]
    l5_hitting: Dict[int, Mapping[str, Any]]
    vs_left_hitting: Dict[int, Mapping[str, Any]]
    vs_right_hitting: Dict[int, Mapping[str, Any]]
    home_hitting: Dict[int, Mapping[str, Any]]
    away_hitting: Dict[int, Mapping[str, Any]]
    season_pitching: Dict[int, Mapping[str, Any]]
    l30_pitching: Dict[int, Mapping[str, Any]]
    vs_left_pitching: Dict[int, Mapping[str, Any]]
    vs_right_pitching: Dict[int, Mapping[str, Any]]
    l3_pitching: Dict[int, Mapping[str, Any]]
    savant_batters_id: Dict[int, Mapping[str, Any]]
    savant_batters_name: Dict[str, Mapping[str, Any]]
    savant_pitchers_id: Dict[int, Mapping[str, Any]]
    savant_pitchers_name: Dict[str, Mapping[str, Any]]
    games: Dict[int, Dict[str, Any]]


@st.cache_resource(show_spinner=False)
def build_data_bundle(game_date: str) -> DataBundle:
    gd = datetime.fromisoformat(game_date).date()
    year = gd.year
    previous = year - 1
    l30_start = (gd - timedelta(days=30)).isoformat()
    l15_start = (gd - timedelta(days=15)).isoformat()
    l5_start = (gd - timedelta(days=5)).isoformat()
    l3_start = (gd - timedelta(days=3)).isoformat()

    requests_to_run = {
        "cur_hit": ("hitting", "season", year, None, None, None),
        "prev_hit": ("hitting", "season", previous, None, None, None),
        "l30_hit": ("hitting", "byDateRange", None, l30_start, game_date, None),
        "l15_hit": ("hitting", "byDateRange", None, l15_start, game_date, None),
        "l5_hit": ("hitting", "byDateRange", None, l5_start, game_date, None),
        "vl_hit": ("hitting", "season", year, None, None, "vl"),
        "vr_hit": ("hitting", "season", year, None, None, "vr"),
        "home_hit": ("hitting", "season", year, None, None, "home"),
        "away_hit": ("hitting", "season", year, None, None, "away"),
        "cur_pitch": ("pitching", "season", year, None, None, None),
        "l30_pitch": ("pitching", "byDateRange", None, l30_start, game_date, None),
        "vl_pitch": ("pitching", "season", year, None, None, "vl"),
        "vr_pitch": ("pitching", "season", year, None, None, "vr"),
        "l3_pitch": ("pitching", "byDateRange", None, l3_start, game_date, None),
    }

    def run(spec: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        group, stat_type, yr, start, end, sit = spec
        return mlb_bulk_stats(group, stat_type, yr, start, end, sit)

    results: Dict[str, List[Dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(run, spec): name for name, spec in requests_to_run.items()}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = []

    indexes: Dict[str, Tuple[Dict[int, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]] = {
        name: indexed_splits(rows) for name, rows in results.items()
    }
    sav_bat = savant_custom_leaderboard(year, "batter")
    sav_pit = savant_custom_leaderboard(year, "pitcher")
    sav_bat_id, sav_bat_name = savant_row_map(sav_bat)
    sav_pit_id, sav_pit_name = savant_row_map(sav_pit)
    games = mlb_game_contexts(game_date)
    return DataBundle(
        year=year,
        previous_year=previous,
        current_hitting=indexes["cur_hit"][0],
        current_hitting_name=indexes["cur_hit"][1],
        previous_hitting=indexes["prev_hit"][0],
        l30_hitting=indexes["l30_hit"][0],
        l15_hitting=indexes["l15_hit"][0],
        l5_hitting=indexes["l5_hit"][0],
        vs_left_hitting=indexes["vl_hit"][0],
        vs_right_hitting=indexes["vr_hit"][0],
        home_hitting=indexes["home_hit"][0],
        away_hitting=indexes["away_hit"][0],
        season_pitching=indexes["cur_pitch"][0],
        l30_pitching=indexes["l30_pitch"][0],
        vs_left_pitching=indexes["vl_pitch"][0],
        vs_right_pitching=indexes["vr_pitch"][0],
        l3_pitching=indexes["l3_pitch"][0],
        savant_batters_id=sav_bat_id,
        savant_batters_name=sav_bat_name,
        savant_pitchers_id=sav_pit_id,
        savant_pitchers_name=sav_pit_name,
        games=games,
    )


# =============================================================================
# BATTER FEATURE ASSEMBLY
# =============================================================================


def player_identity(player_name: str, bundle: DataBundle) -> Dict[str, Any]:
    split = bundle.current_hitting_name.get(norm_name(player_name))
    if split:
        player = split.get("player", {}) or {}
        team = split.get("team", {}) or {}
        return {
            "id": safe_int(player.get("id"), 0) or 0,
            "name": player.get("fullName", player_name),
            "team_id": safe_int(team.get("id"), 0) or 0,
            "team": team_abbr(team.get("abbreviation") or team.get("teamCode")),
            "bats": ((player.get("batSide") or {}).get("code") or ""),
            "age": safe_float(player.get("currentAge"), None),
        }
    person = mlb_person_search(player_name)
    team = person.get("currentTeam", {}) if isinstance(person, dict) else {}
    return {
        "id": safe_int(person.get("id"), 0) or 0,
        "name": person.get("fullName", player_name),
        "team_id": safe_int(team.get("id"), 0) or 0,
        "team": team_abbr(team.get("abbreviation") or team.get("teamCode")),
        "bats": ((person.get("batSide") or {}).get("code") or ""),
        "age": safe_float(person.get("currentAge"), None),
    }


def matchup_context(identity: Mapping[str, Any], bundle: DataBundle, game_date: str) -> Dict[str, Any]:
    team_id = safe_int(identity.get("team_id"), 0) or 0
    player_id = safe_int(identity.get("id"), 0) or 0
    for ctx in bundle.games.values():
        if team_id not in {ctx.get("away_id"), ctx.get("home_id")}:
            continue
        is_home = team_id == ctx.get("home_id")
        opponent_id = ctx.get("away_id") if is_home else ctx.get("home_id")
        opponent = ctx.get("away") if is_home else ctx.get("home")
        pitcher_id = ctx.get("away_pitcher_id") if is_home else ctx.get("home_pitcher_id")
        pitcher_name = ctx.get("away_pitcher") if is_home else ctx.get("home_pitcher")
        lineup = (ctx.get("lineups") or {}).get(team_id, [])
        slot = None
        confirmed = False
        for item in lineup:
            if safe_int(item.get("player_id"), 0) == player_id or norm_name(item.get("player")) == norm_name(identity.get("name")):
                slot = safe_int(item.get("slot"), None)
                confirmed = True
                break
        recent = recent_lineup_usage(team_id, game_date) if team_id else {}
        recent_row = recent.get(player_id, {}) if player_id else {}
        if slot is None:
            slot = safe_int(recent_row.get("avg_slot"), None)
        return {
            "game_pk": ctx.get("game_pk"),
            "team_id": team_id,
            "team": identity.get("team"),
            "opponent_id": opponent_id,
            "opponent": opponent,
            "home": is_home,
            "pitcher_id": pitcher_id,
            "pitcher": pitcher_name or "TBD",
            "lineup_slot": slot,
            "lineup_confirmed": confirmed,
            "recent_start_rate": safe_float(recent_row.get("start_rate"), 0.0) or 0.0,
            "venue": ctx.get("venue", ""),
            "status": ctx.get("status", ""),
        }
    return {
        "game_pk": None,
        "team_id": team_id,
        "team": identity.get("team"),
        "opponent_id": 0,
        "opponent": "—",
        "home": None,
        "pitcher_id": None,
        "pitcher": "TBD",
        "lineup_slot": None,
        "lineup_confirmed": False,
        "recent_start_rate": 0.0,
        "venue": "",
        "status": "No scheduled MLB game matched",
    }


def build_batter_features(identity: Mapping[str, Any], context: Mapping[str, Any], bundle: DataBundle) -> BatterFeatures:
    pid = safe_int(identity.get("id"), 0) or 0
    current = sample_from_split(bundle.current_hitting.get(pid), "Current")
    previous = sample_from_split(bundle.previous_hitting.get(pid), "Previous")
    l30 = sample_from_split(bundle.l30_hitting.get(pid), "L30")
    l15 = sample_from_split(bundle.l15_hitting.get(pid), "L15")
    l5 = sample_from_split(bundle.l5_hitting.get(pid), "L5 Display Only")
    pitcher_id = safe_int(context.get("pitcher_id"), 0) or 0
    pitcher_hand = pitcher_hand_code(pitcher_id, bundle)
    split_source = bundle.vs_left_hitting if pitcher_hand == "L" else bundle.vs_right_hitting
    split = sample_from_split(split_source.get(pid), f"vs {pitcher_hand or '?'}")
    home_source = bundle.home_hitting if context.get("home") else bundle.away_hitting
    home_away = sample_from_split(home_source.get(pid), "Home" if context.get("home") else "Away")

    career = HittingSample("Career")
    minors = HittingSample("Minors")
    if current.pa + previous.pa < 300 and pid:
        career = sample_from_split(first_split(individual_stats(pid, "hitting", "career")), "Career")
    if current.pa < 180 and previous.pa < 120 and pid:
        minor_rows = individual_stats(pid, "hitting", "yearByYear", league_list_id="milb_full")
        if minor_rows:
            # Most recent meaningful minor-league season; translate rates toward
            # MLB with a conservative 0.82 MLE quality factor.
            chosen = max(minor_rows, key=lambda s: safe_float((s.get("stat") or {}).get("plateAppearances"), 0) or 0)
            minors = sample_from_split(chosen, "Minors", "MLB Stats API MiLB MLE")
            minors.hits *= 0.86
            minors.doubles *= 0.86
            minors.triples *= 0.90
            minors.homers *= 0.78
            minors.walks *= 0.88
            minors.strikeouts *= 1.08
            minors.runs *= 0.84
            minors.rbi *= 0.82

    savant = bundle.savant_batters_id.get(pid) or bundle.savant_batters_name.get(norm_name(identity.get("name"))) or {}
    whiff = savant_value(savant, ("whiff_percent", "whiff_percent_value"), percent=True)
    k_rate = current.event_rate("K")
    contact = 1.0 - whiff if whiff is not None else (1.0 - k_rate * 0.73 if k_rate is not None else None)
    notes: List[str] = []
    if not savant:
        notes.append("Baseball Savant batter row unavailable; expected metrics use neutral/MLB-stat fallbacks.")
    if current.pa < 80:
        notes.append("Small current-season sample; previous season/career/minor-league translation receives more weight.")

    return BatterFeatures(
        player_id=pid,
        player=str(identity.get("name") or ""),
        bats=str(identity.get("bats") or ""),
        age=safe_float(identity.get("age"), None),
        current=current,
        previous=previous,
        l30=l30,
        l15=l15,
        l5=l5,
        career=career,
        split=split,
        home_away=home_away,
        minors=minors,
        xba=savant_value(savant, ("xba", "est_ba", "estimated_ba_using_speedangle")),
        xwoba=savant_value(savant, ("xwoba", "est_woba", "estimated_woba_using_speedangle")),
        xslg=savant_value(savant, ("xslg", "est_slg")),
        whiff_rate=whiff,
        contact_rate=contact,
        zone_contact_rate=savant_value(savant, ("z_contact_percent", "zone_contact_percent"), percent=True),
        chase_rate=savant_value(savant, ("chase_percent", "oz_swing_percent"), percent=True),
        csw_rate=savant_value(savant, ("csw_percent",), percent=True),
        avg_ev=savant_value(savant, ("avg_hit_speed", "avg_exit_velocity")),
        max_ev=savant_value(savant, ("max_hit_speed", "max_exit_velocity")),
        ev50=savant_value(savant, ("ev50", "ev_50")),
        hard_hit_rate=savant_value(savant, ("hard_hit_percent", "hard_hit_rate"), percent=True),
        barrel_rate=savant_value(savant, ("barrel_batted_rate", "barrel_percent"), percent=True),
        sweet_spot_rate=savant_value(savant, ("sweet_spot_percent",), percent=True),
        launch_angle=savant_value(savant, ("launch_angle", "avg_launch_angle")),
        squared_up_rate=savant_value(savant, ("squared_up_rate", "squared_up_percent"), percent=True),
        pull_rate=savant_value(savant, ("pull_percent",), percent=True),
        oppo_rate=savant_value(savant, ("oppo_percent", "opposite_percent"), percent=True),
        sprint_speed=savant_value(savant, ("sprint_speed",)),
        data_notes=notes,
    )


# =============================================================================
# TRUE-TALENT BASELINE
# =============================================================================


@dataclass
class BaselineResult:
    probabilities: Dict[str, float]
    weights: Dict[str, float]
    component_hrr_per_pa: Dict[str, Optional[float]]
    age_factor: float
    data_quality: float
    notes: List[str]


def age_adjustment(age: Optional[float], game_date: date) -> float:
    if age is None:
        return 1.0
    opening = date(game_date.year, 3, 20)
    early_factor = 1.0 - clamp((game_date - opening).days / 110.0, 0.0, 1.0)
    if age <= 24:
        raw = 1.0 + 0.018
    elif age <= 29:
        raw = 1.0
    elif age <= 33:
        raw = 0.992
    elif age <= 36:
        raw = 0.978
    else:
        raw = 0.960
    return 1.0 + (raw - 1.0) * early_factor


def true_talent_baseline(features: BatterFeatures, game_date: date) -> BaselineResult:
    weight_meta = load_learned_weights()
    weights = season_progress_weights(weight_meta["weights"], game_date)
    samples = {
        "current": features.current,
        "previous": features.previous,
        "career": features.career,
        "l30": features.l30,
        "l15": features.l15,
        "split": features.split,
        "home_away": features.home_away,
        "minors": features.minors,
    }
    outcome_probs: Dict[str, float] = {}
    prior_strength = {
        "K": 175.0,
        "BB": 190.0,
        "HBP": 260.0,
        "1B": 210.0,
        "2B": 245.0,
        "3B": 520.0,
        "HR": 260.0,
        "DOUBLE_PLAY": 300.0,
    }
    for event in ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "DOUBLE_PLAY"):
        prior = LEAGUE_OUTCOME_PRIORS[event]
        numerator = prior * prior_strength[event]
        denominator = prior_strength[event]
        for name, sample in samples.items():
            rate = sample.event_rate(event)
            if rate is None or sample.pa <= 0:
                continue
            # Weighted effective PA prevents tiny L15/L30 samples from overpowering
            # the season/previous foundation.
            effective_pa = min(sample.pa, 750.0) * weights.get(name, 0.0)
            numerator += rate * effective_pa
            denominator += effective_pa
        outcome_probs[event] = clamp(numerator / max(1e-9, denominator), 0.0005, 0.80)

    # Expected-stat blend: only a bounded correction to hit/damage probabilities.
    if features.xba is not None and features.current.avg is not None:
        delta = clamp(features.xba - features.current.avg, -0.070, 0.070)
        outcome_probs["1B"] *= 1.0 + delta * 1.9
        outcome_probs["2B"] *= 1.0 + delta * 0.75
    if features.xslg is not None and features.current.slg is not None:
        delta = clamp(features.xslg - features.current.slg, -0.140, 0.140)
        outcome_probs["HR"] *= 1.0 + delta * 0.90
        outcome_probs["2B"] *= 1.0 + delta * 0.42
    if features.contact_rate is not None:
        contact_delta = clamp(features.contact_rate - 0.755, -0.12, 0.12)
        outcome_probs["K"] *= 1.0 - contact_delta * 1.35
        outcome_probs["1B"] *= 1.0 + contact_delta * 0.45
    if features.barrel_rate is not None:
        barrel_delta = clamp(features.barrel_rate - 0.075, -0.06, 0.12)
        outcome_probs["HR"] *= 1.0 + barrel_delta * 4.0
        outcome_probs["2B"] *= 1.0 + barrel_delta * 1.2
    if features.sprint_speed is not None:
        speed_delta = clamp((features.sprint_speed - 27.0) / 3.0, -1.0, 1.0)
        outcome_probs["1B"] *= 1.0 + 0.025 * speed_delta
        outcome_probs["3B"] *= 1.0 + 0.16 * speed_delta

    factor = age_adjustment(features.age, game_date)
    for event in ("1B", "2B", "3B", "HR"):
        outcome_probs[event] *= factor

    outcome_probs["ROE"] = LEAGUE_OUTCOME_PRIORS["ROE"]
    outcome_probs["PRODUCTIVE_OUT"] = LEAGUE_OUTCOME_PRIORS["PRODUCTIVE_OUT"]
    used = sum(outcome_probs.values())
    outcome_probs["OTHER_OUT"] = max(0.03, 1.0 - used)
    outcome_probs = normalize_outcome_probabilities(outcome_probs)

    component_hrr = {name: sample.expected_hrr_per_pa() for name, sample in samples.items()}
    available = sum(1 for sample in samples.values() if sample.pa >= 25)
    statcast_fields = sum(
        value is not None
        for value in (
            features.xba,
            features.xwoba,
            features.xslg,
            features.contact_rate,
            features.hard_hit_rate,
            features.barrel_rate,
            features.sprint_speed,
        )
    )
    quality = clamp(43 + available * 5.0 + statcast_fields * 2.6 + min(features.current.pa, 300) / 20.0, 35, 96)
    notes = list(features.data_notes)
    notes.append(
        "True-talent weights are learned from graded history when available; otherwise constrained priors are season-progress adjusted."
    )
    notes.append("L5 streaks are not used. L15 is tightly capped; L30 has moderate influence only.")
    return BaselineResult(outcome_probs, weights, component_hrr, factor, quality, notes)


def normalize_outcome_probabilities(probs: Mapping[str, float]) -> Dict[str, float]:
    keys = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "ROE", "PRODUCTIVE_OUT", "DOUBLE_PLAY", "OTHER_OUT"]
    clean = {k: max(0.00001, safe_float(probs.get(k), 0.0) or 0.0) for k in keys}
    total = sum(clean.values())
    return {k: v / total for k, v in clean.items()}


# =============================================================================
# PLATE-APPEARANCE PREDICTOR
# =============================================================================


@dataclass
class PAProjection:
    expected_pa: float
    p3: float
    p4: float
    p5: float
    p6: float
    pa_vs_starter: float
    pa_vs_bullpen: float
    pinch_hit_risk: float
    substitution_risk: float
    lineup_slot: Optional[int]
    lineup_confirmed: bool
    everyday_starter: bool
    team_implied_runs: float
    extra_inning_probability: float
    notes: List[str]


def team_offense_context(team_id: int, opponent_id: int, bundle: DataBundle) -> Dict[str, float]:
    hitters = [split for split in bundle.current_hitting.values() if safe_int(((split.get("team") or {}).get("id")), 0) == team_id]
    runs = sum(safe_float((split.get("stat") or {}).get("runs"), 0.0) or 0.0 for split in hitters)
    games = max([safe_float((split.get("stat") or {}).get("gamesPlayed"), 0.0) or 0.0 for split in hitters] + [0.0])
    team_rpg = clamp(runs / max(games, 1.0), 2.6, 7.0) if hitters else 4.35
    opponent_pitchers = [split for split in bundle.season_pitching.values() if safe_int(((split.get("team") or {}).get("id")), 0) == opponent_id]
    opp_era_vals = [safe_float((split.get("stat") or {}).get("era"), None) for split in opponent_pitchers]
    opp_era_vals = [v for v in opp_era_vals if v is not None]
    opp_era = float(np.median(opp_era_vals)) if opp_era_vals else 4.25
    implied = clamp(0.72 * team_rpg + 0.28 * opp_era, 2.5, 7.2)
    return {"team_rpg": team_rpg, "opponent_era": opp_era, "implied_runs": implied}


def pitcher_expected_innings(pitcher_id: Optional[int], bundle: DataBundle) -> Tuple[float, float, float]:
    if not pitcher_id or pitcher_id not in bundle.season_pitching:
        return 4.7, 20.5, 0.40
    stat = (bundle.season_pitching[pitcher_id].get("stat") or {})
    games_started = safe_float(stat.get("gamesStarted"), 0.0) or 0.0
    innings = parse_innings(stat.get("inningsPitched"))
    batters = safe_float(stat.get("battersFaced"), 0.0) or 0.0
    season_ip = innings / games_started if games_started > 0 else 4.2
    season_bf = batters / games_started if games_started > 0 else 18.5
    recent = (bundle.l30_pitching.get(pitcher_id) or {}).get("stat", {})
    recent_gs = safe_float(recent.get("gamesStarted"), 0.0) or 0.0
    recent_ip = parse_innings(recent.get("inningsPitched")) / recent_gs if recent_gs > 0 else season_ip
    projected_ip = clamp(0.67 * recent_ip + 0.33 * season_ip, 2.0, 7.0)
    projected_bf = clamp(0.67 * (projected_ip * 4.15) + 0.33 * season_bf, 8.5, 30.0)
    opener_risk = 0.70 if season_ip < 2.3 else 0.20 if season_ip < 4.0 else 0.05
    return projected_ip, projected_bf, opener_risk


def parse_innings(value: Any) -> float:
    try:
        text = str(value or "0")
        whole, _, frac = text.partition(".")
        return float(whole) + ({"1": 1 / 3, "2": 2 / 3}.get(frac[:1], 0.0))
    except Exception:
        return 0.0


def pa_distribution(expected: float, extra_inning_prob: float) -> Dict[int, float]:
    # Discrete normal-like probabilities on exactly 3-6 PA. The extra-inning
    # component adds a small amount to the 6-PA tail.
    values = np.array([3.0, 4.0, 5.0, 6.0])
    sigma = 0.72
    scores = np.exp(-0.5 * ((values - expected) / sigma) ** 2)
    scores[-1] *= 1.0 + 2.0 * extra_inning_prob
    scores /= scores.sum()
    return {int(v): float(p) for v, p in zip(values, scores)}


def project_plate_appearances(context: Mapping[str, Any], bundle: DataBundle) -> PAProjection:
    slot = safe_int(context.get("lineup_slot"), None)
    confirmed = bool(context.get("lineup_confirmed"))
    recent_start_rate = safe_float(context.get("recent_start_rate"), 0.0) or 0.0
    slot_prior = {1: 4.68, 2: 4.57, 3: 4.47, 4: 4.38, 5: 4.28, 6: 4.18, 7: 4.08, 8: 3.98, 9: 3.86}
    base = slot_prior.get(slot, 4.08)
    offense = team_offense_context(
        safe_int(context.get("team_id"), 0) or 0,
        safe_int(context.get("opponent_id"), 0) or 0,
        bundle,
    )
    run_adj = clamp((offense["implied_runs"] - 4.35) * 0.085, -0.16, 0.22)
    home_adj = -0.055 if context.get("home") else 0.035
    starter_ip, starter_bf, opener_risk = pitcher_expected_innings(safe_int(context.get("pitcher_id"), None), bundle)
    quality_adj = clamp((4.25 - offense["opponent_era"]) * 0.025, -0.08, 0.08)
    everyday = confirmed or recent_start_rate >= 0.72
    pinch_risk = 0.02 if confirmed else clamp(0.42 - recent_start_rate * 0.42, 0.06, 0.48)
    substitution = clamp(0.04 + (0.10 if slot and slot >= 7 else 0.0) + (0.12 if recent_start_rate < 0.55 else 0.0), 0.03, 0.34)
    extra_prob = clamp(0.085 + max(0.0, 4.1 - abs(offense["implied_runs"] - offense["opponent_era"])) * 0.003, 0.07, 0.115)
    expected = base + run_adj + home_adj + quality_adj + extra_prob * 0.16
    expected -= pinch_risk * 0.43 + substitution * 0.14
    expected = clamp(expected, 3.05, 5.25)
    dist = pa_distribution(expected, extra_prob)
    starter_share = clamp(starter_ip / 9.0 + 0.035 - opener_risk * 0.18, 0.18, 0.78)
    pa_vs_starter = expected * starter_share
    notes = [
        f"Batting slot {slot if slot else 'projected/unknown'} supplies the main opportunity prior.",
        f"Starter projection: {starter_ip:.2f} IP / {starter_bf:.1f} BF.",
        "Confirmed lineups override recent lineup usage; platoon/substitution risk is penalized before simulation.",
    ]
    return PAProjection(
        expected_pa=expected,
        p3=dist[3],
        p4=dist[4],
        p5=dist[5],
        p6=dist[6],
        pa_vs_starter=pa_vs_starter,
        pa_vs_bullpen=max(0.0, expected - pa_vs_starter),
        pinch_hit_risk=pinch_risk,
        substitution_risk=substitution,
        lineup_slot=slot,
        lineup_confirmed=confirmed,
        everyday_starter=everyday,
        team_implied_runs=offense["implied_runs"],
        extra_inning_probability=extra_prob,
        notes=notes,
    )


# =============================================================================
# PITCHER VULNERABILITY / BULLPEN
# =============================================================================


@dataclass
class PitcherVulnerability:
    pitcher_id: Optional[int]
    pitcher: str
    hand: str
    contact_allowed: float
    damage_allowed: float
    traffic_allowed: float
    platoon_vulnerability: float
    arsenal_matchup: float
    current_condition: float
    target_score: float
    expected_ip: float
    expected_bf: float
    opener_risk: float
    details: Dict[str, Any]
    notes: List[str]


@dataclass
class BullpenExposure:
    starter_weight: float
    bullpen_weight: float
    starter_ip: float
    starter_bf: float
    chance_faces_starter_twice: float
    chance_faces_starter_three_times: float
    bullpen_innings: float
    bullpen_xwoba_allowed: Optional[float]
    bullpen_k_rate: Optional[float]
    bullpen_bb_rate: Optional[float]
    bullpen_barrel_rate: Optional[float]
    bullpen_hard_hit_rate: Optional[float]
    bullpen_handedness: str
    reliever_availability: float
    bullpen_vulnerability_score: float
    notes: List[str]


def pitcher_hand_code(pitcher_id: int, bundle: DataBundle) -> str:
    split = bundle.season_pitching.get(pitcher_id)
    player = (split or {}).get("player", {}) if isinstance(split, Mapping) else {}
    return str(((player.get("pitchHand") or {}).get("code") or "")).upper()


def rate_from_pitch_stat(stat: Mapping[str, Any], key: str, denominator_key: str = "battersFaced") -> Optional[float]:
    num = safe_float(stat.get(key), None)
    den = safe_float(stat.get(denominator_key), None)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def score_bad_high(value: Optional[float], league: float, scale: float) -> float:
    if value is None:
        return 50.0
    return clamp(50.0 + (value - league) / max(scale, 1e-9) * 15.0, 5.0, 95.0)


def score_bad_low(value: Optional[float], league: float, scale: float) -> float:
    if value is None:
        return 50.0
    return clamp(50.0 + (league - value) / max(scale, 1e-9) * 15.0, 5.0, 95.0)


def arsenal_matchup_score(
    batter: BatterFeatures,
    pitcher_id: Optional[int],
    game_date: date,
    deep_enabled: bool,
) -> Tuple[float, Dict[str, Any], str]:
    if not deep_enabled or not pitcher_id or not batter.player_id:
        return 50.0, {}, "Pitch-type deep data not requested for this row; neutral score used."
    start = (game_date - timedelta(days=90)).isoformat()
    end = game_date.isoformat()
    pitcher_rows = statcast_pitch_rows(pitcher_id, "pitcher", start, end)
    batter_rows = statcast_pitch_rows(batter.player_id, "batter", start, end)
    if not pitcher_rows or not batter_rows:
        return 50.0, {}, "Pitch-level Statcast data unavailable; neutral arsenal score used."
    pit = pd.DataFrame(pitcher_rows)
    bat = pd.DataFrame(batter_rows)
    pitch_col_p = "pitch_name" if "pitch_name" in pit.columns else "pitch_type"
    pitch_col_b = "pitch_name" if "pitch_name" in bat.columns else "pitch_type"
    if pitch_col_p not in pit.columns or pitch_col_b not in bat.columns:
        return 50.0, {}, "Pitch type columns unavailable; neutral arsenal score used."
    usage = pit[pitch_col_p].value_counts(normalize=True)
    details: Dict[str, Any] = {}
    weighted = 0.0
    total = 0.0
    for pitch_name, use in usage.head(7).items():
        subset = bat[bat[pitch_col_b] == pitch_name]
        if len(subset) < 8:
            run_value = 0.0
            woba = None
        else:
            woba_col = "estimated_woba_using_speedangle" if "estimated_woba_using_speedangle" in subset.columns else "woba_value"
            woba = pd.to_numeric(subset.get(woba_col), errors="coerce").mean() if woba_col in subset.columns else None
            rv = pd.to_numeric(subset.get("delta_run_exp"), errors="coerce").mean() if "delta_run_exp" in subset.columns else 0.0
            # Aggressive small-sample shrinkage.
            shrink = len(subset) / (len(subset) + 75.0)
            run_value = float(rv if pd.notna(rv) else 0.0) * shrink
            if woba is not None and pd.notna(woba):
                run_value += (float(woba) - 0.315) * 0.45 * shrink
        pitch_score = clamp(50 + run_value * 180.0, 15, 85)
        weighted += float(use) * pitch_score
        total += float(use)
        details[str(pitch_name)] = {"usage": round(float(use), 3), "batter_pa": int(len(subset)), "score": round(pitch_score, 1)}
    return (weighted / total if total > 0 else 50.0), details, "Pitch usage × batter pitch-type performance with strong sample shrinkage."


@st.cache_data(ttl=21600, show_spinner=False)
def pitcher_current_condition_metrics(pitcher_id: int, game_date: str, enabled: bool) -> Dict[str, Any]:
    """Best-effort velocity, pitch-mix, command, rest and return-risk context."""
    neutral = {
        "velocity_trend": None,
        "pitch_mix_shift": None,
        "command_ball_rate": None,
        "recent_hard_hit_rate": None,
        "rest_days": None,
        "injury_return_risk": 0.0,
        "pitch_count_limit_risk": 0.0,
    }
    if not enabled or not pitcher_id:
        return neutral
    gd = datetime.fromisoformat(game_date).date()
    rows = statcast_pitch_rows(pitcher_id, "pitcher", (gd - timedelta(days=100)).isoformat(), gd.isoformat())
    if not rows:
        return neutral
    df = pd.DataFrame(rows)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"]).sort_values("game_date")
    if df.empty:
        return neutral
    recent_cut = pd.Timestamp(gd - timedelta(days=30))
    prior_cut = pd.Timestamp(gd - timedelta(days=90))
    recent = df[df.get("game_date", pd.Series(index=df.index, dtype="datetime64[ns]")) >= recent_cut] if "game_date" in df.columns else df.tail(max(1, len(df)//3))
    prior = df[(df.get("game_date") < recent_cut) & (df.get("game_date") >= prior_cut)] if "game_date" in df.columns else df.head(max(1, len(df)*2//3))
    vel_recent = pd.to_numeric(recent.get("release_speed"), errors="coerce").mean() if "release_speed" in recent.columns else np.nan
    vel_prior = pd.to_numeric(prior.get("release_speed"), errors="coerce").mean() if "release_speed" in prior.columns else np.nan
    velocity_trend = float(vel_recent - vel_prior) if pd.notna(vel_recent) and pd.notna(vel_prior) else None
    pitch_col = "pitch_name" if "pitch_name" in df.columns else "pitch_type" if "pitch_type" in df.columns else None
    mix_shift = None
    if pitch_col and not recent.empty and not prior.empty:
        r_mix = recent[pitch_col].value_counts(normalize=True)
        p_mix = prior[pitch_col].value_counts(normalize=True)
        all_pitches = set(r_mix.index) | set(p_mix.index)
        mix_shift = float(sum(abs(float(r_mix.get(k,0)) - float(p_mix.get(k,0))) for k in all_pitches) / 2.0)
    ball_rate = None
    if "description" in recent.columns and len(recent):
        desc = recent["description"].astype(str).str.lower()
        ball_rate = float(desc.str.contains("ball|blocked_ball|pitchout", regex=True).mean())
    hard_rate = None
    if "launch_speed" in recent.columns:
        ev = pd.to_numeric(recent["launch_speed"], errors="coerce").dropna()
        hard_rate = float((ev >= 95).mean()) if len(ev) else None
    rest_days = None
    injury_return = 0.0
    pitch_limit = 0.0
    if "game_date" in df.columns and len(df):
        dates = sorted(pd.Series(df["game_date"].dt.date.unique()).dropna().tolist())
        if dates:
            rest_days = max(0, (gd - dates[-1]).days)
            if len(dates) >= 2:
                prior_gap = max((b-a).days for a,b in zip(dates[:-1], dates[1:]))
                injury_return = 0.75 if prior_gap >= 25 and (gd - dates[-1]).days <= 10 else 0.25 if prior_gap >= 15 else 0.0
            recent_games = recent.groupby(recent["game_date"].dt.date).size() if not recent.empty else pd.Series(dtype=float)
            if len(recent_games):
                avg_recent_pitches = float(recent_games.mean())
                pitch_limit = clamp((82.0 - avg_recent_pitches) / 35.0, 0.0, 0.9)
    return {
        "velocity_trend": velocity_trend,
        "pitch_mix_shift": mix_shift,
        "command_ball_rate": ball_rate,
        "recent_hard_hit_rate": hard_rate,
        "rest_days": rest_days,
        "injury_return_risk": injury_return,
        "pitch_count_limit_risk": pitch_limit,
    }


@st.cache_data(ttl=21600, show_spinner=False)
def bvp_minor_score(batter_id: int, pitcher_id: int, year: int, enabled: bool) -> Tuple[float, int]:
    """Very small BvP input with aggressive shrinkage; never a primary driver."""
    if not enabled or not batter_id or not pitcher_id:
        return 50.0, 0
    payload = HTTP.json(
        f"{MLB_BASE}/people/{batter_id}/stats",
        params={"stats":"vsPlayer", "group":"hitting", "season":year, "opposingPlayerId":pitcher_id},
    )
    split = first_split(_extract_splits(payload))
    if not split:
        return 50.0, 0
    sample = sample_from_split(split, "BvP")
    if sample.pa <= 0:
        return 50.0, 0
    ops = sample.ops if sample.ops is not None else 0.710
    raw = clamp(50 + (ops - 0.710) / 0.18 * 15.0, 15, 85)
    shrink = sample.pa / (sample.pa + 80.0)
    return 50.0 + (raw - 50.0) * shrink, int(sample.pa)


def build_pitcher_vulnerability(
    context: Mapping[str, Any],
    batter: BatterFeatures,
    bundle: DataBundle,
    game_date: date,
    deep_enabled: bool,
) -> PitcherVulnerability:
    pid = safe_int(context.get("pitcher_id"), None)
    name = str(context.get("pitcher") or "TBD")
    hand = pitcher_hand_code(pid or 0, bundle) if pid else ""
    split = bundle.season_pitching.get(pid or 0, {})
    stat = split.get("stat", {}) if isinstance(split, Mapping) else {}
    recent = (bundle.l30_pitching.get(pid or 0) or {}).get("stat", {})
    savant = bundle.savant_pitchers_id.get(pid or 0) or bundle.savant_pitchers_name.get(norm_name(name)) or {}
    bf = safe_float(stat.get("battersFaced"), 0.0) or 0.0
    hits = safe_float(stat.get("hits"), 0.0) or 0.0
    walks = safe_float(stat.get("baseOnBalls"), 0.0) or 0.0
    hbp = safe_float(stat.get("hitBatsmen"), safe_float(stat.get("hitByPitch"), 0.0)) or 0.0
    strikeouts = safe_float(stat.get("strikeOuts"), 0.0) or 0.0
    homers = safe_float(stat.get("homeRuns"), 0.0) or 0.0
    innings = parse_innings(stat.get("inningsPitched"))
    avg_allowed = hits / max(1.0, safe_float(stat.get("atBats"), bf - walks - hbp) or 1.0)
    k_rate = strikeouts / bf if bf > 0 else None
    bb_rate = walks / bf if bf > 0 else None
    hbp_rate = hbp / bf if bf > 0 else None
    hr_rate = homers / bf if bf > 0 else None
    whip = (hits + walks) / innings if innings > 0 else safe_float(stat.get("whip"), None)
    xba = savant_value(savant, ("xba", "est_ba", "estimated_ba_using_speedangle"))
    xslg = savant_value(savant, ("xslg", "est_slg"))
    xwoba = savant_value(savant, ("xwoba", "est_woba", "estimated_woba_using_speedangle"))
    whiff = savant_value(savant, ("whiff_percent",), percent=True)
    hard_hit = savant_value(savant, ("hard_hit_percent",), percent=True)
    barrel = savant_value(savant, ("barrel_batted_rate", "barrel_percent"), percent=True)
    avg_ev = savant_value(savant, ("avg_hit_speed", "avg_exit_velocity"))
    max_ev = savant_value(savant, ("max_hit_speed", "max_exit_velocity"))

    contact_score = np.mean(
        [
            score_bad_high(xba, 0.245, 0.035),
            score_bad_high(avg_allowed, 0.245, 0.035),
            score_bad_low(whiff, 0.245, 0.055),
            score_bad_low(k_rate, 0.225, 0.055),
        ]
    )
    damage_score = np.mean(
        [
            score_bad_high(xslg, 0.410, 0.075),
            score_bad_high(xwoba, 0.315, 0.045),
            score_bad_high(barrel, 0.075, 0.035),
            score_bad_high(hard_hit, 0.385, 0.070),
            score_bad_high(hr_rate, 0.031, 0.014),
            score_bad_high(avg_ev, 88.5, 2.5),
        ]
    )
    traffic_score = np.mean(
        [
            score_bad_high(bb_rate, 0.082, 0.035),
            score_bad_high(hbp_rate, 0.011, 0.009),
            score_bad_high(whip, 1.28, 0.28),
            score_bad_low((k_rate - bb_rate) if k_rate is not None and bb_rate is not None else None, 0.143, 0.055),
        ]
    )
    batter_hand = batter.bats.upper()
    split_map = bundle.vs_left_pitching if batter_hand == "L" else bundle.vs_right_pitching
    p_split = (split_map.get(pid or 0) or {}).get("stat", {})
    p_bf = safe_float(p_split.get("battersFaced"), 0.0) or 0.0
    p_hits = safe_float(p_split.get("hits"), 0.0) or 0.0
    p_bb = safe_float(p_split.get("baseOnBalls"), 0.0) or 0.0
    p_k = safe_float(p_split.get("strikeOuts"), 0.0) or 0.0
    p_hr = safe_float(p_split.get("homeRuns"), 0.0) or 0.0
    platoon_score = np.mean(
        [
            score_bad_high(p_hits / max(p_bf - p_bb, 1.0) if p_bf else None, 0.245, 0.040),
            score_bad_high(p_bb / p_bf if p_bf else None, 0.082, 0.040),
            score_bad_low(p_k / p_bf if p_bf else None, 0.225, 0.060),
            score_bad_high(p_hr / p_bf if p_bf else None, 0.031, 0.017),
        ]
    )
    arsenal, arsenal_details, arsenal_note = arsenal_matchup_score(batter, pid, game_date, deep_enabled)
    condition = pitcher_current_condition_metrics(pid or 0, game_date.isoformat(), deep_enabled)
    bvp_score, bvp_pa = bvp_minor_score(batter.player_id, pid or 0, bundle.year, deep_enabled)
    season_era = safe_float(stat.get("era"), None)
    recent_era = safe_float(recent.get("era"), None)
    season_bb = rate_from_pitch_stat(stat, "baseOnBalls")
    recent_bb = rate_from_pitch_stat(recent, "baseOnBalls")
    velocity_score = 50.0 if condition.get("velocity_trend") is None else clamp(50.0 - condition["velocity_trend"] / 1.5 * 15.0, 15, 85)
    mix_score = 50.0 if condition.get("pitch_mix_shift") is None else clamp(50.0 + condition["pitch_mix_shift"] * 35.0, 35, 80)
    command_score = score_bad_high(condition.get("command_ball_rate"), 0.365, 0.045)
    hard_contact_score = score_bad_high(condition.get("recent_hard_hit_rate"), 0.385, 0.075)
    return_score = 50.0 + 20.0 * safe_float(condition.get("injury_return_risk"), 0.0)
    pitch_limit_score = 50.0 + 18.0 * safe_float(condition.get("pitch_count_limit_risk"), 0.0)
    current_score = np.mean(
        [
            score_bad_high(recent_era, season_era if season_era is not None else 4.25, 1.1),
            score_bad_high(recent_bb, season_bb if season_bb is not None else 0.082, 0.032),
            score_bad_high(
                safe_float(recent.get("homeRuns"), None) / max(safe_float(recent.get("battersFaced"), 0.0) or 0.0, 1.0)
                if recent
                else None,
                hr_rate if hr_rate is not None else 0.031,
                0.015,
            ),
            velocity_score,
            mix_score,
            command_score,
            hard_contact_score,
            return_score,
            pitch_limit_score,
        ]
    )
    exp_ip, exp_bf, opener_risk = pitcher_expected_innings(pid, bundle)
    # BvP is intentionally only 2% of the target score and is heavily shrunk.
    target = float(
        np.average(
            [contact_score, damage_score, traffic_score, platoon_score, arsenal, current_score, bvp_score],
            weights=[0.195, 0.235, 0.165, 0.165, 0.115, 0.105, 0.020],
        )
    )
    details = {
        "xBA Allowed": xba,
        "AVG Allowed": avg_allowed,
        "xSLG Allowed": xslg,
        "xwOBA Allowed": xwoba,
        "K%": pct(k_rate),
        "BB%": pct(bb_rate),
        "Whiff%": pct(whiff),
        "WHIP": whip,
        "Barrel%": pct(barrel),
        "Hard-Hit%": pct(hard_hit),
        "Avg EV Allowed": avg_ev,
        "Max EV Allowed": max_ev,
        "HR Probability/PA": pct(hr_rate, 2),
        "Arsenal Details": arsenal_details,
        "Velocity Trend MPH": condition.get("velocity_trend"),
        "Pitch Mix Shift": condition.get("pitch_mix_shift"),
        "Command Ball Rate": pct(condition.get("command_ball_rate")),
        "Recent Hard-Hit%": pct(condition.get("recent_hard_hit_rate")),
        "Rest Days": condition.get("rest_days"),
        "Injury Return Risk": condition.get("injury_return_risk"),
        "Pitch Count Limit Risk": condition.get("pitch_count_limit_risk"),
        "BvP Score (2% weight)": round(bvp_score,1),
        "BvP PA": bvp_pa,
    }
    notes = [arsenal_note]
    if pid is None:
        notes.append("Probable starter is not posted; neutral starter inputs are used and data quality is reduced.")
    return PitcherVulnerability(
        pitcher_id=pid,
        pitcher=name,
        hand=hand,
        contact_allowed=round(float(contact_score), 1),
        damage_allowed=round(float(damage_score), 1),
        traffic_allowed=round(float(traffic_score), 1),
        platoon_vulnerability=round(float(platoon_score), 1),
        arsenal_matchup=round(float(arsenal), 1),
        current_condition=round(float(current_score), 1),
        target_score=round(target, 1),
        expected_ip=round(exp_ip, 2),
        expected_bf=round(exp_bf, 1),
        opener_risk=round(opener_risk, 3),
        details=details,
        notes=notes,
    )


def aggregate_bullpen(opponent_id: int, starter_id: Optional[int], bundle: DataBundle) -> Dict[str, Any]:
    rows = []
    for pid, split in bundle.season_pitching.items():
        team = split.get("team", {}) or {}
        if safe_int(team.get("id"), 0) != opponent_id or pid == starter_id:
            continue
        stat = split.get("stat", {}) or {}
        gp = safe_float(stat.get("gamesPlayed"), safe_float(stat.get("gamesPitched"), 0.0)) or 0.0
        gs = safe_float(stat.get("gamesStarted"), 0.0) or 0.0
        if gp <= 0 or gs / gp > 0.45:
            continue
        rows.append((pid, stat))
    bf = sum(safe_float(stat.get("battersFaced"), 0.0) or 0.0 for _, stat in rows)
    hits = sum(safe_float(stat.get("hits"), 0.0) or 0.0 for _, stat in rows)
    walks = sum(safe_float(stat.get("baseOnBalls"), 0.0) or 0.0 for _, stat in rows)
    strikeouts = sum(safe_float(stat.get("strikeOuts"), 0.0) or 0.0 for _, stat in rows)
    homers = sum(safe_float(stat.get("homeRuns"), 0.0) or 0.0 for _, stat in rows)
    xwoba_values = []
    barrel_values = []
    hard_values = []
    hand_bf = {"L":0.0, "R":0.0, "S":0.0}
    for pid, stat_row in rows:
        split_row = bundle.season_pitching.get(pid, {})
        player_meta = split_row.get("player", {}) if isinstance(split_row, Mapping) else {}
        hand = str(((player_meta.get("pitchHand") or {}).get("code") or "")).upper()
        if hand in hand_bf:
            hand_bf[hand] += safe_float(stat_row.get("battersFaced"), 0.0) or 0.0
        sav = bundle.savant_pitchers_id.get(pid, {})
        xw = savant_value(sav, ("xwoba", "estimated_woba_using_speedangle"))
        br = savant_value(sav, ("barrel_batted_rate", "barrel_percent"), percent=True)
        hh = savant_value(sav, ("hard_hit_percent",), percent=True)
        if xw is not None:
            xwoba_values.append(xw)
        if br is not None:
            barrel_values.append(br)
        if hh is not None:
            hard_values.append(hh)
    recent_ip = 0.0
    for pid, split in bundle.l3_pitching.items():
        team = split.get("team", {}) or {}
        if safe_int(team.get("id"), 0) == opponent_id and pid != starter_id:
            recent_ip += parse_innings((split.get("stat") or {}).get("inningsPitched"))
    availability = clamp(1.0 - max(0.0, recent_ip - 8.5) / 18.0, 0.35, 1.0)
    return {
        "bf": bf,
        "avg": hits / max(bf - walks, 1.0) if bf else None,
        "k_rate": strikeouts / bf if bf else None,
        "bb_rate": walks / bf if bf else None,
        "hr_rate": homers / bf if bf else None,
        "xwoba": float(np.mean(xwoba_values)) if xwoba_values else None,
        "barrel": float(np.mean(barrel_values)) if barrel_values else None,
        "hard_hit": float(np.mean(hard_values)) if hard_values else None,
        "availability": availability,
        "recent_ip": recent_ip,
        "relievers": len(rows),
        "handedness": " / ".join(f"{h} {v/sum(hand_bf.values()):.0%}" for h,v in hand_bf.items() if v > 0) if sum(hand_bf.values()) > 0 else "Unknown",
    }


def build_bullpen_exposure(
    context: Mapping[str, Any],
    pa: PAProjection,
    pitcher: PitcherVulnerability,
    bundle: DataBundle,
) -> BullpenExposure:
    opponent_id = safe_int(context.get("opponent_id"), 0) or 0
    bp = aggregate_bullpen(opponent_id, pitcher.pitcher_id, bundle)
    starter_weight = clamp(pa.pa_vs_starter / max(pa.expected_pa, 0.01), 0.15, 0.82)
    bullpen_weight = 1.0 - starter_weight
    bp_score = float(
        np.mean(
            [
                score_bad_high(bp.get("xwoba"), 0.315, 0.045),
                score_bad_low(bp.get("k_rate"), 0.235, 0.055),
                score_bad_high(bp.get("bb_rate"), 0.085, 0.035),
                score_bad_high(bp.get("barrel"), 0.075, 0.035),
                score_bad_high(bp.get("hard_hit"), 0.385, 0.07),
                score_bad_low(bp.get("availability"), 0.78, 0.25),
            ]
        )
    )
    return BullpenExposure(
        starter_weight=starter_weight,
        bullpen_weight=bullpen_weight,
        starter_ip=pitcher.expected_ip,
        starter_bf=pitcher.expected_bf,
        chance_faces_starter_twice=clamp((pitcher.expected_bf - 9) / 10, 0.05, 0.98),
        chance_faces_starter_three_times=clamp((pitcher.expected_bf - 18) / 9, 0.0, 0.70),
        bullpen_innings=max(0.0, 9.0 - pitcher.expected_ip),
        bullpen_xwoba_allowed=bp.get("xwoba"),
        bullpen_k_rate=bp.get("k_rate"),
        bullpen_bb_rate=bp.get("bb_rate"),
        bullpen_barrel_rate=bp.get("barrel"),
        bullpen_hard_hit_rate=bp.get("hard_hit"),
        bullpen_handedness=bp.get("handedness", "Unknown"),
        reliever_availability=bp.get("availability", 0.7),
        bullpen_vulnerability_score=round(bp_score, 1),
        notes=[
            f"Starter/bullpen blend: {starter_weight:.0%}/{bullpen_weight:.0%}.",
            f"Opponent bullpen used approximately {bp.get('recent_ip', 0):.1f} IP over the last three days.",
        ],
    )


# =============================================================================
# PA OUTCOME MODEL / OPTIONAL LIGHTGBM ENSEMBLE
# =============================================================================


OUTCOME_ORDER = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "ROE", "PRODUCTIVE_OUT", "OTHER_OUT", "DOUBLE_PLAY"]


def adjust_outcomes_for_matchup(
    baseline: BaselineResult,
    pitcher: PitcherVulnerability,
    bullpen: BullpenExposure,
    batter: BatterFeatures,
) -> Dict[str, float]:
    probs = dict(baseline.probabilities)
    combined_contact = pitcher.contact_allowed * bullpen.starter_weight + bullpen.bullpen_vulnerability_score * bullpen.bullpen_weight
    combined_damage = pitcher.damage_allowed * bullpen.starter_weight + bullpen.bullpen_vulnerability_score * bullpen.bullpen_weight
    combined_traffic = pitcher.traffic_allowed * bullpen.starter_weight + bullpen.bullpen_vulnerability_score * bullpen.bullpen_weight
    platoon = pitcher.platoon_vulnerability
    arsenal = pitcher.arsenal_matchup
    contact_adj = clamp((combined_contact - 50.0) / 100.0, -0.30, 0.30)
    damage_adj = clamp((combined_damage - 50.0) / 100.0, -0.30, 0.30)
    traffic_adj = clamp((combined_traffic - 50.0) / 100.0, -0.30, 0.30)
    platoon_adj = clamp((platoon - 50.0) / 100.0, -0.25, 0.25)
    arsenal_adj = clamp((arsenal - 50.0) / 100.0, -0.20, 0.20)
    probs["K"] *= 1.0 - contact_adj * 0.38 - arsenal_adj * 0.18
    probs["BB"] *= 1.0 + traffic_adj * 0.42
    probs["HBP"] *= 1.0 + traffic_adj * 0.08
    probs["1B"] *= 1.0 + contact_adj * 0.35 + platoon_adj * 0.15
    probs["2B"] *= 1.0 + damage_adj * 0.34 + arsenal_adj * 0.12
    probs["3B"] *= 1.0 + contact_adj * 0.10
    probs["HR"] *= 1.0 + damage_adj * 0.62 + platoon_adj * 0.18 + arsenal_adj * 0.22
    # Opposing defense proxy: sprint speed helps beat infield/ROE outcomes.
    if batter.sprint_speed is not None:
        probs["ROE"] *= 1.0 + clamp((batter.sprint_speed - 27.0) / 20.0, -0.08, 0.12)
    return normalize_outcome_probabilities(probs)


def optional_lightgbm_probabilities(feature_row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Load a trained multiclass LightGBM artifact when the user supplies one.

    No untrained model is fabricated. Without the artifact the hierarchical
    Bayesian model remains active and the app reports Bayesian-only mode.
    """
    model_path = Path(os.getenv("BATTER_LGBM_MODEL", STORAGE_DIR / "pa_outcome_lightgbm.txt"))
    if not model_path.exists():
        return None
    try:
        import lightgbm as lgb  # optional runtime dependency

        model = lgb.Booster(model_file=str(model_path))
        names = model.feature_name()
        values = np.array([[safe_float(feature_row.get(name), 0.0) or 0.0 for name in names]])
        prediction = np.asarray(model.predict(values))[0]
        if len(prediction) != len(OUTCOME_ORDER):
            return None
        return normalize_outcome_probabilities(dict(zip(OUTCOME_ORDER, prediction.tolist())))
    except Exception:
        return None


def ensemble_outcomes(bayesian: Mapping[str, float], lgbm: Optional[Mapping[str, float]]) -> Tuple[Dict[str, float], str]:
    if not lgbm:
        return dict(bayesian), "Hierarchical Bayesian"
    blended = {key: 0.62 * bayesian.get(key, 0.0) + 0.38 * lgbm.get(key, 0.0) for key in OUTCOME_ORDER}
    return normalize_outcome_probabilities(blended), "Bayesian + multiclass LightGBM ensemble"


# =============================================================================
# MONTE CARLO — PA → OUTCOME → BASE ADVANCEMENT → RUNS/RBI
# =============================================================================


@dataclass
class SimulationResult:
    projection_hrr: float
    projection_hits: float
    projection_runs: float
    projection_rbi: float
    projection_hr: float
    probability_hr: float
    hrr_distribution: Dict[int, float]
    pa_distribution: Dict[int, float]
    p10_hrr: float
    p50_hrr: float
    p90_hrr: float
    simulations: int


def sample_pa(rng: np.random.Generator, pa: PAProjection) -> int:
    values = np.array([3, 4, 5, 6])
    probs = np.array([pa.p3, pa.p4, pa.p5, pa.p6], dtype=float)
    probs /= probs.sum()
    return int(rng.choice(values, p=probs))


def lineup_rbi_environment(slot: Optional[int], implied_runs: float) -> Tuple[float, float, float]:
    # Expected occupancy probabilities before a PA. Middle-order hitters receive
    # more runners on base; leadoff hitters receive fewer but score more often.
    occupancy_base = {1: 0.36, 2: 0.42, 3: 0.49, 4: 0.55, 5: 0.52, 6: 0.46, 7: 0.41, 8: 0.37, 9: 0.34}.get(slot, 0.43)
    run_factor = clamp(implied_runs / 4.35, 0.65, 1.55)
    p_first = clamp(occupancy_base * 0.58 * run_factor, 0.10, 0.45)
    p_second = clamp(occupancy_base * 0.34 * run_factor, 0.06, 0.31)
    p_third = clamp(occupancy_base * 0.21 * run_factor, 0.03, 0.22)
    return p_first, p_second, p_third


def score_after_reaching_probability(outcome: str, slot: Optional[int], implied_runs: float) -> float:
    slot_base = {1: 0.38, 2: 0.40, 3: 0.42, 4: 0.43, 5: 0.39, 6: 0.35, 7: 0.32, 8: 0.29, 9: 0.27}.get(slot, 0.35)
    power = {"BB": 0.86, "HBP": 0.87, "1B": 0.90, "2B": 1.22, "3B": 1.58, "ROE": 0.88}.get(outcome, 1.0)
    return clamp(slot_base * power * (implied_runs / 4.35), 0.08, 0.78)


def simulate_player(
    player: str,
    game_date: str,
    outcomes: Mapping[str, float],
    pa: PAProjection,
    simulations: int,
) -> SimulationResult:
    rng = np.random.default_rng(stable_seed(player, game_date, simulations, round(pa.expected_pa, 3)))
    keys = np.array(OUTCOME_ORDER)
    probabilities = np.array([outcomes.get(k, 0.0) for k in OUTCOME_ORDER], dtype=float)
    probabilities /= probabilities.sum()
    hrr_totals = np.zeros(simulations, dtype=float)
    hit_totals = np.zeros(simulations, dtype=float)
    run_totals = np.zeros(simulations, dtype=float)
    rbi_totals = np.zeros(simulations, dtype=float)
    hr_totals = np.zeros(simulations, dtype=float)
    pa_totals = np.zeros(simulations, dtype=int)
    p_first, p_second, p_third = lineup_rbi_environment(pa.lineup_slot, pa.team_implied_runs)

    for sim in range(simulations):
        n_pa = sample_pa(rng, pa)
        pa_totals[sim] = n_pa
        hits = runs = rbi = hrs = 0
        for _ in range(n_pa):
            outcome = str(rng.choice(keys, p=probabilities))
            on_first = rng.random() < p_first
            on_second = rng.random() < p_second
            on_third = rng.random() < p_third
            if outcome in {"1B", "2B", "3B", "HR"}:
                hits += 1
            if outcome == "HR":
                hrs += 1
                rbi += 1 + int(on_first) + int(on_second) + int(on_third)
                runs += 1
            elif outcome == "3B":
                rbi += int(on_first) + int(on_second) + int(on_third)
                if rng.random() < score_after_reaching_probability(outcome, pa.lineup_slot, pa.team_implied_runs):
                    runs += 1
            elif outcome == "2B":
                rbi += int(on_second) + int(on_third) + int(on_first and rng.random() < 0.47)
                if rng.random() < score_after_reaching_probability(outcome, pa.lineup_slot, pa.team_implied_runs):
                    runs += 1
            elif outcome == "1B":
                rbi += int(on_third) + int(on_second and rng.random() < 0.58)
                if rng.random() < score_after_reaching_probability(outcome, pa.lineup_slot, pa.team_implied_runs):
                    runs += 1
            elif outcome in {"BB", "HBP", "ROE"}:
                # Bases-loaded walk/HBP is approximated through sampled occupancy.
                if on_first and on_second and on_third:
                    rbi += 1
                if rng.random() < score_after_reaching_probability(outcome, pa.lineup_slot, pa.team_implied_runs):
                    runs += 1
            elif outcome == "PRODUCTIVE_OUT":
                if on_third and rng.random() < 0.48:
                    rbi += 1
            elif outcome == "DOUBLE_PLAY":
                # Double play removes a potential scoring chance; no direct HRR.
                pass
        hit_totals[sim] = hits
        run_totals[sim] = runs
        rbi_totals[sim] = rbi
        hr_totals[sim] = hrs
        hrr_totals[sim] = hits + runs + rbi

    unique, counts = np.unique(hrr_totals.astype(int), return_counts=True)
    hrr_dist = {int(k): float(v / simulations) for k, v in zip(unique, counts)}
    pa_unique, pa_counts = np.unique(pa_totals, return_counts=True)
    pa_dist = {int(k): float(v / simulations) for k, v in zip(pa_unique, pa_counts)}
    return SimulationResult(
        projection_hrr=float(hrr_totals.mean()),
        projection_hits=float(hit_totals.mean()),
        projection_runs=float(run_totals.mean()),
        projection_rbi=float(rbi_totals.mean()),
        projection_hr=float(hr_totals.mean()),
        probability_hr=float((hr_totals >= 1).mean()),
        hrr_distribution=hrr_dist,
        pa_distribution=pa_dist,
        p10_hrr=float(np.quantile(hrr_totals, 0.10)),
        p50_hrr=float(np.quantile(hrr_totals, 0.50)),
        p90_hrr=float(np.quantile(hrr_totals, 0.90)),
        simulations=simulations,
    )


def probability_over_from_distribution(distribution: Mapping[int, float], line: float) -> float:
    return sum(prob for value, prob in distribution.items() if value > line)


# =============================================================================
# PROJECTION ROW
# =============================================================================


@dataclass
class ProjectionRecord:
    Date: str
    Player: str
    Player_ID: int
    Team: str
    Opponent: str
    Matchup: str
    Market: str
    Line: float
    Pick: str
    Projection: float
    Edge: float
    Win_Probability: float
    Confidence: float
    Grade: str
    Official_Status: str
    Data_Quality: float
    Projected_PA: float
    P3_PA: float
    P4_PA: float
    P5_PA: float
    P6_PA: float
    PA_vs_Starter: float
    PA_vs_Bullpen: float
    Lineup_Slot: Optional[int]
    Lineup_Confirmed: bool
    Pinch_Hit_Risk: float
    Team_Implied_Runs: float
    Pitcher: str
    Pitcher_Hand: str
    Pitcher_Target_Score: float
    Contact_Allowed_Score: float
    Damage_Allowed_Score: float
    Traffic_Allowed_Score: float
    Platoon_Vulnerability_Score: float
    Arsenal_Matchup_Score: float
    Current_Condition_Score: float
    Starter_Weight: float
    Bullpen_Weight: float
    Bullpen_Vulnerability_Score: float
    Bullpen_Handedness: str
    Hits_Projection: float
    Runs_Projection: float
    RBI_Projection: float
    HR_Projection: float
    HR_Probability: float
    P10_HRR: float
    P50_HRR: float
    P90_HRR: float
    Model_Mode: str
    Notes: str
    Features: Dict[str, Any] = field(default_factory=dict)
    Outcome_Probabilities: Dict[str, float] = field(default_factory=dict)
    Weight_Blend: Dict[str, float] = field(default_factory=dict)
    Component_Projections: Dict[str, Optional[float]] = field(default_factory=dict)

    def flat(self) -> Dict[str, Any]:
        row = asdict(self)
        row["Win Probability %"] = round(self.Win_Probability * 100, 1)
        row["Confidence / 10"] = round(self.Confidence, 1)
        row["Data Quality %"] = round(self.Data_Quality, 1)
        row["Projected PA"] = round(self.Projected_PA, 2)
        row["Pitcher Target Score"] = round(self.Pitcher_Target_Score, 1)
        return row


def data_quality_score(
    baseline: BaselineResult,
    pa: PAProjection,
    pitcher: PitcherVulnerability,
    bullpen: BullpenExposure,
    context: Mapping[str, Any],
    model_mode: str,
) -> float:
    score = baseline.data_quality * 0.58
    score += 10 if pa.lineup_confirmed else 4 if pa.everyday_starter else 0
    score += 7 if pitcher.pitcher_id else 0
    score += 5 if pitcher.arsenal_matchup != 50 else 0
    score += 5 if bullpen.bullpen_xwoba_allowed is not None else 2
    score += 4 if model_mode != "Hierarchical Bayesian" else 0
    if context.get("opponent") in (None, "", "—"):
        score -= 12
    return clamp(score, 30, 98)


def official_gate(
    market: str,
    line: float,
    pick: str,
    projection: float,
    win_prob: float,
    edge: float,
    data_quality: float,
    pa: PAProjection,
    context: Mapping[str, Any],
) -> Tuple[str, List[str]]:
    fails: List[str] = []
    if context.get("opponent") in (None, "", "—"):
        fails.append("No MLB game/opponent matched")
    if not pa.lineup_confirmed and not pa.everyday_starter:
        fails.append("Not confirmed and recent everyday-start rate is weak")
    if pa.expected_pa < 3.72:
        fails.append("Projected PA below 3.72")
    if pa.pinch_hit_risk > 0.28:
        fails.append("Pinch-hit/platoon risk too high")
    if data_quality < 65:
        fails.append("Data quality below 65")
    if market == "HRR" and abs(edge) < 0.32:
        fails.append("H+R+RBI edge below 0.32")
    if market == "HR" and abs(edge) < 0.055:
        fails.append("Home-run probability edge is thin")
    if win_prob < 0.565:
        fails.append("Win probability below 56.5%")
    if fails:
        return "TRACK / PASS", fails
    if win_prob >= 0.60 and data_quality >= 75:
        return "OFFICIAL", []
    return "SELECTIVE", []


def project_line(
    line_row: Mapping[str, Any],
    bundle: DataBundle,
    game_date: str,
    simulations: int,
    deep_enabled: bool,
) -> Optional[ProjectionRecord]:
    player = str(line_row.get("player") or "").strip()
    market = str(line_row.get("market") or "").upper()
    line = safe_float(line_row.get("line"), None)
    if not player or market not in {"HRR", "HR"} or line is None:
        return None
    identity = player_identity(player, bundle)
    if not identity.get("id"):
        return None
    context = matchup_context(identity, bundle, game_date)
    batter = build_batter_features(identity, context, bundle)
    baseline = true_talent_baseline(batter, datetime.fromisoformat(game_date).date())
    pa = project_plate_appearances(context, bundle)
    pitcher = build_pitcher_vulnerability(context, batter, bundle, datetime.fromisoformat(game_date).date(), deep_enabled)
    bullpen = build_bullpen_exposure(context, pa, pitcher, bundle)
    bayesian = adjust_outcomes_for_matchup(baseline, pitcher, bullpen, batter)
    feature_row = {
        **batter.display_dict(),
        "projected_pa": pa.expected_pa,
        "team_implied_runs": pa.team_implied_runs,
        "pitcher_target_score": pitcher.target_score,
        "bullpen_vulnerability": bullpen.bullpen_vulnerability_score,
        "lineup_slot": pa.lineup_slot or 6,
    }
    lgbm = optional_lightgbm_probabilities(feature_row)
    outcomes, model_mode = ensemble_outcomes(bayesian, lgbm)
    simulation = simulate_player(player, game_date, outcomes, pa, simulations)

    if market == "HRR":
        projection = simulation.projection_hrr
        p_over = probability_over_from_distribution(simulation.hrr_distribution, line)
    else:
        projection = simulation.projection_hr
        p_over = simulation.probability_hr if line == 0.5 else probability_over_from_distribution(
            {k: v for k, v in simulation.hrr_distribution.items()}, line
        )
        # For unusual HR lines, derive Poisson tail from projected HR count.
        if line != 0.5:
            threshold = int(math.floor(line)) + 1
            lam = max(1e-6, simulation.projection_hr)
            p_over = 1.0 - sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(threshold))
    pick = "OVER" if p_over >= 0.50 else "UNDER"
    win_prob = max(p_over, 1.0 - p_over)
    edge = projection - line
    if pick == "UNDER":
        edge = line - projection
    quality = data_quality_score(baseline, pa, pitcher, bullpen, context, model_mode)
    confidence = clamp(5.0 + (win_prob - 0.50) * 24.0 + (quality - 65.0) / 22.0, 1.0, 10.0)
    status, gate_fails = official_gate(market, line, pick, projection, win_prob, edge, quality, pa, context)
    grade = implied_grade(win_prob, edge, quality, market)
    component_projection = {
        key: (value * pa.expected_pa if value is not None else None) for key, value in baseline.component_hrr_per_pa.items()
    }
    notes = baseline.notes + pa.notes + pitcher.notes + bullpen.notes
    if gate_fails:
        notes.append("Official gate: " + "; ".join(gate_fails))
    matchup = f"{context.get('team','—')} @ {context.get('opponent','—')}" if not context.get("home") else f"{context.get('opponent','—')} @ {context.get('team','—')}"
    return ProjectionRecord(
        Date=game_date,
        Player=str(identity.get("name") or player),
        Player_ID=safe_int(identity.get("id"), 0) or 0,
        Team=str(context.get("team") or identity.get("team") or ""),
        Opponent=str(context.get("opponent") or ""),
        Matchup=matchup,
        Market=market,
        Line=line,
        Pick=pick,
        Projection=round(projection, 3),
        Edge=round(edge, 3),
        Win_Probability=win_prob,
        Confidence=confidence,
        Grade=grade,
        Official_Status=status,
        Data_Quality=quality,
        Projected_PA=pa.expected_pa,
        P3_PA=pa.p3,
        P4_PA=pa.p4,
        P5_PA=pa.p5,
        P6_PA=pa.p6,
        PA_vs_Starter=pa.pa_vs_starter,
        PA_vs_Bullpen=pa.pa_vs_bullpen,
        Lineup_Slot=pa.lineup_slot,
        Lineup_Confirmed=pa.lineup_confirmed,
        Pinch_Hit_Risk=pa.pinch_hit_risk,
        Team_Implied_Runs=pa.team_implied_runs,
        Pitcher=pitcher.pitcher,
        Pitcher_Hand=pitcher.hand,
        Pitcher_Target_Score=pitcher.target_score,
        Contact_Allowed_Score=pitcher.contact_allowed,
        Damage_Allowed_Score=pitcher.damage_allowed,
        Traffic_Allowed_Score=pitcher.traffic_allowed,
        Platoon_Vulnerability_Score=pitcher.platoon_vulnerability,
        Arsenal_Matchup_Score=pitcher.arsenal_matchup,
        Current_Condition_Score=pitcher.current_condition,
        Starter_Weight=bullpen.starter_weight,
        Bullpen_Weight=bullpen.bullpen_weight,
        Bullpen_Vulnerability_Score=bullpen.bullpen_vulnerability_score,
        Bullpen_Handedness=bullpen.bullpen_handedness,
        Hits_Projection=simulation.projection_hits,
        Runs_Projection=simulation.projection_runs,
        RBI_Projection=simulation.projection_rbi,
        HR_Projection=simulation.projection_hr,
        HR_Probability=simulation.probability_hr,
        P10_HRR=simulation.p10_hrr,
        P50_HRR=simulation.p50_hrr,
        P90_HRR=simulation.p90_hrr,
        Model_Mode=model_mode,
        Notes=" | ".join(notes),
        Features=batter.display_dict(),
        Outcome_Probabilities=outcomes,
        Weight_Blend=baseline.weights,
        Component_Projections=component_projection,
    )


@st.cache_data(ttl=240, show_spinner=False)
def build_projection_board(
    lines_json: str,
    game_date: str,
    simulations: int,
    deep_matchups: int,
) -> List[Dict[str, Any]]:
    lines = json.loads(lines_json)
    bundle = build_data_bundle(game_date)
    rows: List[Optional[ProjectionRecord]] = [None] * len(lines)

    def run(index_and_line: Tuple[int, Mapping[str, Any]]) -> Tuple[int, Optional[ProjectionRecord]]:
        idx, line = index_and_line
        return idx, project_line(line, bundle, game_date, simulations, idx < deep_matchups)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_BOARD_WORKERS) as executor:
        futures = [executor.submit(run, item) for item in enumerate(lines)]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, row = future.result()
                rows[idx] = row
            except Exception:
                continue
    return [row.flat() for row in rows if row is not None]


# =============================================================================
# GRADING / LEARNING
# =============================================================================


def save_snapshot(board: pd.DataFrame) -> int:
    if board.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for record in board.to_dict("records"):
        item = dict(record)
        item["Captured_At"] = now
        components = item.get("Component_Projections") or {}
        for key in ("current", "previous", "career", "l30", "l15", "split", "home_away", "minors"):
            item[f"Comp_{key}"] = components.get(key) if isinstance(components, dict) else None
        rows.append(item)
    return append_json_rows(SNAPSHOT_FILE, rows, ("Date", "Player", "Market", "Line", "Pick"))


@st.cache_data(ttl=600, show_spinner=False)
def exact_day_hitting(player_id: int, game_date: str) -> HittingSample:
    rows = individual_stats(player_id, "hitting", "byDateRange", start_date=game_date, end_date=game_date)
    return sample_from_split(first_split(rows), "Game")


def auto_grade_snapshots() -> Tuple[int, int]:
    snapshots = read_json(SNAPSHOT_FILE, [])
    history = read_json(GRADE_FILE, [])
    if not isinstance(snapshots, list):
        snapshots = []
    if not isinstance(history, list):
        history = []
    existing = {(r.get("Date"), r.get("Player"), r.get("Market"), r.get("Line"), r.get("Pick")) for r in history if isinstance(r, dict)}
    graded = 0
    skipped = 0
    for row in snapshots:
        if not isinstance(row, dict):
            continue
        key = (row.get("Date"), row.get("Player"), row.get("Market"), row.get("Line"), row.get("Pick"))
        if key in existing:
            continue
        try:
            event_date = datetime.fromisoformat(str(row.get("Date"))).date()
        except Exception:
            skipped += 1
            continue
        if event_date >= today_local():
            skipped += 1
            continue
        sample = exact_day_hitting(safe_int(row.get("Player_ID"), 0) or 0, event_date.isoformat())
        if sample.pa <= 0:
            skipped += 1
            continue
        market = str(row.get("Market"))
        actual = sample.hits + sample.runs + sample.rbi if market == "HRR" else sample.homers
        line = safe_float(row.get("Line"), 0.0) or 0.0
        pick = str(row.get("Pick") or "").upper()
        if actual == line:
            result = "PUSH"
        elif (pick == "OVER" and actual > line) or (pick == "UNDER" and actual < line):
            result = "WIN"
        else:
            result = "LOSS"
        graded_row = dict(row)
        graded_row.update(
            {
                "Actual": actual,
                "Actual_Hits": sample.hits,
                "Actual_Runs": sample.runs,
                "Actual_RBI": sample.rbi,
                "Actual_HR": sample.homers,
                "Actual_PA": sample.pa,
                "Result": result,
                "Graded_At": datetime.now(timezone.utc).isoformat(),
            }
        )
        history.append(graded_row)
        existing.add(key)
        graded += 1
    write_json(GRADE_FILE, history)
    return graded, skipped


# =============================================================================
# UI RENDERING
# =============================================================================


def compact_board(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Player",
        "Matchup",
        "Market",
        "Line",
        "Pick",
        "Projection",
        "Edge",
        "Win Probability %",
        "Grade",
        "Official_Status",
        "Projected PA",
        "Lineup_Slot",
        "Lineup_Confirmed",
        "Pitcher",
        "Pitcher Target Score",
        "Data Quality %",
        "Confidence / 10",
    ]
    return frame[[c for c in columns if c in frame.columns]].copy()


def market_board(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    base_columns = [
        "Player",
        "Matchup",
        "Line",
        "Pick",
        "Projection",
        "Edge",
        "Win Probability %",
        "Grade",
        "Official_Status",
    ]
    if market == "HRR":
        projection_columns = [
            "Hits_Projection",
            "Runs_Projection",
            "RBI_Projection",
            "HR_Projection",
            "P10_HRR",
            "P50_HRR",
            "P90_HRR",
        ]
    else:
        projection_columns = [
            "HR_Projection",
            "HR_Probability",
            "Hits_Projection",
            "Runs_Projection",
            "RBI_Projection",
        ]
    context_columns = [
        "Projected PA",
        "Lineup_Slot",
        "Lineup_Confirmed",
        "Pitcher",
        "Pitcher Target Score",
        "Data Quality %",
        "Confidence / 10",
    ]
    out = frame[[c for c in base_columns + projection_columns + context_columns if c in frame.columns]].copy()
    rename = {
        "Hits_Projection": "Proj Hits",
        "Runs_Projection": "Proj Runs",
        "RBI_Projection": "Proj RBI",
        "HR_Projection": "Proj HR",
        "HR_Probability": "HR Prob",
        "P10_HRR": "H+R+RBI P10",
        "P50_HRR": "H+R+RBI Median",
        "P90_HRR": "H+R+RBI P90",
    }
    out = out.rename(columns=rename)
    for col in ("HR Prob",):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").mul(100).round(1).astype("Float64").astype(str) + "%"
            out[col] = out[col].replace("<NA>%", "—")
    numeric_cols = [
        "Projection",
        "Edge",
        "Proj Hits",
        "Proj Runs",
        "Proj RBI",
        "Proj HR",
        "H+R+RBI P10",
        "H+R+RBI Median",
        "H+R+RBI P90",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def render_player_card(row: Mapping[str, Any]) -> None:
    pick = str(row.get("Pick", "PASS"))
    pick_class = "ow-over" if pick == "OVER" else "ow-under" if pick == "UNDER" else "ow-pass"
    status = str(row.get("Official_Status", "TRACK"))
    status_class = "ow-good" if status == "OFFICIAL" else "ow-warn" if status == "SELECTIVE" else "ow-bad"
    st.markdown(
        f"""
<div class="ow-card">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
    <div><div class="ow-player">{row.get('Player','')}</div><div class="ow-muted">{row.get('Matchup','—')} · vs {row.get('Pitcher','TBD')} ({row.get('Pitcher_Hand','—')})</div></div>
    <div><span class="ow-badge {status_class}">{status}</span><span class="ow-badge">Grade {row.get('Grade','PASS')}</span></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:14px;">
    <div><div class="ow-muted">LINE</div><div style="font-size:27px;font-weight:900;">{row.get('Line','—')}</div></div>
    <div><div class="ow-muted">PICK</div><div class="{pick_class}" style="font-size:27px;">{pick}</div></div>
    <div><div class="ow-muted">PROJECTION</div><div style="font-size:27px;font-weight:900;">{format_value(row.get('Projection'),2)}</div></div>
    <div><div class="ow-muted">WIN PROB</div><div style="font-size:27px;font-weight:900;">{format_value(row.get('Win Probability %'),1)}%</div></div>
    <div><div class="ow-muted">PA</div><div style="font-size:27px;font-weight:900;">{format_value(row.get('Projected PA'),2)}</div></div>
    <div><div class="ow-muted">TARGET SCORE</div><div style="font-size:27px;font-weight:900;">{format_value(row.get('Pitcher Target Score'),1)}</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("Full model card", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hits", format_value(row.get("Hits_Projection"), 2))
        c2.metric("Runs", format_value(row.get("Runs_Projection"), 2))
        c3.metric("RBI", format_value(row.get("RBI_Projection"), 2))
        c4.metric("HR", f"{format_value(row.get('HR_Projection'),2)} · {format_value((safe_float(row.get('HR_Probability'), 0) or 0)*100,1)}%")
        st.markdown("#### Plate-appearance distribution")
        pa_df = pd.DataFrame(
            {
                "PA": [3, 4, 5, 6],
                "Probability %": [
                    (safe_float(row.get("P3_PA"), 0) or 0) * 100,
                    (safe_float(row.get("P4_PA"), 0) or 0) * 100,
                    (safe_float(row.get("P5_PA"), 0) or 0) * 100,
                    (safe_float(row.get("P6_PA"), 0) or 0) * 100,
                ],
            }
        )
        st.dataframe(pa_df, hide_index=True, use_container_width=True)
        st.markdown("#### Pitcher vulnerability components")
        vuln = pd.DataFrame(
            {
                "Component": ["Contact allowed", "Damage allowed", "Traffic allowed", "Platoon", "Arsenal", "Current condition"],
                "Score": [
                    row.get("Contact_Allowed_Score"),
                    row.get("Damage_Allowed_Score"),
                    row.get("Traffic_Allowed_Score"),
                    row.get("Platoon_Vulnerability_Score"),
                    row.get("Arsenal_Matchup_Score"),
                    row.get("Current_Condition_Score"),
                ],
            }
        )
        st.dataframe(vuln, hide_index=True, use_container_width=True)
        st.markdown("#### Batter feature engine")
        feature_dict = row.get("Features") if isinstance(row.get("Features"), dict) else {}
        if feature_dict:
            feature_frame = pd.DataFrame([feature_dict]).T.reset_index()
            feature_frame.columns = ["Feature", "Value"]
            st.dataframe(feature_frame, hide_index=True, use_container_width=True)
        st.markdown("#### PA outcome probabilities")
        outcome_dict = row.get("Outcome_Probabilities") if isinstance(row.get("Outcome_Probabilities"), dict) else {}
        if outcome_dict:
            outcome_frame = pd.DataFrame(
                {"Outcome": list(outcome_dict.keys()), "Probability %": [round(v * 100, 2) for v in outcome_dict.values()]}
            )
            st.dataframe(outcome_frame, hide_index=True, use_container_width=True)
        st.markdown("#### H+R+RBI simulation range")
        st.info(f"P10 {row.get('P10_HRR','—')} · Median {row.get('P50_HRR','—')} · P90 {row.get('P90_HRR','—')}")
        st.caption(str(row.get("Notes", "")))


def render_market_tab(frame: pd.DataFrame, market: str) -> None:
    data = frame[frame["Market"] == market].copy() if not frame.empty else pd.DataFrame()
    label = "Hits + Runs + RBIs" if market == "HRR" else "Home Runs"
    st.subheader(label)
    if data.empty:
        st.warning(f"No active Underdog {label} rows were matched. The app will not create fake lines.")
        return
    data = data.sort_values(["Official_Status", "Win Probability %", "Data Quality %"], ascending=[True, False, False])
    st.dataframe(market_board(data, market), hide_index=True, use_container_width=True)
    st.markdown("### Player cards")
    for row in data.head(35).to_dict("records"):
        render_player_card(row)


def render_learning_tab() -> None:
    history = read_json(GRADE_FILE, [])
    frame = pd.DataFrame(history) if isinstance(history, list) else pd.DataFrame()
    weights = load_learned_weights()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graded rows", len(frame))
    if not frame.empty and "Result" in frame:
        decisions = frame[frame["Result"].isin(["WIN", "LOSS"])]
        wr = (decisions["Result"] == "WIN").mean() * 100 if len(decisions) else 0
        c2.metric("Win rate", f"{wr:.1f}%")
    else:
        c2.metric("Win rate", "—")
    c3.metric("Weight mode", "LEARNED" if weights["learned"] else "PRIORS")
    c4.metric("Training rows", weights["rows"])
    st.markdown("#### Current true-talent blend")
    weight_frame = pd.DataFrame({"Sample": list(weights["weights"].keys()), "Weight %": [v * 100 for v in weights["weights"].values()]})
    st.dataframe(weight_frame, hide_index=True, use_container_width=True)
    if st.button("Learn true-talent weights from graded history", type="primary", use_container_width=True):
        try:
            result = learn_weights_from_history(frame)
            st.success(f"Weights learned from {result['rows']} rows. Validation MSE: {result['validation_mse']}")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if frame.empty:
        st.info("Save official snapshots and grade completed games to activate backtested weight learning.")
    else:
        show = [c for c in ["Date", "Player", "Market", "Line", "Pick", "Projection", "Actual", "Result", "Win Probability %", "Grade", "Official_Status"] if c in frame.columns]
        st.dataframe(frame[show].tail(250), hide_index=True, use_container_width=True)


def render_settings(debug: Mapping[str, Any], board: pd.DataFrame) -> None:
    st.subheader("Batter-only status")
    st.write(
        {
            "App version": APP_VERSION,
            "Build": APP_BUILD,
            "Visible markets": ["Hits + Runs + RBIs", "Home Runs"],
            "Pitcher prop tabs": "REMOVED",
            "Pitching outs": "REMOVED",
            "Strikeout props": "REMOVED",
            "Moneyline": "REMOVED",
            "Opposing pitcher usage": "Matchup input only",
            "Underdog endpoint": debug.get("endpoint"),
            "Underdog parser status": debug.get("status"),
            "Underdog parsed rows": debug.get("parsed_rows"),
            "Projection rows": len(board),
            "LightGBM": "Loaded when BATTER_LGBM_MODEL points to a trained multiclass artifact; otherwise Bayesian-only",
        }
    )
    with st.expander("Underdog parser diagnostics"):
        st.json(debug)
    if st.button("Clear all app caches", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared.")


# =============================================================================
# MAIN APP
# =============================================================================


st.markdown(
    f"""
<div class="ow-hero">
  <div class="ow-title">⚾ ONE WAY PICKZ — BATTER PROJECTIONS</div>
  <div class="ow-sub">True talent → projected plate appearances → PA outcomes → base advancement → Hits, Runs, RBI and Home Runs</div>
  <div class="ow-sub">{APP_VERSION} · Underdog batter lines only</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Board controls")
    selected_date = st.date_input("Slate date", value=today_local())
    simulations = st.select_slider("Monte Carlo simulations", options=[3000, 5000, 7000, 10000, 15000], value=DEFAULT_SIMULATIONS if DEFAULT_SIMULATIONS in [3000, 5000, 7000, 10000, 15000] else 7000)
    deep_matchups = st.slider("Deep pitch-type matchups", min_value=0, max_value=30, value=12, help="Pitch-level Savant matching is slower. Other rows use a neutral arsenal score and are clearly marked.")
    refresh = st.button("🔄 Refresh Underdog + MLB data", type="primary", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.cache_resource.clear()
    st.caption("L5 streaks are display-only and are not used to inflate projections.")

with st.spinner("Pulling active Underdog batter lines..."):
    lines, ud_debug = fetch_underdog_batter_lines()
record_line_history(lines)

if not lines:
    st.error("No active Underdog H+R+RBI or Home Run lines matched. No fake lines were created.")
    with st.expander("Underdog diagnostics", expanded=True):
        st.json(ud_debug)
    st.stop()

with st.spinner("Building true-talent, PA, pitcher/bullpen and Monte Carlo projections..."):
    board_rows = build_projection_board(
        json.dumps(lines, sort_keys=True),
        selected_date.isoformat(),
        int(simulations),
        int(deep_matchups),
    )
board = pd.DataFrame(board_rows)

if board.empty:
    st.error("Underdog lines were found, but no rows could be matched to MLB hitters. Open Settings for parser diagnostics.")

# Top metrics.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Underdog lines", len(lines))
c2.metric("Projected hitters", len(board))
c3.metric("Official", int((board.get("Official_Status") == "OFFICIAL").sum()) if not board.empty else 0)
c4.metric("Confirmed lineups", int(pd.to_numeric(board.get("Lineup_Confirmed"), errors="coerce").fillna(0).sum()) if not board.empty else 0)
c5.metric("Avg data quality", f"{pd.to_numeric(board.get('Data Quality %'), errors='coerce').mean():.1f}%" if not board.empty else "—")

if not board.empty:
    official = board[board["Official_Status"].isin(["OFFICIAL", "SELECTIVE"])].sort_values("Win Probability %", ascending=False)
else:
    official = pd.DataFrame()

(top_tab, hrr_tab, hr_tab, grade_tab, learning_tab, settings_tab) = st.tabs(
    ["🔥 BATTER UPSIDE", "1️⃣ H+R+RBI", "2️⃣ HOME RUNS", "✅ SAVE / GRADE", "🧠 LEARNING", "⚙️ SETTINGS"]
)

with top_tab:
    st.subheader("Best batter-only opportunities")
    if official.empty:
        st.info("No rows currently clear the official/selective gate. Use the market tabs to review track-only rows.")
    else:
        st.dataframe(compact_board(official), hide_index=True, use_container_width=True)
        for row in official.head(20).to_dict("records"):
            render_player_card(row)

with hrr_tab:
    render_market_tab(board, "HRR")

with hr_tab:
    render_market_tab(board, "HR")

with grade_tab:
    st.subheader("Persistent batter snapshots and automatic grading")
    a, b = st.columns(2)
    with a:
        if st.button("Save current board snapshot", type="primary", use_container_width=True):
            added = save_snapshot(board)
            st.success(f"Saved/updated the batter snapshot. {added} new rows added.")
    with b:
        if st.button("Grade completed saved games", use_container_width=True):
            graded, skipped = auto_grade_snapshots()
            st.success(f"Graded {graded} rows. {skipped} rows were not final/available yet.")
            st.cache_data.clear()
    snapshots = read_json(SNAPSHOT_FILE, [])
    grades = read_json(GRADE_FILE, [])
    s1, s2 = st.columns(2)
    s1.metric("Saved rows", len(snapshots) if isinstance(snapshots, list) else 0)
    s2.metric("Graded rows", len(grades) if isinstance(grades, list) else 0)
    if isinstance(snapshots, list) and snapshots:
        snap_df = pd.DataFrame(snapshots)
        show = [c for c in ["Date", "Player", "Market", "Line", "Pick", "Projection", "Win Probability %", "Official_Status", "Captured_At"] if c in snap_df.columns]
        st.dataframe(snap_df[show].tail(250), hide_index=True, use_container_width=True)

with learning_tab:
    render_learning_tab()

with settings_tab:
    render_settings(ud_debug, board)
