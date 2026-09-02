"""Publish already-built MLB batter boards to the member platform.

This module is deliberately downstream of every projection engine. It reads only
frozen Streamlit session state and persisted official/result logs, sanitizes those
rows to the public member contract, and sends them to the platform ingest route.
It never imports or calls a projection builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


SYNC_STATE_KEY = "ow_member_board_sync_fingerprints_v1"
MAX_ITEMS_PER_BOARD = 500
MARKET_LABELS = {
    "upside": "Batter Upside",
    "games": "Games",
    "hrr": "H+R+RBI",
    "home-runs": "Home Runs",
    "fantasy": "Batter Fantasy",
    "official": "Official Plays",
    "results": "Results",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_HTTP_OPENER = build_opener(_NoRedirect)


def _open_request(request: Request, timeout: float):
    return _HTTP_OPENER.open(request, timeout=timeout)


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _scalar(value: Any) -> Any:
    """Convert pandas/numpy/datetime values to bounded JSON-safe scalars."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "—", "-"}:
            return None
        return text[:500]
    return None


def _lookup(row: Mapping[str, Any], *candidates: str) -> Any:
    normalized = {_normalized_key(key): value for key, value in row.items()}
    for candidate in candidates:
        value = _scalar(normalized.get(_normalized_key(candidate)))
        if value is not None:
            return value
    return None


def _number_or_text(value: Any) -> Any:
    value = _scalar(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value).strip()
    numeric = text.replace(",", "")
    if numeric.endswith("%"):
        numeric = numeric[:-1].strip()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", numeric):
        parsed = float(numeric)
        return int(parsed) if parsed.is_integer() else round(parsed, 4)
    return text[:120]


def _flags(row: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    normalized = {_normalized_key(key): value for key, value in row.items()}
    for candidate in ("Flags", "Risk Flags", "Status Flags", "Warnings", "Warning"):
        raw = normalized.get(_normalized_key(candidate))
        if isinstance(raw, (list, tuple, set)):
            values.extend(raw)
        elif raw is not None:
            values.extend(re.split(r"\s*[|;\n]+\s*", str(raw)))
    clean: list[str] = []
    for value in values:
        text = _scalar(value)
        if isinstance(text, str) and text not in clean:
            clean.append(text[:160])
        if len(clean) >= 12:
            break
    return clean


def _records(source: Any) -> list[dict[str, Any]] | None:
    """Return records; None means the source was absent, [] means present/empty."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        nested = source.get("rows")
        if isinstance(nested, list):
            return [dict(row) for row in nested if isinstance(row, Mapping)]
        return [dict(source)]
    if isinstance(source, (list, tuple)):
        return [dict(row) for row in source if isinstance(row, Mapping)]
    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
            if isinstance(records, list):
                return [dict(row) for row in records if isinstance(row, Mapping)]
        except TypeError:
            pass
        except Exception:
            return []
    return []


def _read_json_rows(path_value: Any) -> list[dict[str, Any]] | None:
    if not path_value:
        return None
    try:
        path = Path(str(path_value))
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _records(payload)
    except Exception:
        return None


def _member_item(
    row: Mapping[str, Any],
    market: str,
    rank: int,
    source_state: str,
    fallback_slate_date: str | None = None,
) -> dict[str, Any] | None:
    player = _lookup(row, "Player", "UD Player", "player", "Batter", "Name")
    if not isinstance(player, str) or not player.strip():
        return None

    underlying_market = _lookup(row, "Best Market", "Market", "market", "Prop Type")
    fixed_label = MARKET_LABELS.get(market, market)
    market_label = underlying_market if market in {"upside", "games", "official", "results"} else fixed_label
    slate_date = _lookup(
        row,
        "Snapshot Date",
        "Official Game Date",
        "Slate Date",
        "Game Date",
        "date",
    ) or fallback_slate_date
    if isinstance(slate_date, str) and re.match(r"^\d{4}-\d{2}-\d{2}", slate_date):
        slate_date = slate_date[:10]

    item: dict[str, Any] = {
        "market": market,
        "marketLabel": market_label or fixed_label,
        "rank": _number_or_text(_lookup(row, "Rank", "Overall Rank", "Board Rank")) or rank,
        "player": player.strip()[:120],
        "sourceState": source_state,
    }

    text_fields = {
        "snapshotId": ("snapshot_id", "Snapshot ID", "pick_id", "Pick ID"),
        "team": ("Team", "Raw Log Team"),
        "opponent": ("Opponent", "Today Opponent", "Opp"),
        "gameTime": ("Game Time", "game_time", "Start Time"),
        "venue": ("Venue", "Ballpark", "Park"),
        "oppPitcher": ("Opp Pitcher", "Opposing Pitcher", "Probable Pitcher"),
        "pitcherHand": ("Pitcher Hand", "Opp Pitcher Hand", "Pitcher Throws"),
        "matchupText": ("Matchup", "Game Matchup", "Team Matchup"),
        "pick": ("Best Pick", "Pick", "Model Direction Alt", "Model Direction", "HR Pick", "HR Signal", "Pick Side", "Side"),
        "status": ("Grade Status", "Status", "Confirmed Lineup Status", "Game Status"),
        "officialStatus": ("Official Status", "Official Play Filter", "Official Filter", "Play Status"),
        "lineupStatus": ("Lineup Status", "Confirmed Lineup Status", "Pitcher Confirmation Note"),
        "batterHand": ("Batter Hand", "Bats", "Bat Side"),
        "crossCheck": ("Cross Check", "FS Cross Check", "Cross-Fit", "Cross Fits", "BFS Cross"),
        "teamRunsSource": ("Team Implied Runs Source", "Team Runs Source", "Run Source"),
        "gameLabel": ("Game V3 Label", "High Scoring Game Label", "Game Environment Label"),
        "blowoutLabel": ("Blowout Risk Display Label", "Blowout Risk Label", "Blowout Label"),
        "pitcherProfile": ("Pitcher HR Profile V3", "Pitcher Profile", "Pitcher Weakness Label", "Pitcher Run Profile"),
        "pitcherSplit": ("Pitcher Split vs Batter Hand", "Pitcher Split", "Split Label"),
        "pitcherConfirmed": ("Pitcher Confirmed", "Pitcher Confirmation Note"),
        "batterSide": ("HR Batter Side V3", "Batter Side", "Bats", "Batter Hand"),
        "windLabel": ("HR Wind Pull Label V3", "Wind Carry Label", "Wind Label"),
        "roofStatus": ("HR Roof Status V3", "Roof Status", "Roof"),
        "result": ("graded_result", "Result", "Grade Result"),
    }
    number_fields = {
        "gamePk": ("Game PK", "game_pk", "GamePk"),
        "playerId": ("Player ID", "player_id", "MLB Player ID"),
        "line": ("Best Line", "Line", "HR Line", "FS Line", "Fantasy Line", "UD Line"),
        "projection": ("Best Projection", "Projection", "HR Projection", "FS Projection", "Fantasy Projection", "Projected"),
        "edge": ("Best Edge", "Edge", "Projection Edge", "HR Edge", "FS Edge"),
        "winProbability": ("Best Win/Hit %", "Best Win", "Win Probability %", "Model Win Probability %", "Over Probability %", "HR Probability %", "Hit Probability %"),
        "confidence": ("Confidence", "Overall Rating", "Likely Score", "Rank Score", "Sync Score", "HR Score"),
        "likelyScore": ("Likely Score", "Model Win Probability %", "Sync Score", "HR Score"),
        "overallRating": ("Overall Rating", "Rank Score", "HR Composite Score V3", "HR Power Score V2"),
        "syncScore": ("Sync Score", "Sync"),
        "p20": ("P20", "FS P20", "Projection P20"),
        "p80": ("P80", "FS P80", "Projection P80"),
        "expectedPa": ("Expected PA", "Projected PA", "Exp PA"),
        "lineupSlot": ("Lineup Slot", "Batting Order", "Batting Slot"),
        "last3": ("Last 3", "L3", "Recent 3", "L3 Clear %", "Last 3 %", "Last 3 HR Rate %"),
        "last5": ("Last 5", "L5", "Recent 5", "L5 Clear %", "Last 5 %", "Last 5 HR Rate %"),
        "skill": ("Skill", "Skill Score", "SKILL", "Batter Quality Factor", "Savant Factor HRR", "HR Power Score V2"),
        "matchup": ("Matchup Score", "MATCH", "Match Score", "Pitcher Run/Contact Score", "Pitcher Allows Hits Score", "Pitcher Power Damage Score", "Pitcher Contact/Leash Score", "Key Matchup Stats Factor"),
        "form": ("Form", "Form Score", "FORM"),
        "contact": ("Contact", "Contact Score", "Batter Pitch Contact%"),
        "dataConfidence": ("Data Confidence", "Data Coverage %", "Final Data Quality Score", "Data Score"),
        "readiness": ("Verification Readiness %", "READY%", "Data Readiness", "Readiness"),
        "teamRuns": ("Team Runs V3", "Team Implied Runs", "Projected Team Runs", "Team Run Env", "ImpR"),
        "gameScore": ("Game V3 Score", "High Scoring Game Score", "Game Environment Score", "Game Score"),
        "gameTotal": ("Game Total V3", "Projected Game Total", "Game Total"),
        "blowoutScore": ("Blowout V3 Score", "Blowout Risk Score", "Blowout Run Score", "Blowout Score"),
        "pitcherEra": ("Pitcher ERA", "Opp Pitcher ERA", "Starter ERA"),
        "pitcherWhip": ("Pitcher WHIP", "Pitcher Recent WHIP", "Opp Pitcher WHIP"),
        "pitcherBaa": ("Pitcher Split BAA", "Pitcher BAA", "Opp Pitcher BAA"),
        "pitcherKPercent": ("Pitcher Split K%", "Pitcher K%", "Pitcher Recent K%", "Opp Pitcher K%"),
        "pitcherHr9": ("Pitcher Recent HR/9", "Pitcher HR9", "Pitcher HR/9", "Opp Pitcher HR/9"),
        "pitcherVulnerabilityScore": ("Pitcher HR Vulnerability Score V3", "Pitcher Vulnerability Score", "Pitcher Damage Score"),
        "hrStadiumScore": ("HR Stadium Score V3", "HR Stadium Score", "Park Score"),
        "hrParkFactor": ("HR Park Current 2026 V3", "HR Park Index V3", "HR Park Factor", "Park Factor"),
        "windMph": ("Weather Wind MPH", "Wind MPH"),
        "tempF": ("Weather Temp F", "Temp F", "Temperature F"),
        "humidity": ("Weather Humidity %", "Humidity %"),
        "actual": ("Actual", "Actual H+R+RBI", "Actual HR", "Actual FPTS"),
    }

    for output_key, aliases in text_fields.items():
        value = _lookup(row, *aliases)
        if value is not None:
            item[output_key] = str(value)[:240]
    for output_key, aliases in number_fields.items():
        value = _number_or_text(_lookup(row, *aliases))
        if value is not None:
            item[output_key] = value
    if slate_date:
        item["slateDate"] = str(slate_date)[:40]
    row_flags = _flags(row)
    if row_flags:
        item["flags"] = row_flags

    supplied_id = _lookup(row, "id", "pick_id", "Pick ID", "snapshot_id", "Snapshot ID")
    if supplied_id is None:
        identity = "|".join(
            str(item.get(key, ""))
            for key in ("slateDate", "gamePk", "playerId", "player", "market", "line", "pick")
        )
        supplied_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    item["id"] = str(supplied_id)[:180]
    return item


def _payload(
    market: str,
    rows: Iterable[Mapping[str, Any]],
    generated_at: str,
    refresh_generation: int,
    source_version: str,
    source_state: str,
) -> dict[str, Any]:
    source_rows = list(rows)[:MAX_ITEMS_PER_BOARD]
    fallback_date = None
    for row in source_rows:
        value = _lookup(row, "Snapshot Date", "Official Game Date", "Slate Date", "Game Date", "date")
        if value:
            fallback_date = str(value)[:10]
            break
    items = [
        item
        for index, row in enumerate(source_rows, start=1)
        if (item := _member_item(row, market, index, source_state, fallback_date)) is not None
    ]
    slate_date = next((str(item.get("slateDate")) for item in items if item.get("slateDate")), None)
    return {
        "market": market,
        "generatedAt": generated_at,
        "refreshGeneration": refresh_generation,
        "slateDate": slate_date,
        "sourceVersion": source_version,
        "items": items,
    }


def _post_board(endpoint: str, token: str, payload: Mapping[str, Any], site_bypass_token: str = "") -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "one-way-pickz-member-board-sync/1.0",
        }
        if site_bypass_token:
            headers["OAI-Sites-Authorization"] = f"Bearer {site_bypass_token}"
        request = Request(
            endpoint,
            data=encoded,
            method="POST",
            headers=headers,
        )
        with _open_request(request, timeout=10) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            raw_body = response.read(64_000)
        if not 200 <= status_code < 300:
            return {"ok": False, "error": f"HTTP_{status_code}"}
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            body = {}
        if isinstance(body, Mapping) and body.get("ok") is False:
            return {"ok": False, "error": "INGEST_REJECTED"}
        return {"ok": True, "items": len(payload.get("items") or [])}
    except HTTPError as exc:
        return {"ok": False, "error": f"HTTP_{exc.code}"}
    except URLError:
        return {"ok": False, "error": "NETWORK_ERROR"}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__.upper()}


def _safe_timestamp(value: Any) -> str:
    scalar = _scalar(value)
    return str(scalar)[:80] if scalar is not None else datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sync_streamlit_member_boards(
    st: Any,
    storage_dir: Any = None,
    pick_log: Any = None,
    result_log: Any = None,
    source_version: Any = None,
) -> dict[str, Any]:
    """Push changed frozen boards; missing configuration is a safe no-op."""
    endpoint = str(os.environ.get("MEMBER_BOARD_INGEST_URL") or "").strip()
    token = str(os.environ.get("MEMBER_BOARD_SYNC_TOKEN") or "").strip()
    site_bypass_token = str(os.environ.get("MEMBER_SITE_BYPASS_TOKEN") or "").strip()
    if not endpoint or not token:
        return {"ok": False, "configured": False, "status": "NOT_CONFIGURED", "posted": [], "skipped": []}
    if not endpoint.startswith("https://") and not re.match(r"^http://(?:localhost|127\.0\.0\.1)(?::\d+)?/", endpoint):
        return {"ok": False, "configured": True, "status": "HTTPS_REQUIRED", "posted": [], "skipped": []}

    session = getattr(st, "session_state", {})
    try:
        generation = int(session.get("ow_manual_refresh_generation_v11", 0))
    except Exception:
        generation = 0
    source_version_text = str(source_version or "MLB_BATTERS")[:180]
    default_generated_at = _safe_timestamp(session.get("ow_manual_refresh_completed_at_v11"))

    sources: dict[str, tuple[list[dict[str, Any]], str, str]] = {}
    core = session.get("ow_core_board_cache_v7") or {}
    if isinstance(core, Mapping):
        for cache_key, market, state in (
            ("HRR", "hrr", "FROZEN_HRR_BOARD"),
            ("HOME_RUNS", "home-runs", "FROZEN_HOME_RUN_BOARD"),
            ("BATTER_UPSIDE", "upside", "FROZEN_UPSIDE_BOARD"),
        ):
            entry = core.get(cache_key)
            if isinstance(entry, Mapping) and "df" in entry:
                rows = _records(entry.get("df"))
                if rows is not None:
                    sources[market] = (rows, _safe_timestamp(entry.get("built_at") or default_generated_at), state)

    if "upside" in sources:
        upside_rows, upside_at, _ = sources["upside"]
        sources["games"] = (list(upside_rows), upside_at, "FROZEN_GAME_SUMMARY")

    if "ow_bfs_df" in session:
        fantasy_rows = _records(session.get("ow_bfs_df"))
        if fantasy_rows is not None:
            sources["fantasy"] = (
                fantasy_rows,
                _safe_timestamp(session.get("ow_bfs_built_at") or default_generated_at),
                "FROZEN_FANTASY_BOARD",
            )

    official_rows = _read_json_rows(pick_log)
    if official_rows is not None:
        latest = list(reversed(official_rows[-MAX_ITEMS_PER_BOARD:]))
        official_at = _lookup(latest[0], "official_snapshot_saved_at", "saved_at", "created_at") if latest else default_generated_at
        sources["official"] = (latest, _safe_timestamp(official_at), "FROZEN_OFFICIAL_PLAY")
    result_rows = _read_json_rows(result_log)
    if result_rows is not None:
        latest = list(reversed(result_rows[-MAX_ITEMS_PER_BOARD:]))
        result_at = _lookup(latest[0], "graded_at", "saved_at", "created_at") if latest else default_generated_at
        sources["results"] = (latest, _safe_timestamp(result_at), "MLB_OFFICIAL_RESULT")

    if not sources:
        return {"ok": False, "configured": True, "status": "NO_SAVED_BOARDS", "posted": [], "skipped": []}

    payloads = {
        market: _payload(market, rows, built_at, generation, source_version_text, source_state)
        for market, (rows, built_at, source_state) in sources.items()
    }
    previous = session.get(SYNC_STATE_KEY)
    fingerprints = dict(previous) if isinstance(previous, Mapping) else {}
    pending: dict[str, tuple[dict[str, Any], str]] = {}
    skipped: list[str] = []
    for market, board_payload in payloads.items():
        encoded = json.dumps(board_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if fingerprints.get(market) == digest:
            skipped.append(market)
        else:
            pending[market] = (board_payload, digest)

    posted: list[str] = []
    errors: dict[str, str] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=min(4, len(pending))) as executor:
            jobs = {
                executor.submit(_post_board, endpoint, token, payload, site_bypass_token): market
                for market, (payload, _) in pending.items()
            }
            for future in as_completed(jobs):
                market = jobs[future]
                result = future.result()
                if result.get("ok"):
                    posted.append(market)
                    fingerprints[market] = pending[market][1]
                else:
                    errors[market] = str(result.get("error") or "SYNC_FAILED")[:120]
    try:
        session[SYNC_STATE_KEY] = fingerprints
    except Exception:
        pass

    status = "SYNCED" if not errors else ("PARTIAL" if posted or skipped else "FAILED")
    return {
        "ok": not errors,
        "configured": True,
        "status": status,
        "posted": sorted(posted),
        "skipped": sorted(skipped),
        "errors": errors,
        "refresh_generation": generation,
    }
