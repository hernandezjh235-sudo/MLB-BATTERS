from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_SCHEDULE_TEAM_IDS_V1_2026_08_26"
TARGET = "_v3_team_schedule_context_map"

text = APP.read_text(encoding="utf-8").replace("\r\n", "\n")
if MARKER in text:
    py_compile.compile(str(APP), doraise=True)
    print("Verified schedule team-ID patch already present.")
    raise SystemExit(0)

tree = ast.parse(text)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == TARGET]
if not nodes:
    raise RuntimeError(f"Could not find {TARGET}")
node = nodes[-1]
lines = text.splitlines(keepends=True)
old_fn = "".join(lines[node.lineno - 1:node.end_lineno])
legacy_fn = re.sub(
    r"^(\s*)def\s+_v3_team_schedule_context_map\s*\(",
    r"\1def _v3_team_schedule_context_map_live_v1(",
    old_fn,
    count=1,
    flags=re.M,
)
if legacy_fn == old_fn:
    raise RuntimeError("Could not safely rename final schedule context function")

wrapper = r'''# VERIFIED_SCHEDULE_TEAM_IDS_V1_2026_08_26
# Data-only repair: guarantee stable MLB team IDs for opponent bullpen routing.
_OW_MLB_TEAM_IDS_2026 = {
    "ARI":109, "ATL":144, "BAL":110, "BOS":111, "CHC":112, "CWS":145,
    "CIN":113, "CLE":114, "COL":115, "DET":116, "HOU":117, "KC":118,
    "KCR":118, "LAA":108, "LAD":119, "MIA":146, "MIL":158, "MIN":142,
    "NYM":121, "NYY":147, "ATH":133, "OAK":133, "PHI":143, "PIT":134,
    "SD":135, "SDP":135, "SF":137, "SFG":137, "SEA":136, "STL":138,
    "TB":139, "TBR":139, "TEX":140, "TOR":141, "WSH":120, "WAS":120,
}


def _ow_team_id_from_abbr_2026(abbr):
    return _OW_MLB_TEAM_IDS_2026.get(str(abbr or "").upper().strip())


@st.cache_data(ttl=300, show_spinner=False)
def _v3_team_schedule_context_map():
    ctx = _v3_team_schedule_context_map_live_v1() or {}
    if not isinstance(ctx, dict):
        return {}
    out = {}
    for key, value in ctx.items():
        row = dict(value) if isinstance(value, dict) else {}
        team = str(row.get("Team") or key or "").upper()
        opp = str(row.get("Opponent") or "").upper()
        if not row.get("Team ID"):
            row["Team ID"] = _ow_team_id_from_abbr_2026(team)
        if not row.get("Opp Team ID"):
            row["Opp Team ID"] = _ow_team_id_from_abbr_2026(opp)
        row["_OW Team ID Source"] = "MLB_STABLE_TEAM_ID_MAP_2026"
        out[str(key).upper()] = row
    return out
'''

replacement = legacy_fn.rstrip() + "\n\n" + wrapper.strip() + "\n"
text = "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])
ast.parse(text)
APP.write_text(text, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Guaranteed Team ID / Opp Team ID for bullpen routing; formulas/UI unchanged.")
