from pathlib import Path
import ast
import re
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# VERIFIED_MLBAM_MATCHING_V1_2026_08_26"

text = APP.read_text(encoding="utf-8").replace("\r\n", "\n")
if MARKER in text:
    py_compile.compile(str(APP), doraise=True)
    print("MLBAM-first Savant matching patch already present.")
    raise SystemExit(0)

# 1) Upgrade the final Savant row matcher: exact MLBAM first, normalized name only as fallback.
tree = ast.parse(text)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ow_savant_player_row"]
if not nodes:
    raise RuntimeError("_ow_savant_player_row not found")
node = nodes[-1]
lines = text.splitlines(keepends=True)
old_fn = "".join(lines[node.lineno - 1:node.end_lineno])
new_fn = r'''# VERIFIED_MLBAM_MATCHING_V1_2026_08_26
def _ow_savant_player_row(df, player, player_id=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # Canonical join: MLBAM/player id. Never fuzzy-match when a valid exact id exists.
    if player_id not in (None, "", "—"):
        id_col = _ow_savant_col(df, ["player_id", "mlbam_id", "mlbam", "key_mlbam", "batter_id", "id"])
        if id_col:
            try:
                target = float(player_id)
                ids = pd.to_numeric(df[id_col], errors="coerce")
                d = df[ids == target]
                if not d.empty:
                    rr = d.iloc[0].to_dict()
                    rr["_OW Savant Match Method"] = "MLBAM_ID_EXACT"
                    rr["_OW Savant Match Confidence"] = "EXACT"
                    return rr
            except Exception:
                pass

    # Secondary fallback: normalized exact name only. No loose fuzzy substitution.
    if not player:
        return None
    candidates = ["player_name", "last_name, first_name", "Name", "Player", "batter", "pitcher", "player"]
    pcol = _ow_savant_col(df, candidates)
    if not pcol:
        return None
    target = _ow_norm_name(player)
    try:
        norm_series = df[pcol].fillna("").astype(str).map(_ow_norm_name)
        exact = df[norm_series == target]
        if not exact.empty:
            rr = exact.iloc[0].to_dict()
            rr["_OW Savant Match Method"] = "NORMALIZED_NAME_EXACT"
            rr["_OW Savant Match Confidence"] = "FALLBACK"
            return rr
    except Exception:
        return None
    return None
'''
text = "".join(lines[:node.lineno - 1]) + new_fn + "\n" + "".join(lines[node.end_lineno:])

# 2) Add optional player_id to every duplicated Savant hitter-context definition and use it in the row join.
text, n_defs = re.subn(
    r"def _ow_baseball_savant_hitter_context\(player, season=2026\):",
    "def _ow_baseball_savant_hitter_context(player, season=2026, player_id=None):",
    text,
)
if n_defs < 1:
    raise RuntimeError("Savant hitter context definition not found")
text, n_rows = re.subn(
    r"row = _ow_savant_player_row\(df, player\)",
    "row = _ow_savant_player_row(df, player, player_id=player_id)",
    text,
)
if n_rows < 1:
    raise RuntimeError("Savant hitter row lookup not found")

# 3) Pass the already-resolved MLBAM ID from HRR/HR builders and BFS context.
text = text.replace(
    "savant_ctx = _ow_baseball_savant_hitter_context(player)",
    "savant_ctx = _ow_baseball_savant_hitter_context(player, player_id=prof.get(\"player_id\"))",
)
text = text.replace(
    'ctx["Savant Hitter"] = _ow_baseball_savant_hitter_context(player)',
    'ctx["Savant Hitter"] = _ow_baseball_savant_hitter_context(player, player_id=pid)',
)

ast.parse(text)
APP.write_text(text, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Applied MLBAM-first batter Savant matching; formulas/UI unchanged.")
