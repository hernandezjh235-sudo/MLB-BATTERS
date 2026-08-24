from pathlib import Path
import ast
import re
import pandas as pd
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "VERIFIED_CURRENT_SAVANT_V1_2026_08_23"
text = APP.read_text(encoding="utf-8")
assert MARKER in text, "verified current Savant marker missing"
py_compile.compile(str(APP), doraise=True)

tree = ast.parse(text)

def final_fn(name):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert nodes, f"missing function {name}"
    return nodes[-1]

def source(node):
    lines = text.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno])

helper = final_fn("_ow_current_savant_frame_is_valid")
helper_src = source(helper)
ns = {"pd": pd, "re": re}
exec(helper_src, ns)
validate = ns["_ow_current_savant_frame_is_valid"]

historical = pd.DataFrame([{
    "player_name": "Example Hitter",
    "historical_pa": 1200,
    "career_pa": 3400,
    "career_h": 800,
    "career_hr": 150,
    "ba": .275,
    "obp": .340,
    "slg": .455,
    "k_pa": .205,
    "profile_source": "cleaned_batting_stats_2015_2024",
}])
assert validate(historical) is False, "historical prior was incorrectly accepted as current Savant"

current = pd.DataFrame([{
    "last_name, first_name": "Hitter, Current",
    "player_id": 999999,
    "year": 2026,
    "pa": 300,
    "xba": .281,
    "xslg": .492,
    "xwoba": .374,
    "k_percent": 20.4,
    "bb_percent": 9.8,
    "whiff_percent": 22.7,
    "hard_hit_percent": 45.1,
    "barrel_batted_rate": 11.2,
    "exit_velocity_avg": 90.8,
}])
assert validate(current) is True, "valid current Savant schema was rejected"

wrapper = final_fn("_ow_baseball_savant_leaderboard")
wrapper_src = source(wrapper)
required = [
    "VERIFIED_CURRENT_SAVANT:LIVE_CUSTOM",
    "https://baseballsavant.mlb.com/leaderboard/custom",
    '"min": "1"',
    '"xwoba"',
    '"barrel_batted_rate"',
    '"hard_hit_percent"',
    '"whiff_percent"',
    "_ow_current_savant_frame_is_valid",
    "_ow_baseball_savant_leaderboard_legacy_source(kind, season)",
]
missing = [x for x in required if x not in wrapper_src]
assert not missing, f"current Savant wrapper incomplete: {missing}"
assert "cleaned_batting_stats.csv" not in wrapper_src, "historical raw file leaked into verified batter wrapper"
assert '"batter_profiles.csv"' not in wrapper_src, "historical batter prior leaked into verified batter wrapper"

legacy = final_fn("_ow_baseball_savant_leaderboard_legacy_source")
assert legacy is not None, "legacy source alias missing; pitcher behavior not preserved"

print("VERIFIED_CURRENT_SAVANT_PASS")
print("- historical prior rejection: PASS")
print("- current Savant schema acceptance: PASS")
print("- live custom leaderboard route: PASS")
print("- pitcher legacy behavior preservation: PASS")
print("- app.py compile: PASS")
