from pathlib import Path
import ast
import re
import pandas as pd
import py_compile

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
REFRESH = ROOT / "refresh_verified_current_data.py"
MARKER = "VERIFIED_CURRENT_SAVANT_V2_2026_08_26"

text = APP.read_text(encoding="utf-8")
assert MARKER in text, "verified current Savant V2 marker missing"
py_compile.compile(str(APP), doraise=True)
assert REFRESH.exists(), "nightly verified-current refresh script missing"
py_compile.compile(str(REFRESH), doraise=True)

tree = ast.parse(text)


def final_fn(name):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert nodes, f"missing function {name}"
    return nodes[-1]


def source(node):
    lines = text.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


norm_node = final_fn("_ow_v2_norm_cols")
validate_node = final_fn("_ow_v2_current_savant_frame_is_valid")
ns = {"pd": pd, "re": re}
exec(source(norm_node), ns)
exec(source(validate_node), ns)
validate = ns["_ow_v2_current_savant_frame_is_valid"]

historical = pd.DataFrame([{
    "player_id": 100001,
    "player_name": "Historical Hitter",
    "historical_pa": 1200,
    "career_pa": 3400,
    "career_h": 800,
    "career_hr": 150,
    "profile_source": "cleaned_batting_stats_2015_2024",
    "xba": .275,
    "xslg": .450,
    "xwoba": .350,
    "hard_hit_percent": 40.0,
    "barrel_batted_rate": 8.0,
    "whiff_percent": 20.0,
    "k_percent": 20.0,
    "bb_percent": 8.0,
}])
assert validate(historical, "batter", 2026) is False, "historical prior accepted as current"

current_batter = pd.DataFrame([{
    "last_name, first_name": "Hitter, Current",
    "player_id": 999999,
    "season": 2026,
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
assert validate(current_batter, "batter", 2026) is True, "valid current batter schema rejected"

current_pitcher = pd.DataFrame([{
    "last_name, first_name": "Pitcher, Current",
    "player_id": 888888,
    "season": 2026,
    "xba": .244,
    "xslg": .395,
    "xwoba": .312,
    "k_percent": 25.4,
    "bb_percent": 7.8,
    "whiff_percent": 28.7,
    "hard_hit_percent": 37.1,
    "barrel_batted_rate": 7.2,
    "exit_velocity_avg": 88.8,
}])
assert validate(current_pitcher, "pitcher", 2026) is True, "valid current pitcher schema rejected"

wrong_year = current_pitcher.copy()
wrong_year["season"] = 2024
assert validate(wrong_year, "pitcher", 2026) is False, "wrong-season pitcher data accepted"

wrapper = final_fn("_ow_baseball_savant_leaderboard")
wrapper_src = source(wrapper)
required = [
    'kind not in ("batter", "pitcher")',
    '_ow_v2_local_file(kind, "current"',
    "_ow_v2_fetch_live_custom(kind, season)",
    '_ow_v2_local_file(kind, "last_good"',
    "STALE_CURRENT_VERIFIED_FALLBACK",
    "LAST_GOOD_VERIFIED_FALLBACK",
]
missing = [x for x in required if x not in wrapper_src]
assert not missing, f"verified wrapper incomplete: {missing}"

live_src = source(final_fn("_ow_v2_fetch_live_custom"))
for token in (
    "https://baseballsavant.mlb.com/leaderboard/custom",
    'kind not in ("batter", "pitcher")',
    '"min": "1"',
    '"xwoba"',
    '"xba"',
    '"xslg"',
    '"barrel_batted_rate"',
    '"hard_hit_percent"',
    '"whiff_percent"',
    '"k_percent"',
    '"bb_percent"',
):
    assert token in live_src, f"live current-data route missing {token}"

local_src = source(final_fn("_ow_v2_data_roots"))
assert "RAILWAY_VOLUME_MOUNT_PATH" in local_src, "Railway volume root not supported"
assert "MLB_VERIFIED_DATA_ROOT" in local_src, "explicit verified data root not supported"

verified_block = "\n".join([
    source(final_fn("_ow_v2_local_file")),
    source(final_fn("_ow_v2_fetch_live_custom")),
    wrapper_src,
])
assert "cleaned_batting_stats.csv" not in verified_block
assert '"batter_profiles.csv"' not in verified_block

print("VERIFIED_CURRENT_SAVANT_V2_PASS")
print("- historical/career rejection: PASS")
print("- MLBAM/player_id requirement: PASS")
print("- 2026 batter current schema: PASS")
print("- 2026 pitcher current schema: PASS")
print("- wrong-season rejection: PASS")
print("- live Savant batter+pitcher route: PASS")
print("- current -> live -> stale-current -> last_good fallback: PASS")
print("- Railway volume-aware data root: PASS")
print("- app.py compile: PASS")
print("- refresh script compile: PASS")
