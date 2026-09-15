import io, json, os, sys, traceback
from pathlib import Path
import requests
import pandas as pd

YEAR = 2026
OUT = Path('savant_refresh_output')
OUT.mkdir(exist_ok=True)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; OneWayPickz/1.0; +https://baseballsavant.mlb.com/)',
    'Accept': 'text/csv,text/plain,*/*',
}

PROFILE_SELECTIONS = [
    'pa','strikeout','k_percent','bb_percent','whiff_percent','swing_percent',
    'xwoba','xba','xslg','hard_hit_percent','barrel_batted_rate',
    'avg_swing_speed','fast_swing_rate','swords','squared_up_contact','woba'
]

def get_csv(url, params, name):
    r = requests.get(url, params=params, headers=HEADERS, timeout=(10,90))
    print(name, r.status_code, r.url, r.headers.get('content-type'), len(r.content))
    r.raise_for_status()
    text = r.content.decode('utf-8-sig', errors='replace')
    (OUT / f'{name}.raw.txt').write_text(text[:10000], encoding='utf-8')
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    df.to_csv(OUT / f'{name}.csv', index=False)
    print(name, 'shape=', df.shape, 'cols=', list(df.columns))
    return df

def custom(kind):
    return get_csv(
        'https://baseballsavant.mlb.com/leaderboard/custom',
        {
            'year': YEAR, 'type': kind, 'filter': '', 'min': '1',
            'selections': ','.join(PROFILE_SELECTIONS), 'chart': 'false',
            'x': 'pa', 'y': 'pa', 'r': 'no', 'chartType': 'beeswarm',
            'sort': 'pa', 'sortDir': 'desc', 'csv': 'true',
        }, f'custom_{kind}'
    )

def arsenal():
    return get_csv(
        'https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats',
        {'type':'pitcher','pitchType':'','year':YEAR,'team':'','min':'1','csv':'true'},
        'pitcher_arsenal'
    )

def platoon(hand):
    params = {
        'all':'true','hfPT':'','hfAB':'','hfGT':'R|','hfPR':'','hfZ':'',
        'hfStadium':'','hfBBL':'','hfNewZones':'','hfPull':'','hfC':'',
        'hfSea':f'{YEAR}|','hfSit':'','player_type':'batter','hfOuts':'',
        'hfOpponent':'','pitcher_throws':hand,'batter_stands':'','hfSA':'',
        'game_date_gt':'','game_date_lt':'','hfMo':'','hfTeam':'','home_road':'',
        'hfRO':'','position':'','hfInfield':'','hfOutfield':'','hfInn':'',
        'hfBBT':'','hfFlag':'','metric_1':'','group_by':'name','min_pitches':'0',
        'min_results':'0','min_pas':'0','sort_col':'pitches',
        'player_event_sort':'api_p_release_speed','sort_order':'desc',
        'chk_stats_pa':'on','chk_stats_k_percent':'on','type':'details',
    }
    return get_csv('https://baseballsavant.mlb.com/statcast_search/csv', params, f'platoon_{hand}')

summary = {}
for name, fn in [('custom_batter', lambda: custom('batter')), ('custom_pitcher', lambda: custom('pitcher')), ('pitcher_arsenal', arsenal), ('platoon_R', lambda: platoon('R')), ('platoon_L', lambda: platoon('L'))]:
    try:
        df = fn()
        summary[name] = {'ok': True, 'rows': int(len(df)), 'cols': list(df.columns)}
    except Exception as e:
        traceback.print_exc()
        summary[name] = {'ok': False, 'error': repr(e)}
(OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
if not all(v.get('ok') for v in summary.values()):
    sys.exit(1)
