"""Shadow Squad: pick an elite club and a departing starter, get a ranked replacement list. Run: streamlit run app/shadow_squad.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import pandas as pd, streamlit as st, matplotlib
matplotlib.use('Agg')
import engine as E, viz
from common import LEAGUES, MID

st.set_page_config(page_title='Shadow Squad', layout='wide')
@st.cache_resource
def load():
    lake = E.Lake(); ev = pd.read_parquet(E.L('replacement_events')); return lake, E.load_models(), ev
lake, models, EV = load(); F = lake.F; qp = st.query_params
starters = F[F['elite_club'] & (F['min_num'] >= MID['ELITE_MIN_MIN'])]
def pick(label, options, key, fmt=str, col=st):
    options = list(options); d = qp.get(key); idx = next((i for i, o in enumerate(options) if str(o) == d), 0); return col.selectbox(label, options, index=idx, format_func=fmt)

st.title('Shadow Squad'); st.caption('Who could replace a departing starter? Candidates: players aged 26 or under with 900+ minutes at non-elite clubs in the five leagues.')
c1, c2, c3, c4 = st.columns(4)
league = pick('League', LEAGUES, 'league', col=c1); s1 = starters[starters['comp'] == league]
club = pick('Elite club', sorted(s1['squad'].unique()), 'club', col=c2); s2 = s1[s1['squad'] == club]
t = pick('Season', sorted(s2['yr'].unique(), reverse=True), 't', lambda y: f'{y}-{str(y + 1)[2:]}', col=c3); s3 = s2[s2['yr'] == t].sort_values('min_num', ascending=False)
S = pick('Departing starter', s3['player_id'].tolist(), 'S', lambda i: f"{s3.set_index('player_id').loc[i, 'player']} ({s3.set_index('player_id').loc[i, 'fine_pos']})", col=c4)
srow = s3.set_index('player_id').loc[S]; C = int(srow['club_id']); P = srow['fine_pos']
R = E.rank(lake, models, C, int(t), int(S), P, top=10 ** 6); R['rank'] = range(1, len(R) + 1)
note = 'using only signings made before this season' if R['out_of_time'].iat[0] else 'whose training data includes this season (shown for illustration, not an out-of-time prediction)'
st.caption(f"Replacing {srow['player']} ({P}, {int(srow['min_num'])} min, age {int(srow['age_num'])}) at {club}. Pool: {len(R)} players. Ranking model: LambdaMART trained on all five-league signings, {note}.")
real = EV[(EV['C'] == C) & (EV['t'] == t) & (EV['S'] == S)]
if len(real):
    bits = [f"{r.N_name} from {r.N_from}" + (f" (ranked {int(R.loc[R['player_id'] == r.N, 'rank'].iat[0])} of {len(R)})" if (R['player_id'] == r.N).any() else ' (outside the candidate pool)') for r in real.itertuples()]
    st.info('What the club actually did: signed ' + '; '.join(bits) + '.')
show = R.head(10)[['rank', 'player', 'squad', 'age_num', 'value', 'score', 'sim_translated', 'traj_sim', 'hazard']].rename(
    columns={'squad': 'club', 'age_num': 'age', 'value': 'value (EUR)', 'sim_translated': 'translated sim.', 'traj_sim': 'trajectory sim.', 'hazard': 'hazard'})
st.dataframe(show.style.format({'value (EUR)': '{:,.0f}', 'age': '{:.0f}', 'score': '{:.2f}', 'translated sim.': '{:.2f}', 'trajectory sim.': '{:.2f}', 'hazard': '{:.0%}'}), hide_index=True)
who = pick('Explain candidate', R.head(15)['player_id'].tolist() + [n for n in real['N'] if (R['player_id'] == n).any() and n not in set(R.head(15)['player_id'])], 'X', lambda i: f"{int(R.loc[R['player_id'] == i, 'rank'].iat[0])}. {R.loc[R['player_id'] == i, 'player'].iat[0]}")
i = int(R.index[R['player_id'] == who][0]); row = R.loc[i]
a, b, c = st.columns([1.1, 1.5, 0.7])
a.subheader('Profile fit'); a.pyplot(viz.radar(lake, row, int(S), int(t))); a.caption('Per-90 numbers in standard deviations for the position. Orange is the candidate after translation into this club\'s style and league.')
b.subheader('Why this rank'); b.pyplot(viz.waterfall(models[int(R['model_used'].iat[0]) if R['model_used'].iat[0] != 'all' else 'all'], R, i)); b.caption('SHAP: how each score pushed this candidate up (red) or down (blue) from the average candidate.')
c.subheader('Value'); c.metric('Market value', f"EUR {row['value'] / 1e6:.1f}m"); c.metric('Departing starter', f"EUR {row['S_value'] / 1e6:.1f}m"); c.metric('Move-up hazard (12 months)', f"{row['hazard']:.0%}" if pd.notna(row['hazard']) else 'n/a'); c.metric('Age', f"{int(row['age_num'])}")
