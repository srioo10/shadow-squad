# %% [markdown]
# # End-sem 06. Explanations, case studies and the dashboard screenshots
# SHAP on the ranker, the translated radar, three hits and one bad miss. The same figures are drawn live by `app/shadow_squad.py`.

# %%
import sys, os, subprocess, time; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from common import *
import engine as E, viz
lake = E.Lake(); models = E.load_models(); PF = pd.read_parquet('lake/pairs.parquet'); LG = pd.read_parquet('lake/scores_lgbm_v2.parquet'); EV = pd.read_parquet('lake/replacement_events')
EV['qid'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str); M = {}

# %% [markdown]
# ## 1. What drives the ranking overall (SHAP, out-of-time rows only)

# %%
parts = []
for Y in (2020, 2021, 2022, 2023, 2024):
    X = PF[(PF['t'] == Y) & PF['q_in_pool']]; parts.append((viz.explain(models[Y], X[E.FEATS]).values, X))
SV = np.vstack([p[0] for p in parts]); XA = pd.concat([p[1] for p in parts]); imp = pd.Series(np.abs(SV).mean(0), index=E.FEATS).sort_values()
dirn = {f: float(np.corrcoef(XA[f].fillna(XA[f].median()), SV[:, i])[0, 1]) for i, f in enumerate(E.FEATS)}
fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.grid(axis='y', visible=False); ax.barh([viz.NICE[f] for f in imp.index], imp.values, color=[ORANGE if dirn[f] > 0 else BLUE for f in imp.index]); ax.set_xlabel('mean |SHAP|, unseen events (orange raises rank, blue lowers)'); ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig('outputs/figs/p6_shap_global.png'); plt.close()
M['shap_global'] = {'mean_abs_shap': imp.round(4).to_dict(), 'direction_corr': {k: round(v, 3) for k, v in dirn.items()}, 'rows': len(XA)}; print(M['shap_global']['mean_abs_shap'])

# %% [markdown]
# ## 2. Case studies: three hits and one miss
# Chosen by rule, not by hand: the three unseen events with the best rank for the real signing (ties broken by the larger fee), and the unseen event where a real signing was ranked worst.

# %%
R = LG.merge(PF[['qid', 'player_id', 'candidate']], on=['qid', 'player_id']); R['rank'] = R.groupby('qid')['score'].rank(ascending=False, method='first'); R['pool'] = R.groupby('qid')['score'].transform('size')
tr = R[R['y'] == 1].merge(EV[['qid', 'N', 'club', 't', 'S_name', 'P', 'N_name', 'N_from', 'fee', 'C', 'S', 'league']].rename(columns={'N': 'player_id', 't': 't_'}), on=['qid', 'player_id'])
hits = tr[tr['rank'] <= 10].sort_values(['rank', 'fee'], ascending=[True, False]).drop_duplicates('qid').head(3); miss = tr.sort_values('rank', ascending=False).head(1); cases = []
for kind, row in [('hit', r) for r in hits.itertuples()] + [('miss', r) for r in miss.itertuples()]:
    X = E.rank(lake, models, row.C, row.t, row.S, row.P, top=10 ** 6); X['rank'] = range(1, len(X) + 1); me = X[X['player_id'] == row.player_id].iloc[0]; tag = f"{kind}_{row.club.replace(' ', '')}_{row.t}_{row.N_name.split()[-1]}"
    pct = {f: float((X[f] < me[f]).mean()) for f in E.FEATS}
    for who, lab_ in ((me, 'signed'), (X.iloc[0], 'top1')):
        viz.radar(lake, who, row.S, row.t).savefig(f'outputs/figs/case_{tag}_{lab_}_radar.png'); plt.close('all'); viz.waterfall(models[row.t], X, int(who.name)).savefig(f'outputs/figs/case_{tag}_{lab_}_shap.png'); plt.close('all')
    cases.append({'kind': kind, 'tag': tag, 'club': row.club, 'league': row.league, 'season': int(row.t), 'starter': row.S_name, 'position': row.P, 'signed': row.N_name, 'from': row.N_from, 'fee': None if pd.isna(row.fee) else float(row.fee), 'rank': int(me['rank']), 'pool': len(X),
                  'signed_feature_percentile_in_pool': {k: round(v, 2) for k, v in pct.items()}, 'signed_values': {f: (None if pd.isna(me[f]) else round(float(me[f]), 3)) for f in E.FEATS}, 'starter_value': None if pd.isna(me['S_value']) else float(me['S_value']), 'signed_value': float(me['value']),
                  'top5': X.head(5)[['player', 'squad', 'age_num', 'value', 'score']].round(3).to_dict('records'), 'url': f"?league={row.league}&club={row.club}&t={row.t}&S={row.S}&X={row.player_id}".replace(' ', '%20')})
    print(f"{kind.upper()}: {row.club} {row.t}-{str(row.t + 1)[2:]}, {row.S_name} ({row.P}) -> {row.N_name} from {row.N_from}; ranked {int(me['rank'])} of {len(X)}; percentiles {cases[-1]['signed_feature_percentile_in_pool']}")
M['cases'] = cases

# %% [markdown]
# ## 3. Dashboard screenshots (taken from the running Streamlit app)

# %%
import json
json.dump([{'tag': c['tag'], 'url': c['url']} for c in cases], open('outputs/screens/cases.json', 'w'))
out = subprocess.run([sys.executable, 'src/screens.py', 'outputs/screens/cases.json'], capture_output=True, text=True); print(out.stderr[-800:]); shots = out.stdout.split()
assert len(shots) == 2 * len(cases), 'screenshots missing'
M['screens'] = shots; print(shots); _ = put_metrics('phase6', M)
