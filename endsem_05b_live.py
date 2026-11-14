# %% [markdown]
# # End-sem 05B. The live engine, the summer-2026 test and the current shortlists
# FBref's advanced statistics stop at 2024-25, so the full engine cannot score the present. The live engine uses only what exists for every season including 2025-26: Understat player-season statistics and Transfermarkt (clubs, minutes, values, transfers).
# The Transfermarkt dump ends in early July 2026 and holds no 2026-27 match. "Now" in this notebook therefore means the state of that dump.

# %%
import sys, os, time, pickle; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *
import engine as E, live as LV, viz
M = {}; rng = np.random.default_rng(SEED); TEST = (2020, 2021, 2022, 2023, 2024); KS = (5, 10, 25, 50)
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
t0 = time.time(); lv = LV.Live(); D = lv.D; P = lv.events(2014, 2025); build_s = time.time() - t0
M['data'] = {'player_seasons': len(D), 'seasons': '2014-15 to 2025-26', 'match_rate_900_by_season': {int(k): float(v) for k, v in lv.match['matched_900'].items()}, 'match_rate_900_min': float(lv.match['matched_900'].min()), 'events': int(P['qid'].nunique()), 'signings': int(P['y'].sum()),
             'signings_by_window': {int(k) + 1: int(v) for k, v in lv.S.groupby('t').size().items()}, 'mean_candidates': float(P.groupby('qid').size().mean()), 'build_seconds': round(build_s, 1)}
print(M['data'])

# %% [markdown]
# ## 1. Which extra features earn their place?
# Four ideas, all available historically: the selling club finished in the bottom three, how often the buyer has bought from that club or league before, a same-league move, and the candidate's minutes trend. Each is added to the base model and kept only if the walk-forward mean reciprocal rank on 2020-21 to 2024-25 improves with at least 90% bootstrap probability. The summer-2026 window plays no part in this choice.

# %%
def wf(feats, n_models=3): R = E.walk_forward(P[P['t'] <= 2024], feats, TEST, use_candidates=False, n_models=n_models); return LV.true_rank(R, 'score')
cur = wf(LV.BASE); feats = list(LV.BASE); sel = []; bs = rng.integers(0, len(cur), (2000, len(cur)))
for f in LV.EXTRA:
    r_new = wf(feats + [f]).reindex(cur.index); d = (1 / r_new - 1 / cur).to_numpy(); pg = float((d[bs].mean(1) > 0).mean()); keep = bool(d.mean() > 0 and pg >= 0.9)
    sel.append({'feature': f, 'mrr_gain': float(d.mean()), 'p_gain': pg, 'recall10_before': float((cur <= 10).mean()), 'recall10_after': float((r_new <= 10).mean()), 'kept': keep})
    if keep: feats.append(f); cur = r_new
SEL = pd.DataFrame(sel); display(SEL.round(4)); FEATS = feats; M['feature_selection'] = {'base': LV.BASE, 'tested': sel, 'final': FEATS}

# %% [markdown]
# ## 2. Backtest of the live engine, next to the full FBref engine

# %%
def perm_p(a, b, n=20000): d = a - b; sg = rng.choice([-1, 1], (n, len(d))); return float(((np.abs((sg * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n + 1))
def table(T, scores, label='Live engine'):
    q = np.array(sorted(T['qid'].unique())); rk = {'Highest market value': LV.true_rank(T, 'value'), 'Best performer (composite)': LV.true_rank(T, 'composite'), 'Value fit (closest in value to the club\'s player)': LV.true_rank(T, 'value_gap', asc=True),
                                                  'Profile similarity to the club\'s player': LV.true_rank(T, 'sim_profile'), label: LV.true_rank(scores, 'score')}
    n_ = T.groupby('qid').size().reindex(q).to_numpy(); b_ = rng.integers(0, len(q), (4000, len(q))); rows = [{'method': 'Random order', 'median_rank': float(np.median((n_ + 1) / 2)), **{f'recall@{k}': float(np.minimum(k / n_, 1).mean()) for k in KS}}]
    for name, r in rk.items():
        r = r.reindex(q).to_numpy(); row = {'method': name, 'median_rank': float(np.median(r)), 'MRR': float((1 / r).mean())}
        for k in KS: h = (r <= k).astype(float); row[f'recall@{k}'] = float(h.mean()); row[f'recall@{k}_lo'], row[f'recall@{k}_hi'] = [float(x) for x in np.percentile(h[b_].mean(1), [2.5, 97.5])]
        if name != label: row['p_vs_engine@10'] = perm_p((rk[label].reindex(q).to_numpy() <= 10).astype(float), (r <= 10).astype(float)); row['p_vs_engine@25'] = perm_p((rk[label].reindex(q).to_numpy() <= 25).astype(float), (r <= 25).astype(float))
        rows.append(row)
    return pd.DataFrame(rows), rk
BT, models = E.walk_forward(P[P['t'] <= 2024], FEATS, TEST, use_candidates=False, keep_models=True); Tb = P[P['t'].isin(TEST)]; TB, rkb = table(Tb, BT); display(TB[['method', 'median_rank'] + [f'recall@{k}' for k in KS]].round(3))
FA = pd.read_parquet('lake/scores_all_signings.parquet'); rf = LV.true_rank(FA, 'score'); rl = rkb['Live engine']; common = rf.index.intersection(rl.index)
cmp_ = {'common_queries': len(common), **{f'full_recall@{k}': float((rf[common] <= k).mean()) for k in (10, 25)}, **{f'live_recall@{k}': float((rl[common] <= k).mean()) for k in (10, 25)}, 'full_median_rank': float(rf[common].median()), 'live_median_rank': float(rl[common].median()),
        'p@10': perm_p((rf[common] <= 10).to_numpy(float), (rl[common] <= 10).to_numpy(float)), 'p@25': perm_p((rf[common] <= 25).to_numpy(float), (rl[common] <= 25).to_numpy(float))}
h24 = Tb[Tb['t'] == 2024]; T24, _ = table(h24, BT[BT['t'] == 2024]); q24 = [q for q in common if q.split('_')[1] == '2024']
M['backtest'] = {'test_queries': int(Tb['qid'].nunique()), 'table': TB.round(4).to_dict('records'), 'vs_full_engine': cmp_, 'holdout_2024': {'queries': int(h24['qid'].nunique()), 'table': T24.round(4).to_dict('records'), 'full_engine_recall@10_common': float((rf[q24] <= 10).mean()), 'live_recall@10_common': float((rl[q24] <= 10).mean()), 'common': len(q24)}}
print(cmp_)

# %% [markdown]
# ## 3. Summer 2026: a window the model never saw
# Everything is frozen at 31 May 2026: statistics from 2025-26, market values dated before 1 June 2026, and a ranker trained on the windows up to summer 2025. Every club-position that made a signing is scored, and the list is compared with the signings recorded in the dump (dated 30 June and 1 July 2026: the deals agreed early in the window).

# %%
m26 = E.fit_ranker(P[P['t'] <= 2024], FEATS); T26 = P[P['t'] == 2025].sort_values(['qid', 'player_id']).copy(); T26['score'] = m26.predict(T26[FEATS]); S26, rk26 = table(T26, T26); display(S26[['method', 'median_rank'] + [f'recall@{k}' for k in KS] + ['p_vs_engine@25']].round(3))
mr = LV.member_ranks(m26, T26, FEATS) if False else None; T26['rank'] = T26.groupby('qid')['score'].rank(ascending=False, method='min'); T26['pool'] = T26.groupby('qid')['score'].transform('size')
for i, mm in enumerate(m26.ms): T26[f'r{i}'] = pd.Series(mm.predict(T26[FEATS]), index=T26.index).groupby(T26['qid']).rank(ascending=False, method='min')
T26['rank_lo'] = T26[[f'r{i}' for i in range(len(m26.ms))]].quantile(0.1, axis=1).round(); T26['rank_hi'] = T26[[f'r{i}' for i in range(len(m26.ms))]].quantile(0.9, axis=1).round()
ev26 = T26[T26['y'] == 1].merge(lv.S[lv.S['t'] == 2025][['qid', 'N', 'fn', 'tn', 'fee', 'd']].rename(columns={'N': 'player_id'}), on=['qid', 'player_id']).sort_values('rank')
ev26['mv_rank'] = [int(T26[T26['qid'] == q]['value'].rank(ascending=False, method='min')[T26[T26['qid'] == q]['player_id'] == p].iat[0]) for q, p in zip(ev26['qid'], ev26['player_id'])]
E26 = ev26[['name', 'fn', 'tn', 'P', 'fee', 'value', 'rank', 'rank_lo', 'rank_hi', 'pool', 'mv_rank']]; display(E26.head(22)); E26.to_csv('outputs/summer_2026_events.csv', index=False)
M['summer_2026'] = {'frozen_at': '2026-05-31', 'signings': len(ev26), 'queries': int(T26['qid'].nunique()), 'mean_candidates': float(T26.groupby('qid').size().mean()), 'window_dates': [str(ev26['d'].min().date()), str(ev26['d'].max().date())], 'table': S26.round(4).to_dict('records'),
                    'events': E26.assign(fee=E26['fee'].fillna(-1)).round(1).to_dict('records'), 'engine_top10': int((ev26['rank'] <= 10).sum()), 'engine_top25': int((ev26['rank'] <= 25).sum()), 'market_value_top10': int((ev26['mv_rank'] <= 10).sum()), 'market_value_top25': int((ev26['mv_rank'] <= 25).sum())}
print({k: v for k, v in M['summer_2026'].items() if k not in ('table', 'events')})

# %% [markdown]
# ## 4. Drift: do the 2025-26 inputs look like the training data?

# %%
tr_ = P[P['t'] <= 2024]; ps = {f: LV.psi(tr_[f].to_numpy(float), T26[f].to_numpy(float)) for f in FEATS}; M['psi'] = {'by_feature': ps, 'flagged_over_0.2': [f for f, v in ps.items() if v > 0.2], 'max': max(ps.values())}; print({k: round(v, 3) for k, v in ps.items()})

# %% [markdown]
# ## 5. Current shortlists
# For every elite club of 2025-26: its starters (1,350+ league minutes), each with a departure risk, and a shadow list for each. Departure risk is a gradient-boosted classifier trained on past elite starters (did he play under 450 league minutes for the club the next season?). Contract expiry is shown where Transfermarkt has it, but it is not a model input: the dump only holds today's contract, so it cannot be backtested.
# Candidates are filtered to the present: at a five-league club per Transfermarkt's current club, no move recorded since 1 June 2026, 900+ minutes in 2025-26, age as of today.

# %%
cm = LV.rp('tm_club_minutes'); nxt = cm.groupby(['player_id', 'club_id', 'season'])['minutes'].sum()
St = D[D['elite'] & (D['minutes'] >= 1350)].copy(); St['tenure'] = St.groupby(['player_id', 'club_id']).cumcount() + 1; St['gone'] = [int(nxt.get((p, c, y + 1), 0) < 450) for p, c, y in zip(St['player_id'], St['club_id'], St['yr'])]; St['lv'] = np.log10(St['value'])
RF = ['age', 'min_share', 'min_trend', 'lv', 'value_trend', 'tenure', 'composite']; St[RF] = St[RF].fillna(St[RF].median()); aucs = {}
for Y in TEST:
    g = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED).fit(St[St['yr'] <= Y - 1][RF], St[St['yr'] <= Y - 1]['gone']); te = St[St['yr'] == Y]; aucs[Y] = float(roc_auc_score(te['gone'], g.predict_proba(te[RF])[:, 1]))
risk_m = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED).fit(St[St['yr'] <= 2024][RF], St[St['yr'] <= 2024]['gone']); NOW = St[St['yr'] == 2025].copy(); NOW['risk'] = risk_m.predict_proba(NOW[RF])[:, 1]
pl = lv.pl; NOW['contract_until'] = NOW['player_id'].map(pd.to_datetime(pl['contract_expiration_date'], errors='coerce')).dt.strftime('%Y-%m'); NOW['club'] = NOW['club_id'].map(lv.names); today = pd.Timestamp('2026-11-14')
NOW['age_today'] = ((today - NOW['player_id'].map(pl['dob'])).dt.days / 365.25).round(1); NOW['qid'] = NOW['club_id'].astype(str) + '_2025_' + NOW['player_id'].astype(str)
Q = NOW.rename(columns={'club_id': 'C', 'yr': 't', 'player_id': 'S', 'fine_pos': 'P'})[['qid', 'C', 't', 'S', 'P']]; SH = lv.pairs(Q, ref='starter')
five_now = set(pd.read_csv('data/transfermarkt/clubs.csv').query("domestic_competition_id in ['GB1','ES1','L1','IT1','FR1']")['club_id']); moved = set(lv.T[lv.T['d'] >= '2026-06-01']['player_id'])
SH = SH[SH['player_id'].map(pl['current_club_id']).isin(five_now) & ~SH['player_id'].isin(moved)].sort_values(['qid', 'player_id']).copy(); SH['score'] = m26.predict(SH[FEATS]); SH['rank'] = SH.groupby('qid')['score'].rank(ascending=False, method='min')
for i, mm in enumerate(m26.ms): SH[f'r{i}'] = pd.Series(mm.predict(SH[FEATS]), index=SH.index).groupby(SH['qid']).rank(ascending=False, method='min')
SH['rank_lo'] = SH[[f'r{i}' for i in range(len(m26.ms))]].quantile(0.1, axis=1).round().astype(int); SH['rank_hi'] = SH[[f'r{i}' for i in range(len(m26.ms))]].quantile(0.9, axis=1).round().astype(int); SH = SH[SH['rank'] <= 15].copy()
WORD = {'sim_profile': lambda r: f"plays like the starter (profile similarity {r.sim_profile:.2f})", 'traj_sim': lambda r: "his last seasons developed the same way", 'seller_sales': lambda r: "his club sells often to five-league clubs", 'log_value': lambda r: f"market value EUR {r.value / 1e6:.0f}m",
        'value_gap': lambda r: "priced close to the starter", 'age': lambda r: f"age {r.age:.0f}", 'composite': lambda r: f"output above {r.composite:.0f}% of his position", 'min_share': lambda r: f"played {r.min_share:.0%} of league minutes", 'seller_bottom3': lambda r: "his club finished in the bottom three",
        'hist_club': lambda r: "the club has bought from his club before", 'hist_league': lambda r: "the club buys often from his league", 'same_league': lambda r: "same league", 'min_trend': lambda r: "his minutes are rising"}
ex = viz.explain(m26, SH[FEATS]); order = np.argsort(-ex.values, axis=1)[:, :3]
SH['reasons'] = ['; '.join(WORD[FEATS[j]](r) for j in o if ex.values[i, j] > 0) for i, (o, r) in enumerate(zip(order, SH.itertuples()))]
SH['club'] = SH['club_id'].map(lv.names); SH['age_today'] = ((today - SH['player_id'].map(pl['dob'])).dt.days / 365.25).round(1); SH['contract_until'] = SH['player_id'].map(pd.to_datetime(pl['contract_expiration_date'], errors='coerce')).dt.strftime('%Y-%m')
keep_sq = ['qid', 'club_id', 'club', 'league', 'player_id', 'name', 'fine_pos', 'age_today', 'minutes', 'min_share', 'min_trend', 'value', 'value_trend', 'tenure', 'composite', 'risk', 'contract_until'] + LV.ZL
NOW[keep_sq].to_parquet('lake/live_squads.parquet', index=False); SH[['qid', 'player_id', 'name', 'club', 'league', 'fine_pos', 'age_today', 'value', 'ref_value', 'score', 'rank', 'rank_lo', 'rank_hi', 'reasons', 'contract_until', 'minutes', 'composite', 'sim_profile'] + FEATS[:0] + LV.ZL].to_parquet('lake/live_shortlists.parquet', index=False)
pickle.dump({'model': m26, 'feats': FEATS}, open('lake/live_model.pkl', 'wb')); D[['player_id', 'yr', 'name', 'league', 'club_id', 'fine_pos', 'minutes', 'value'] + LV.ZL].to_parquet('lake/live_players.parquet', index=False)
top_risk = NOW.nlargest(8, 'risk')[['name', 'club', 'fine_pos', 'age_today', 'risk', 'contract_until']]; display(top_risk.round(2))
M['shortlists'] = {'elite_clubs': int(NOW['club_id'].nunique()), 'starters': len(NOW), 'risk_auc_by_season': aucs, 'risk_auc_mean': float(np.mean(list(aucs.values()))), 'base_rate_gone': float(St[St['yr'] <= 2024]['gone'].mean()), 'starters_risk_over_0.5': int((NOW['risk'] >= 0.5).sum()),
                   'candidates_per_list': 15, 'shortlist_rows': len(SH), 'mean_rank_interval_width_top5': float((SH[SH['rank'] <= 5]['rank_hi'] - SH[SH['rank'] <= 5]['rank_lo']).mean()), 'contract_known_share': float(NOW['contract_until'].notna().mean()), 'top_risk': top_risk.round(3).to_dict('records')}
print({k: v for k, v in M['shortlists'].items() if k != 'top_risk'})

# %%
LABS = {'Random order': ('Random', GREY), 'Highest market value': ('Market value', BLUE), 'Best performer (composite)': ('Best performer', GREEN), "Value fit (closest in value to the club's player)": ('Value fit', AMBER), "Profile similarity to the club's player": ('Profile similarity', PINK), 'Live engine': ('Live engine', ORANGE)}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1, 1.25]}); Sx = S26.set_index('method')
for mth, (lb, col) in LABS.items():
    y = [Sx.loc[mth, f'recall@{k}'] for k in KS]; ax[0].plot(KS, y, 'o-', color=col, lw=2.8 if lb == 'Live engine' else 1.3, zorder=3 if lb == 'Live engine' else 2); ax[0].text(KS[-1] + 1.2, y[-1], lb, color=col, fontsize=8, va='center', fontweight='bold' if lb == 'Live engine' else 'normal')
ax[0].set_xticks(KS); ax[0].set_xlim(3, 66); ax[0].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[0].set_xlabel('length of the list'); ax[0].set_title(f"Summer 2026: {len(ev26)} signings, data frozen on 31 May"); ax[0].set_ylabel('signings found in the list')
e_ = ev26.sort_values('rank').head(14).iloc[::-1]; yy = np.arange(len(e_)); ax[1].barh(yy, e_['pool'], color='#eceae4', height=.6); ax[1].scatter(e_['mv_rank'], yy, color=BLUE, s=26, zorder=3); ax[1].scatter(e_['rank'], yy, color=ORANGE, s=40, zorder=4)
ax[1].hlines(yy, e_['rank_lo'], e_['rank_hi'], color=ORANGE, lw=2, zorder=3); ax[1].set_yticks(yy, [f"{n} to {t}" for n, t in zip(e_['name'], e_['tn'])], fontsize=7.5); ax[1].set_xscale('log'); ax[1].set_xlabel('rank among the candidates for that club and position (log scale; grey bar = number of candidates)')
ax[1].text(e_['rank'].iloc[-1] * 1.25, yy[-1] + .55, 'engine rank, with its interval', color=ORANGE, fontsize=7.5); ax[1].text(e_['mv_rank'].iloc[0] * 1.2, yy[0] - .62, 'rank by market value', color=BLUE, fontsize=7.5); ax[1].set_title('Where each real signing sat in our list'); ax[1].grid(axis='y', visible=False); savefig('p5b_summer_2026')
ids = set(D['player_id']) | set(pd.read_parquet('lake/features', columns=['player_id'])['player_id']); vh = lv.V[lv.V['player_id'].isin(ids) & (lv.V['date'] >= '2016-01-01')].copy(); vh['q'] = vh['date'].dt.to_period('Q')
vh.sort_values('date').drop_duplicates(['player_id', 'q'], keep='last')[['player_id', 'date', 'value']].to_parquet('lake/value_history.parquet', index=False)      # one valuation per quarter, for the dashboard sparkline
_ = put_metrics('live', M)
