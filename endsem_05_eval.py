# %% [markdown]
# # End-sem 05. Evaluation
# Main metric: replacement recall@K and the median rank of the real signing, on unseen seasons (2020-21 to 2023-24), for events whose real signing was inside the candidate pool. Every method ranks the same set: all same-position pool players of that season.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
import engine as E
M = {}; rng = np.random.default_rng(SEED); KS = (5, 10, 25, 50); TEST = (2020, 2021, 2022, 2023)
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
PF = pd.read_parquet('lake/pairs.parquet'); T = PF[PF['q_in_pool'] & PF['t'].isin(TEST)].copy(); qids = np.array(sorted(T['qid'].unique()))
def true_rank(df, col, asc=False):
    """Rank of the best-placed real signing in each query (ties get the average rank; missing scores go last)."""
    s = df[col].fillna(-np.inf if not asc else np.inf); r = s.groupby(df['qid']).rank(ascending=asc, method='average'); return r[df['y'] == 1].groupby(df['qid']).min().reindex(qids)
ranks = {'Best young performer (composite index)': true_rank(T, 'composite_perf_index'), 'Raw cosine similarity to the starter': true_rank(T, 'sim_raw'), 'Highest market value': true_rank(T, 'value')}
LG = pd.read_parquet('lake/scores_lgbm.parquet'); GB = pd.read_parquet('lake/scores_gbt.parquet')
ranks['Shadow Squad (LambdaMART, all scores)'] = true_rank(LG[LG['qid'].isin(qids)], 'score'); ranks['Shadow Squad fusion with Spark GBT (pointwise)'] = true_rank(GB[GB['qid'].isin(qids)], 'score')
for name, (drop, swap) in E.ABLATIONS.items():
    feats = [swap if f == drop else f for f in E.FEATS if not (f == drop and swap is None)]; R = E.walk_forward(PF, feats, TEST); ranks['Ablation: ' + name] = true_rank(R[R['qid'].isin(qids)], 'score')
for name, feats in (('Single score: translated similarity', 'sim_translated'), ('Single score: hazard', 'hazard'), ('Single score: value gap to the starter', 'value_gap')):
    ranks[name] = true_rank(T, feats, asc=(feats == 'value_gap'))
n_q = T.groupby('qid').size().reindex(qids).to_numpy(); n_pos = T.groupby('qid')['y'].sum().reindex(qids).to_numpy()
sim = np.array([[np.sort(rng.permutation(n)[:p])[0] + 1 for n, p in zip(n_q, n_pos)] for _ in range(2000)])      # 2,000 random orderings of every pool

# %% [markdown]
# ## 1. Master table: recall@K, median rank, bootstrap intervals over events, permutation test against "best young performer"

# %%
bs = rng.integers(0, len(qids), (4000, len(qids))); base = ranks['Best young performer (composite index)'].to_numpy(); rows = []
def perm_p(a, b, n=20000):   # paired sign-flip test on per-event differences, two-sided
    d = a - b; obs = d.mean(); sg = rng.choice([-1, 1], (n, len(d))); return float(((np.abs((sg * d).mean(1)) >= abs(obs) - 1e-12).sum() + 1) / (n + 1))
rows.append({'method': 'Random order of the pool', 'median_rank': float(np.median(sim)), **{f'recall@{k}': float((sim <= k).mean()) for k in KS}, **{f'recall@{k}_lo': float(np.percentile((sim <= k).mean(1), 2.5)) for k in KS},
             **{f'recall@{k}_hi': float(np.percentile((sim <= k).mean(1), 97.5)) for k in KS}, 'MRR': float((1 / sim).mean())})
for name, r in ranks.items():
    r = r.to_numpy(); row = {'method': name, 'median_rank': float(np.median(r)), 'median_rank_lo': float(np.percentile(np.median(r[bs], 1), 2.5)), 'median_rank_hi': float(np.percentile(np.median(r[bs], 1), 97.5)), 'MRR': float((1 / r).mean())}
    for k in KS:
        h = (r <= k).astype(float); row[f'recall@{k}'] = float(h.mean()); row[f'recall@{k}_lo'], row[f'recall@{k}_hi'] = [float(x) for x in np.percentile(h[bs].mean(1), [2.5, 97.5])]
        if name != 'Best young performer (composite index)': row[f'p_vs_best_young@{k}'] = perm_p(h, (base <= k).astype(float))
    if name != 'Best young performer (composite index)': row['p_vs_best_young_MRR'] = perm_p(1 / r, 1 / base)
    rows.append(row)
MT = pd.DataFrame(rows); MT.to_csv('outputs/master_results.csv', index=False)
pd.set_option('display.width', 250); display(MT[['method', 'median_rank'] + [f'recall@{k}' for k in KS] + ['MRR', 'p_vs_best_young@10', 'p_vs_best_young_MRR']].round(3))
full = ranks['Shadow Squad (LambdaMART, all scores)'].to_numpy(); dq = T.drop_duplicates('qid').set_index('qid').reindex(qids); qt_, qp_ = dq['t'].to_numpy(), dq['P'].to_numpy()
M['main'] = {'test_queries': len(qids), 'test_seasons': list(TEST), 'mean_pool_size': float(n_q.mean()), 'queries_by_season': T.drop_duplicates('qid').groupby('t').size().to_dict(), 'table': MT.round(4).to_dict('records'),
             'full_minus_best_young_recall10_ci95': [float(x) for x in np.percentile(((full <= 10).astype(float)[bs] - (base <= 10).astype(float)[bs]).mean(1), [2.5, 97.5])],
             'full_minus_best_young_recall25_ci95': [float(x) for x in np.percentile(((full <= 25).astype(float)[bs] - (base <= 25).astype(float)[bs]).mean(1), [2.5, 97.5])],
             'recall10_by_season_full': {int(t): float((full[qt_ == t] <= 10).mean()) for t in TEST},
             'recall10_by_position_full': {p: [int((qp_ == p).sum()), float((full[qp_ == p] <= 10).mean())] for p in sorted(set(qp_))}}
show = ['Random order of the pool', 'Highest market value', 'Raw cosine similarity to the starter', 'Best young performer (composite index)', 'Shadow Squad fusion with Spark GBT (pointwise)', 'Shadow Squad (LambdaMART, all scores)']
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8)); S = MT.set_index('method')
for mth in show:
    y = [S.loc[mth, f'recall@{k}'] for k in KS]; lo = [S.loc[mth, f'recall@{k}_lo'] for k in KS]; hi = [S.loc[mth, f'recall@{k}_hi'] for k in KS]
    ax[0].plot(KS, y, 'o-', label=mth, lw=2.2 if 'LambdaMART' in mth else 1.2); 
    if 'LambdaMART' in mth or 'young' in mth: ax[0].fill_between(KS, lo, hi, alpha=.12)
ax[0].set_xticks(KS); ax[0].set_xlabel('K (length of the list)'); ax[0].set_ylabel('share of events whose real signing is in the top K'); ax[0].legend(fontsize=6.5); ax[0].set_title(f'Replacement recall@K, {len(qids)} unseen events (bands: 95% bootstrap)', fontsize=9)
ab = ['Shadow Squad (LambdaMART, all scores)'] + ['Ablation: ' + a for a in E.ABLATIONS]; y = [S.loc[a, 'recall@10'] for a in ab]
ax[1].bar(range(len(ab)), y, yerr=[[v - S.loc[a, 'recall@10_lo'] for a, v in zip(ab, y)], [S.loc[a, 'recall@10_hi'] - v for a, v in zip(ab, y)]], capsize=3, color=[ORANGE] + [GREY] * 4)
ax[1].set_xticks(range(len(ab)), ['all scores'] + list(E.ABLATIONS), fontsize=7); ax[1].set_ylabel('recall@10'); ax[1].set_title('Ablation: one score removed at a time', fontsize=9); savefig('p5_recall_ablation')

# %% [markdown]
# ## 2. Survival model (out of course): discrete-time hazard and Cox proportional hazards
# The discrete-time model is the Spark logistic regression on person-period rows from notebook 04 (walk-forward scores). Cox PH (`lifelines`) is fitted on players whose first feeder season is 2019-20 or earlier and tested on later entrants, with covariates from the first feeder season. Concordance = probability that, of two players, the one who moved up earlier had the higher predicted risk.

# %%
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
lake = E.Lake(); FT = lake.F; PP = pd.read_parquet('lake/person_period.parquet')[['player_id', 'yr', 'event']]; HZ = pd.read_parquet('lake/hazard.parquet')
pe = PP.merge(HZ, on=['player_id', 'yr']); auc_pp = roc_auc_score(pe['event'], pe['hazard']); auc_by = {int(y): float(roc_auc_score(g['event'], g['hazard'])) for y, g in pe.groupby('yr') if g['event'].nunique() == 2}
surv = PP.sort_values('yr').groupby('player_id').agg(first=('yr', 'min'), dur=('yr', 'size'), event=('event', 'max')).reset_index()
cov = ['age_num', 'composite_perf_index', 'min_num'] + E.ZC; sv = surv.merge(FT[['player_id', 'yr'] + cov], left_on=['player_id', 'first'], right_on=['player_id', 'yr']).merge(HZ.rename(columns={'hazard': 'hz_first'}), on=['player_id', 'yr'], how='left')
tr, te = sv[sv['first'] <= 2019], sv[sv['first'] >= 2020]; mu, sd = tr[cov].mean(), tr[cov].std()
cph = CoxPHFitter(penalizer=0.1).fit(pd.concat([(tr[cov] - mu) / sd, tr[['dur', 'event']]], axis=1), 'dur', 'event'); risk = cph.predict_partial_hazard((te[cov] - mu) / sd)
c_cox = concordance_index(te['dur'], -risk, te['event']); ok = te['hz_first'].notna(); c_dt = concordance_index(te.loc[ok, 'dur'], -te.loc[ok, 'hz_first'], te.loc[ok, 'event'])
hr = np.exp(cph.params_).sort_values(); pe['dec'] = pd.qcut(pe['hazard'], 10, labels=False, duplicates='drop'); cal = pe.groupby('dec').agg(pred=('hazard', 'mean'), obs=('event', 'mean'), n=('event', 'size'))
M['survival'] = {'person_period_rows_scored': len(pe), 'discrete_time_auc_walk_forward': float(auc_pp), 'discrete_time_auc_by_season': auc_by, 'cox_train_players': len(tr), 'cox_test_players': len(te), 'cox_test_events': int(te['event'].sum()),
                 'c_index_cox_test': float(c_cox), 'c_index_discrete_time_test': float(c_dt), 'c_index_cox_train': float(cph.concordance_index_), 'top_hazard_ratios': hr.tail(4).round(3).to_dict(), 'bottom_hazard_ratios': hr.head(3).round(3).to_dict(),
                 'calibration': cal.round(4).reset_index().to_dict('records'), 'calibration_mean_abs_gap': float((cal['pred'] - cal['obs']).abs().mean())}
print({k: v for k, v in M['survival'].items() if k != 'calibration'})

# %% [markdown]
# ## 3. Conformal prediction (out of course): how long must the list be?
# Split-conformal. **(A) Movers.** Score = hazard. On a calibration season, find the threshold that the hazards of real movers exceed with probability 1 - alpha; next season, list everyone above it. The promise: the list catches a real mover with probability >= 1 - alpha. **(B) Replacement events.** Score = rank of the real signing as a share of the pool. The calibration quantile gives a list length k with the same promise for the next event. Both are checked on later seasons.

# %%
def cq(s, alpha): s = np.sort(np.asarray(s)); k = int(np.ceil((len(s) + 1) * (1 - alpha))); return s[min(k, len(s)) - 1]
ALPHAS = (0.5, 0.3, 0.2, 0.1); conf_a, conf_b = [], []
for a in ALPHAS:
    cov_, size_, n_m = [], [], 0
    for Y in range(2020, 2024):                       # calibrate on season Y-2 (its 12-month outcomes are known when season Y ends), list for season Y
        cal_, tst = pe[(pe['yr'] == Y - 2) & (pe['event'] == 1)], pe[pe['yr'] == Y]; thr = -cq(-cal_['hazard'], a); inl = tst['hazard'] >= thr
        cov_.append(inl[tst['event'] == 1].sum()); n_m += int((tst['event'] == 1).sum()); size_.append(inl.mean())
    conf_a.append({'alpha': a, 'target': 1 - a, 'coverage': float(sum(cov_) / n_m), 'movers': n_m, 'list_share_of_pool': float(np.mean(size_))})
qt = T.drop_duplicates('qid').set_index('qid')['t'].reindex(qids).to_numpy(); frac = full / n_q; calm, tstm = qt <= 2021, qt >= 2022
for a in ALPHAS:
    q = cq(frac[calm], a); conf_b.append({'alpha': a, 'target': 1 - a, 'list_share_of_pool': float(q), 'list_length_for_mean_pool': float(q * n_q[tstm].mean()), 'coverage': float((frac[tstm] <= q).mean()), 'calibration_events': int(calm.sum()), 'test_events': int(tstm.sum())})
M['conformal'] = {'movers': conf_a, 'replacement': conf_b}; display(pd.DataFrame(conf_a).round(3)); display(pd.DataFrame(conf_b).round(3))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].plot(cal['pred'], cal['obs'], 'o-'); ax[0].plot([0, cal['pred'].max()], [0, cal['pred'].max()], ':', color=GREY); ax[0].set_xlabel('predicted 12-month hazard (decile mean)'); ax[0].set_ylabel('observed move-up rate'); ax[0].set_title('Hazard calibration, walk-forward', fontsize=9)
for d, lab_ in ((conf_a, 'movers caught by the hazard list'), (conf_b, 'real signing inside the ranked list')): ax[1].plot([r['target'] for r in d], [r['coverage'] for r in d], 'o-', label=lab_)
ax[1].plot([.5, .9], [.5, .9], ':', color=GREY); ax[1].set_xlabel('promised coverage (1 - alpha)'); ax[1].set_ylabel('coverage on later seasons'); ax[1].legend(fontsize=7); ax[1].set_title('Conformal coverage check', fontsize=9)
for d, lab_ in ((conf_a, 'hazard list (share of feeders)'), (conf_b, 'replacement list (share of pool)')): ax[2].plot([r['target'] for r in d], [r['list_share_of_pool'] for r in d], 'o-', label=lab_)
ax[2].set_xlabel('promised coverage'); ax[2].set_ylabel('list size needed, as a share of the pool'); ax[2].legend(fontsize=7); ax[2].set_title('The price of a guarantee', fontsize=9); savefig('p5_calibration_coverage')
_ = put_metrics('phase5', M)
