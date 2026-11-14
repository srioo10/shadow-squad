# %% [markdown]
# # End-sem 05. Evaluation
# Main metric: replacement recall@K and the median rank of the real signing, on unseen seasons (2020-21 to 2024-25), for events whose real signing was inside the candidate pool. Every method ranks the same set: all same-position pool players of that season. A second, much larger test asks the same of every summer signing by any five-league club.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
import engine as E
M = {}; rng = np.random.default_rng(SEED); KS = (5, 10, 25, 50); TEST = (2020, 2021, 2022, 2023, 2024)
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
SS, SS0, GBT, BY, MV, RC, RN, HZN = 'Shadow Squad (trained on all signings)', 'Shadow Squad, replacement events only', 'Shadow Squad fusion with Spark GBT (pointwise)', 'Best young performer (composite index)', 'Highest market value', 'Raw cosine similarity to the starter', 'Random order of the pool', 'Single score: hazard'
PF = pd.read_parquet('lake/pairs.parquet'); PA = pd.read_parquet('lake/pairs_all.parquet'); T = PF[PF['q_in_pool'] & PF['t'].isin(TEST)].copy(); qids = np.array(sorted(T['qid'].unique()))
def true_rank(df, col, asc=False, idx=None):
    """Rank of the best-placed real signing in each query (ties get the average rank; missing scores go last)."""
    s = df[col].fillna(-np.inf if not asc else np.inf); r = s.groupby(df['qid']).rank(ascending=asc, method='average'); return r[df['y'] == 1].groupby(df['qid']).min().reindex(qids if idx is None else idx)
rd = lambda f: pd.read_parquet(f'lake/{f}.parquet'); sub = lambda R: R[R['qid'].isin(qids)]
ranks = {BY: true_rank(T, 'composite_perf_index'), RC: true_rank(T, 'sim_raw'), MV: true_rank(T, 'value'), SS: true_rank(sub(rd('scores_lgbm_v2')), 'score'), SS0: true_rank(sub(rd('scores_lgbm')), 'score'), GBT: true_rank(sub(rd('scores_gbt')), 'score')}
for name, (drop, swap) in E.ABLATIONS.items():
    feats = [swap if f == drop else f for f in E.FEATS if not (f == drop and swap is None)]; R = E.walk_forward(PF, feats, TEST, train_pairs=PA, use_candidates=False); ranks['Ablation: ' + name] = true_rank(sub(R), 'score')
for name, feats in (('Single score: translated similarity', 'sim_translated'), (HZN, 'hazard'), ('Single score: value gap to the starter', 'value_gap')): ranks[name] = true_rank(T, feats, asc=(feats == 'value_gap'))
n_q = T.groupby('qid').size().reindex(qids).to_numpy(); n_pos = T.groupby('qid')['y'].sum().reindex(qids).to_numpy()
sim = np.array([[np.sort(rng.permutation(n)[:p])[0] + 1 for n, p in zip(n_q, n_pos)] for _ in range(2000)])      # 2,000 random orderings of every pool

# %% [markdown]
# ## 1. Master table: recall@K, median rank, bootstrap intervals over events, permutation tests

# %%
def perm_p(a, b, n=20000):   # paired sign-flip test on per-event differences, two-sided
    d = a - b; obs = d.mean(); sg = rng.choice([-1, 1], (n, len(d))); return float(((np.abs((sg * d).mean(1)) >= abs(obs) - 1e-12).sum() + 1) / (n + 1))
def table(ranks, sim, base_name, mv_name=None):
    nq_ = len(next(iter(ranks.values()))); bs = rng.integers(0, nq_, (4000, nq_)); base = ranks[base_name].to_numpy()
    rows = [{'method': RN, 'median_rank': float(np.median(sim)), **{f'recall@{k}': float((sim <= k).mean()) for k in KS}, **{f'recall@{k}_lo': float(np.percentile((sim <= k).mean(1), 2.5)) for k in KS}, **{f'recall@{k}_hi': float(np.percentile((sim <= k).mean(1), 97.5)) for k in KS}, 'MRR': float((1 / sim).mean())}]
    for name, r in ranks.items():
        r = r.to_numpy(); row = {'method': name, 'median_rank': float(np.median(r)), 'median_rank_lo': float(np.percentile(np.median(r[bs], 1), 2.5)), 'median_rank_hi': float(np.percentile(np.median(r[bs], 1), 97.5)), 'MRR': float((1 / r).mean())}
        for k in KS:
            h = (r <= k).astype(float); row[f'recall@{k}'] = float(h.mean()); row[f'recall@{k}_lo'], row[f'recall@{k}_hi'] = [float(x) for x in np.percentile(h[bs].mean(1), [2.5, 97.5])]
            if name != base_name: row[f'p_vs_best_young@{k}'] = perm_p(h, (base <= k).astype(float))
            if mv_name and name != mv_name: row[f'p_vs_market_value@{k}'] = perm_p(h, (ranks[mv_name].to_numpy() <= k).astype(float))
        if name != base_name: row['p_vs_best_young_MRR'] = perm_p(1 / r, 1 / base)
        rows.append(row)
    return pd.DataFrame(rows), bs
MT, bs = table(ranks, sim, BY, MV); MT.to_csv('outputs/master_results.csv', index=False)
pd.set_option('display.width', 250); display(MT[['method', 'median_rank'] + [f'recall@{k}' for k in KS] + ['MRR', 'p_vs_best_young@10', 'p_vs_market_value@10']].round(3))
full = ranks[SS].to_numpy(); base = ranks[BY].to_numpy(); mvr = ranks[MV].to_numpy(); dq = T.drop_duplicates('qid').set_index('qid').reindex(qids); qt_, qp_ = dq['t'].to_numpy(), dq['P'].to_numpy()
dci = lambda a, b, k: [float(x) for x in np.percentile(((a <= k).astype(float)[bs] - (b <= k).astype(float)[bs]).mean(1), [2.5, 97.5])]
M['main'] = {'test_queries': len(qids), 'test_seasons': list(TEST), 'mean_pool_size': float(n_q.mean()), 'queries_by_season': T.drop_duplicates('qid').groupby('t').size().to_dict(), 'table': MT.round(4).to_dict('records'),
             'full_minus_best_young_recall10_ci95': dci(full, base, 10), 'full_minus_best_young_recall25_ci95': dci(full, base, 25), 'full_minus_market_value_recall10_ci95': dci(full, mvr, 10), 'full_minus_market_value_recall25_ci95': dci(full, mvr, 25),
             'recall10_by_season_full': {int(t): float((full[qt_ == t] <= 10).mean()) for t in TEST}, 'recall10_by_position_full': {p: [int((qp_ == p).sum()), float((full[qp_ == p] <= 10).mean())] for p in sorted(set(qp_))}}
LAB = {RN: 'Random', MV: 'Market value', RC: 'Raw cosine', BY: 'Best young performer', HZN: 'Hazard', SS: 'Shadow Squad'}; S = MT.set_index('method')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
for mth, lab_ in LAB.items():
    y = [S.loc[mth, f'recall@{k}'] for k in KS]; ax[0].plot(KS, y, 'o-', color=MCOL[lab_], label=lab_, lw=2.6 if lab_ == 'Shadow Squad' else 1.4, zorder=3 if lab_ == 'Shadow Squad' else 2)
    if lab_ in ('Shadow Squad', 'Market value'): ax[0].fill_between(KS, [S.loc[mth, f'recall@{k}_lo'] for k in KS], [S.loc[mth, f'recall@{k}_hi'] for k in KS], color=MCOL[lab_], alpha=.10, lw=0)
ax[0].set_xticks(KS); ax[0].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[0].set_xlabel('K (length of the list)'); ax[0].set_ylabel('events with the real signing in the top K'); ax[0].legend(fontsize=7.5); ax[0].set_title(f'Replacement recall@K, {len(qids)} unseen events (bands: 95% bootstrap)')
ab = [SS] + ['Ablation: ' + a for a in E.ABLATIONS]; y = [S.loc[a, 'recall@10'] for a in ab]
ax[1].bar(range(len(ab)), y, width=.55, yerr=[[v - S.loc[a, 'recall@10_lo'] for a, v in zip(ab, y)], [S.loc[a, 'recall@10_hi'] - v for a, v in zip(ab, y)]], error_kw={'lw': 1, 'ecolor': INK2, 'capsize': 3}, color=[ORANGE] + [GREY] * 4)
for i, v in enumerate(y): ax[1].text(i + .3, v, f'{v:.0%}', va='center', fontsize=8)
ax[1].set_xticks(range(len(ab)), ['all scores'] + list(E.ABLATIONS), fontsize=7.5); ax[1].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[1].set_ylabel('recall@10'); ax[1].set_title('Ablation: one score removed at a time'); ax[1].grid(axis='x', visible=False); savefig('p5_recall_ablation')

# %% [markdown]
# ## 1b. Cheap against expensive signings
# Hypothesis from the first round: market value is hard to beat for the big-money signings, so the engine should earn its keep on the cheap ones. Events are split at the median fee of the real signing (events with a recorded fee above zero); free and undisclosed moves are a third group.

# %%
EV = pd.read_parquet('lake/replacement_events'); EV['qid'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str); fee = EV[EV['N_in_pool']].groupby('qid')['fee'].max().reindex(qids).to_numpy()
def tiers(fee_arr, r_eng, r_mv, r_by=None):
    med = float(np.nanmedian(fee_arr[fee_arr > 0])); out = {'median_fee_eur': med}
    for name, m in (('below median fee', (fee_arr > 0) & (fee_arr <= med)), ('above median fee', fee_arr > med), ('free or undisclosed', ~(fee_arr > 0))):
        e, v = r_eng[m], r_mv[m]; b_ = rng.integers(0, m.sum(), (4000, m.sum())); row = {'events': int(m.sum())}
        for lab_, r in (('engine', e), ('market_value', v)) + ((('best_young', r_by[m]),) if r_by is not None else ()):
            row[lab_] = {'recall@10': float((r <= 10).mean()), 'recall@10_ci95': [float(x) for x in np.percentile((r[b_] <= 10).mean(1), [2.5, 97.5])], 'recall@25': float((r <= 25).mean()), 'median_rank': float(np.median(r)), 'MRR': float((1 / r).mean())}
        for k in (10, 25): row[f'engine_minus_market@{k}'] = float((e <= k).mean() - (v <= k).mean()); row[f'engine_minus_market@{k}_ci95'] = [float(x) for x in np.percentile(((e <= k).astype(float)[b_] - (v <= k).astype(float)[b_]).mean(1), [2.5, 97.5])]; row[f'p@{k}'] = perm_p((e <= k).astype(float), (v <= k).astype(float))
        row['p_MRR'] = perm_p(1 / e, 1 / v); out[name] = row
    return out
M['fee_tiers_replacement'] = tiers(fee, full, mvr, base)
# how much does one model move from seed to seed? Ten single LambdaMART models, each on its own 80% row subsample; the engine is the average of ten
med_f = np.nanmedian(fee[fee > 0]); lo_m, hi_m = (fee > 0) & (fee <= med_f), fee > med_f; sr = []
for s_ in range(10):
    r_ = true_rank(sub(E.walk_forward(PF, E.FEATS, TEST, train_pairs=PA, use_candidates=False, n_models=1, seed=1000 + s_)), 'score').to_numpy()
    sr.append({'recall@10': (r_ <= 10).mean(), 'recall@25': (r_ <= 25).mean(), 'median_rank': np.median(r_), 'cheap_recall@10': (r_[lo_m] <= 10).mean(), 'dear_recall@10': (r_[hi_m] <= 10).mean()})
SR = pd.DataFrame(sr); M['seed_spread'] = {c: {'mean': float(SR[c].mean()), 'sd': float(SR[c].std()), 'min': float(SR[c].min()), 'max': float(SR[c].max())} for c in SR.columns}; print('single-model spread:', {c: (round(v['min'], 3), round(v['max'], 3)) for c, v in M['seed_spread'].items()})

# %% [markdown]
# ## 1c. The large test: every summer signing by a five-league club
# Query = (club, season, position); candidates = every 900-minute player of that position at another club; the reference profile is what the club already fields there. The model is the same LambdaMART, trained walk-forward on earlier signings.

# %%
SA = pd.read_parquet('lake/signing_events'); TA_ = PA[PA['t'].isin(TEST)]; qa = np.array(sorted(TA_['qid'].unique())); LA = rd('scores_all_signings')
ra = {'Best young performer (composite index)': true_rank(TA_, 'composite_perf_index', idx=qa), 'Raw cosine similarity to the club profile': true_rank(TA_, 'sim_raw', idx=qa), MV: true_rank(TA_, 'value', idx=qa), HZN: true_rank(TA_, 'hazard', idx=qa), 'Shadow Squad (LambdaMART)': true_rank(LA[LA['qid'].isin(qa)], 'score', idx=qa)}
na = TA_.groupby('qid').size().reindex(qa).to_numpy(); npa = TA_.groupby('qid')['y'].sum().reindex(qa).to_numpy(); sima = np.array([[int(rng.integers(0, n - p + 1)) // max(p, 1) + 1 if p > 1 else int(rng.integers(1, n + 1)) for n, p in zip(na, npa)] for _ in range(300)])
MA, bsa = table(ra, sima, 'Best young performer (composite index)', MV); MA.to_csv('outputs/master_results_all_signings.csv', index=False); display(MA[['method', 'median_rank'] + [f'recall@{k}' for k in KS] + ['MRR', 'p_vs_market_value@10']].round(3))
qmeta = SA.groupby('qid').agg(elite_rep=('elite_replacement', 'max'), elite_club=('C_elite', 'max'), fee=('fee', 'max')).reindex(qa); ea, va, ba = ra['Shadow Squad (LambdaMART)'].to_numpy(), ra[MV].to_numpy(), ra['Best young performer (composite index)'].to_numpy(); subsets = {}
for name, m in (('all signings', np.ones(len(qa), bool)), ('bought by an elite club', qmeta['elite_club'].to_numpy(bool)), ('elite replacements', qmeta['elite_rep'].to_numpy(bool)), ('bought by a non-elite club', ~qmeta['elite_club'].to_numpy(bool))):
    b_ = rng.integers(0, m.sum(), (4000, m.sum())); subsets[name] = {'events': int(m.sum()), **{lab_: {'recall@10': float((r[m] <= 10).mean()), 'recall@25': float((r[m] <= 25).mean()), 'median_rank': float(np.median(r[m]))} for lab_, r in (('engine', ea), ('market_value', va), ('best_young', ba))},
        'engine_minus_market@10': float((ea[m] <= 10).mean() - (va[m] <= 10).mean()), 'engine_minus_market@10_ci95': [float(x) for x in np.percentile(((ea[m] <= 10).astype(float)[b_] - (va[m] <= 10).astype(float)[b_]).mean(1), [2.5, 97.5])], 'p@10': perm_p((ea[m] <= 10).astype(float), (va[m] <= 10).astype(float)),
        'engine_minus_market@25': float((ea[m] <= 25).mean() - (va[m] <= 25).mean()), 'p@25': perm_p((ea[m] <= 25).astype(float), (va[m] <= 25).astype(float))}
M['all_signings'] = {'test_queries': len(qa), 'mean_candidates': float(na.mean()), 'table': MA.round(4).to_dict('records'), 'subsets': subsets}; M['fee_tiers_all_signings'] = tiers(qmeta['fee'].to_numpy(float), ea, va, ba)
for nm_, d in (('replacement', M['fee_tiers_replacement']), ('all signings', M['fee_tiers_all_signings'])):
    print(nm_, 'median fee', d['median_fee_eur'], {k: (v['events'], round(v['engine']['recall@10'], 3), round(v['market_value']['recall@10'], 3), fmt_p(v['p@10']), round(v['engine']['recall@25'], 3), round(v['market_value']['recall@25'], 3), fmt_p(v['p@25'])) for k, v in d.items() if k != 'median_fee_eur'})
print({k: (v['events'], round(v['engine']['recall@10'], 3), round(v['market_value']['recall@10'], 3), fmt_p(v['p@10'])) for k, v in subsets.items()})
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True); tn = ['below median fee', 'above median fee', 'free or undisclosed']
for j, (d, ttl) in enumerate(((M['fee_tiers_replacement'], f'Elite replacements ({len(qids)} events)'), (M['fee_tiers_all_signings'], f'All signings ({len(qa):,} events)'))):
    for i, (lab_, key) in enumerate((('Shadow Squad', 'engine'), ('Market value', 'market_value'), ('Best young performer', 'best_young'))):
        y = [d[t_][key]['recall@10'] for t_ in tn]; ax[j].bar(np.arange(3) + (i - 1) * .26, y, width=.24, color=MCOL[lab_], label=lab_)
        for x_, v in zip(np.arange(3) + (i - 1) * .26, y): ax[j].text(x_, v + .006, f'{v:.0%}', ha='center', fontsize=7)
    ax[j].set_xticks(range(3), [f"{t_}\n(n={d[t_]['events']})" for t_ in tn], fontsize=7.5); ax[j].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[j].set_title(ttl.split(' (')[0] + f" (median fee EUR {d['median_fee_eur'] / 1e6:.0f}m)"); ax[j].grid(axis='x', visible=False)
ax[0].set_ylabel('recall@10'); ax[0].legend(fontsize=7.5); savefig('p5_fee_tiers')

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
    for Y in range(2020, 2025):                       # calibrate on season Y-2 (its 12-month outcomes are known when season Y ends), list for season Y
        cal_, tst = pe[(pe['yr'] == Y - 2) & (pe['event'] == 1)], pe[pe['yr'] == Y]; thr = -cq(-cal_['hazard'], a); inl = tst['hazard'] >= thr
        cov_.append(inl[tst['event'] == 1].sum()); n_m += int((tst['event'] == 1).sum()); size_.append(inl.mean())
    conf_a.append({'alpha': a, 'target': 1 - a, 'coverage': float(sum(cov_) / n_m), 'movers': n_m, 'list_share_of_pool': float(np.mean(size_))})
qt = T.drop_duplicates('qid').set_index('qid')['t'].reindex(qids).to_numpy(); frac = full / n_q; calm, tstm = qt <= 2021, qt >= 2022
for a in ALPHAS:
    q = cq(frac[calm], a); conf_b.append({'alpha': a, 'target': 1 - a, 'list_share_of_pool': float(q), 'list_length_for_mean_pool': float(q * n_q[tstm].mean()), 'coverage': float((frac[tstm] <= q).mean()), 'calibration_events': int(calm.sum()), 'test_events': int(tstm.sum())})
M['conformal'] = {'movers': conf_a, 'replacement': conf_b}; display(pd.DataFrame(conf_a).round(3)); display(pd.DataFrame(conf_b).round(3))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
ax[0].plot(cal['pred'], cal['obs'], 'o-', color=VIOLET); ax[0].plot([0, cal['pred'].max()], [0, cal['pred'].max()], ':', color=GREY); ax[0].set_xlabel('predicted 12-month hazard (decile mean)'); ax[0].set_ylabel('observed move-up rate'); ax[0].set_title('Hazard calibration, walk-forward', fontsize=9)
for d, lab_ in ((conf_a, 'movers caught by the hazard list'), (conf_b, 'real signing inside the ranked list')): ax[1].plot([r['target'] for r in d], [r['coverage'] for r in d], 'o-', label=lab_)
ax[1].plot([.5, .9], [.5, .9], ':', color=GREY); ax[1].set_xlabel('promised coverage (1 - alpha)'); ax[1].set_ylabel('coverage on later seasons'); ax[1].legend(fontsize=7); ax[1].set_title('Conformal coverage check', fontsize=9)
for d, lab_ in ((conf_a, 'hazard list (share of feeders)'), (conf_b, 'replacement list (share of pool)')): ax[2].plot([r['target'] for r in d], [r['list_share_of_pool'] for r in d], 'o-', label=lab_)
ax[2].set_xlabel('promised coverage'); ax[2].set_ylabel('list size needed, as a share of the pool'); ax[2].legend(fontsize=7); ax[2].set_title('The price of a guarantee', fontsize=9); savefig('p5_calibration_coverage')
_ = put_metrics('phase5', M)
