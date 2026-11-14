# %% [markdown]
# # End-sem 04B. Beyond the syllabus: three questions a club, an agent or a player would ask
# A. Is a scouting shortlist a good investment, and where does the market misprice players? B. Does moving to an elite club help a young player? C. Can a player game the model?

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np, pickle, time
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
import engine as E
lake = E.Lake(); FT = lake.F.merge(lake.HZ, on=['player_id', 'yr'], how='left'); M = {}; rng = np.random.default_rng(SEED)
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
def boot_ci(x, f=np.mean, n=4000): x = np.asarray(x); return [float(v) for v in np.percentile([f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)], [2.5, 97.5])]

# %% [markdown]
# ## A. Scouting as an investment portfolio
# Each summer a portfolio of 25 pool players is "bought" at Transfermarkt value (last valuation before the window opens on 1 June). The return is the realised fee if he was sold for a fee in the next 24 months, otherwise his market value 24 months later, relative to the purchase value.

# %%
V = pd.read_parquet('lake/tm_valuations', columns=['player_id', 'date', 'market_value_in_eur']).rename(columns={'market_value_in_eur': 'v_end'}); V['date'] = pd.to_datetime(V['date']).astype('datetime64[ns]'); V = V.dropna().sort_values('date')
U = FT[['player_id', 'yr', 'value']].copy(); U['cut'] = pd.to_datetime((U['yr'] + 3).astype(str) + '-05-31').astype('datetime64[ns]'); U = pd.merge_asof(U.sort_values('cut'), V, left_on='cut', right_on='date', by='player_id', direction='backward').drop(columns='date')
TC = pd.read_parquet('lake/tm_transfers_clean'); TC['d'] = pd.to_datetime(TC['d']); sales = TC[(TC['fee'] > 0) & ~TC['loan_out'] & ~TC['loan_return']][['player_id', 'd', 'fee']]
s = U[['player_id', 'yr']].merge(sales, on='player_id'); s = s[(s['d'] >= pd.to_datetime((s['yr'] + 1).astype(str) + '-06-01')) & (s['d'] < pd.to_datetime((s['yr'] + 3).astype(str) + '-06-01'))].sort_values('d').drop_duplicates(['player_id', 'yr'])
U = U.merge(s[['player_id', 'yr', 'fee']], on=['player_id', 'yr'], how='left'); U['sold'] = U['fee'].notna(); U['end'] = U['fee'].fillna(U['v_end']); U['ret'] = U['end'] / U['value'] - 1; U['ret_value_only'] = U['v_end'] / U['value'] - 1
FT = FT.merge(U[['player_id', 'yr', 'v_end', 'sold', 'fee', 'end', 'ret', 'ret_value_only']], on=['player_id', 'yr'], how='left'); FT['min_share'] = FT['x_starter_share']; FT['lv'] = np.log(FT['value'].where(FT['value'] > 0)); FT['age2'] = FT['age_num'] ** 2
HFORM = ' + '.join(E.ZC) + ' + age_num + age2 + C(comp, Treatment("Premier League")) + min_share + elite_club + table_pos + C(fine_pos) + C(yr)'
# hedonic score for the portfolio: fitted on seasons up to t only (prices at t are known at t)
FT['hed_resid'] = np.nan
for t in range(2018, 2025):
    fit = smf.ols('lv ~ ' + HFORM, FT[(FT['yr'] <= t) & FT['lv'].notna()]).fit(); cur = (FT['yr'] == t) & FT['lv'].notna(); FT.loc[cur, 'hed_resid'] = FT.loc[cur, 'lv'] - fit.predict(FT[cur])
SEAS = (2020, 2021, 2022, 2023); pool = lake.pool[['player_id', 'yr']].merge(FT, on=['player_id', 'yr']); pool = pool[pool['yr'].isin(SEAS) & (pool['value'] > 0) & pool['ret'].notna()].copy()
LV = pd.read_parquet('lake/scores_lgbm_v2.parquet'); LV['pr'] = LV.groupby('qid')['score'].rank(pct=True); eng = LV.groupby(['player_id', 't'])['pr'].mean().rename('engine').reset_index().rename(columns={'t': 'yr'})
pool = pool.merge(eng, on=['player_id', 'yr'], how='left')
PORT = {'Shadow Squad': ('engine', False), 'Best young performer': ('composite_perf_index', False), 'Market value': ('value', False), 'Hazard': ('hazard', False), 'Under-priced (hedonic)': ('hed_resid', True)}
def stats_of(d):
    r = d['ret'].to_numpy(); vw = d['end'].sum() / d['value'].sum() - 1
    bs = [d.iloc[rng.integers(0, len(d), len(d))] for _ in range(2000)]
    return {'n': len(d), 'mean_return': float(r.mean()), 'mean_return_ci': [float(x) for x in np.percentile([b['ret'].mean() for b in bs], [2.5, 97.5])], 'value_weighted_return': float(vw),
            'value_weighted_ci': [float(x) for x in np.percentile([b['end'].sum() / b['value'].sum() - 1 for b in bs], [2.5, 97.5])], 'median_return': float(np.median(r)), 'hit_rate': float((r > 0).mean()), 'sharpe': float(r.mean() / r.std()),
            'spend_eur_m': float(d['value'].sum() / 1e6), 'sold_with_fee': int(d['sold'].sum()), 'realised_fee_return_sold_only': float((d.loc[d['sold'], 'fee'] / d.loc[d['sold'], 'value'] - 1).mean()) if d['sold'].any() else None,
            'value_only_return': float(d['ret_value_only'].mean())}
res, picks = {}, {}
for name, (col, asc) in PORT.items():
    picks[name] = pd.concat([g.dropna(subset=[col]).sort_values(col, ascending=asc).head(25) for _, g in pool.groupby('yr')]); res[name] = stats_of(picks[name])
groups = [g for _, g in pool.groupby('yr')]; null_m, null_vw = np.empty(10000), np.empty(10000)
for i in range(10000):
    d = pd.concat([g.iloc[rng.choice(len(g), 25, replace=False)][['ret', 'end', 'value']] for g in groups]); null_m[i] = d['ret'].mean(); null_vw[i] = d['end'].sum() / d['value'].sum() - 1
res['Random'] = {'n': 25 * len(groups), 'mean_return': float(null_m.mean()), 'mean_return_ci': [float(x) for x in np.percentile(null_m, [2.5, 97.5])], 'value_weighted_return': float(null_vw.mean()), 'value_weighted_ci': [float(x) for x in np.percentile(null_vw, [2.5, 97.5])],
                 'hit_rate': float((pool['ret'] > 0).mean()), 'sharpe': float(pool['ret'].mean() / pool['ret'].std()), 'median_return': float(pool['ret'].median())}
for name in PORT: res[name]['p_vs_random_mean'] = float((null_m >= res[name]['mean_return']).mean()); res[name]['p_vs_random_value_weighted'] = float((null_vw >= res[name]['value_weighted_return']).mean())
PT = pd.DataFrame(res).T; display(PT[['n', 'mean_return', 'value_weighted_return', 'median_return', 'hit_rate', 'sharpe', 'p_vs_random_mean', 'sold_with_fee', 'realised_fee_return_sold_only']].astype(float).round(3))
ex = picks['Shadow Squad'].nlargest(5, 'ret')[['player', 'squad', 'yr', 'value', 'end', 'ret']]; display(ex.round(2))

# %% [markdown]
# ### Hedonic price model: what the market pays for, and what it under-prices
# Model 1 regresses log market value on the style-neutral profile, age, league, minutes and club tier: the coefficients are today's prices of attributes. Model 2 regresses the next-24-month log return on the same attributes plus today's value: a positive coefficient means the market under-prices that attribute today. Standard errors are clustered by player.

# %%
H1 = FT[FT['lv'].notna()]; m1 = smf.ols('lv ~ ' + HFORM, H1).fit(cov_type='cluster', cov_kwds={'groups': H1['player_id']})
H2 = FT[FT['lv'].notna() & (FT['end'] > 0) & (FT['yr'] <= 2023)].assign(lret=lambda d: np.log(d['end'] / d['value'])); m2 = smf.ols('lret ~ lv + ' + HFORM, H2).fit(cov_type='cluster', cov_kwds={'groups': H2['player_id']})
def coefs(m, pat): c = m.conf_int(); return {k.split('T.')[-1].rstrip(']') if 'T.' in k else k: {'coef': float(m.params[k]), 'lo': float(c.loc[k, 0]), 'hi': float(c.loc[k, 1]), 'p': float(m.pvalues[k])} for k in m.params.index if pat in k}
lg1, lg2 = coefs(m1, 'comp'), coefs(m2, 'comp'); z1, z2 = coefs(m1, 'z_'), coefs(m2, 'z_')
rq = pool.dropna(subset=['hed_resid']).copy(); rq['q'] = pd.qcut(rq['hed_resid'], 5, labels=False) + 1; byq = rq.groupby('q')['ret'].agg(['mean', 'median', 'size']); rho_r, p_r = stats.spearmanr(rq['hed_resid'], rq['ret'])
under = sorted(z2.items(), key=lambda kv: -kv[1]['coef']); age_r = {k: {'coef': float(m2.params[k]), 'p': float(m2.pvalues[k])} for k in ('age_num', 'age2', 'lv', 'min_share')}
M['A'] = {'seasons': list(SEAS), 'pool_rows': len(pool), 'pool_mean_return': float(pool['ret'].mean()), 'portfolios': res, 'null_draws': 10000, 'top_engine_picks': ex.round(3).to_dict('records'),
          'hedonic_value': {'n': int(m1.nobs), 'r2': float(m1.rsquared), 'league': lg1, 'metrics': z1, 'age': {k: float(m1.params[k]) for k in ('age_num', 'age2')}, 'peak_value_age': float(-m1.params['age_num'] / (2 * m1.params['age2']))},
          'hedonic_return': {'n': int(m2.nobs), 'r2': float(m2.rsquared), 'league': lg2, 'metrics': z2, 'other': age_r, 'most_underpriced_metrics': [(k, round(v['coef'], 4), round(v['p'], 4)) for k, v in under[:4]], 'most_overpriced_metrics': [(k, round(v['coef'], 4), round(v['p'], 4)) for k, v in under[-4:]]},
          'residual_vs_return': {'spearman': float(rho_r), 'p': float(p_r), 'by_quintile_mean': byq['mean'].round(4).to_dict(), 'by_quintile_median': byq['median'].round(4).to_dict()}}
print({k: (round(v['coef'], 3), fmt_p(v['p'])) for k, v in lg1.items()}, '| return model:', {k: (round(v['coef'], 3), fmt_p(v['p'])) for k, v in lg2.items()}); print('R2', round(m1.rsquared, 3), round(m2.rsquared, 3), '| residual vs return', round(rho_r, 3), fmt_p(p_r), byq['mean'].round(3).to_dict()); print('under-priced:', M['A']['hedonic_return']['most_underpriced_metrics'], 'over-priced:', M['A']['hedonic_return']['most_overpriced_metrics'])
order = ['Shadow Squad', 'Under-priced (hedonic)', 'Hazard', 'Best young performer', 'Market value']; fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for j, (key, cik, ttl, nul) in enumerate((('mean_return', 'mean_return_ci', 'Mean return per player', null_m), ('value_weighted_return', 'value_weighted_ci', 'Return on the money spent (value-weighted)', null_vw))):
    y = [res[o][key] for o in order]; lo = [res[o][key] - res[o][cik][0] for o in order]; hi = [res[o][cik][1] - res[o][key] for o in order]; lo_n, hi_n = np.percentile(nul, [2.5, 97.5])
    ax[j].axhspan(lo_n, hi_n, color=GREY, alpha=.18, lw=0, label='95% of 10,000 random portfolios'); ax[j].axhline(nul.mean(), color=GREY, lw=1)
    ax[j].bar(range(len(order)), y, width=.55, color=[MCOL[o] for o in order], yerr=[lo, hi], error_kw={'lw': 1, 'ecolor': INK2, 'capsize': 3})
    for i, v in enumerate(y): ax[j].text(i + .31, v, f'{v:+.0%}', va='center', fontsize=8, color=INK)
    ax[j].set_xticks(range(len(order)), [o.replace(' (', '\n(').replace('Best young ', 'Best young\n') for o in order], fontsize=7.5); ax[j].yaxis.set_major_formatter(lambda v, _: f'{v:+.0%}'); ax[j].set_title(ttl); ax[j].legend(loc='upper right', fontsize=7); ax[j].grid(axis='x', visible=False)
fig.suptitle(f'Top-25 portfolios bought each summer {SEAS[0]}-{SEAS[-1]}, held 24 months (bars: 95% bootstrap interval)', x=0.01, ha='left', fontsize=10, fontweight='bold'); savefig('p4b_portfolio')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1, 1.5]}); L4 = list(lg1)
for i, (d, lab_, c) in enumerate(((lg1, 'price today (value model)', BLUE), (lg2, 'return over the next 24 months', ORANGE))):
    ax[0].errorbar([d[k]['coef'] for k in L4], np.arange(len(L4)) + (i - .5) * .28, xerr=[[d[k]['coef'] - d[k]['lo'] for k in L4], [d[k]['hi'] - d[k]['coef'] for k in L4]], fmt='o', color=c, label=lab_, capsize=2, lw=1.2)
ax[0].axvline(0, color=INK2, lw=.8); ax[0].set_yticks(range(len(L4)), L4); ax[0].set_xlabel('log difference from the Premier League, like-for-like player'); ax[0].set_title('League: priced lower, and does it come back?'); ax[0].legend(fontsize=7, loc='lower left'); ax[0].grid(axis='y', visible=False)
ms = [k for k, _ in under]
for i, (d, lab_, c) in enumerate(((z1, 'price today, per SD of the metric', BLUE), (z2, 'return over the next 24 months', ORANGE))):
    ax[1].errorbar(np.arange(len(ms)) + (i - .5) * .28, [d[k]['coef'] for k in ms], yerr=[[d[k]['coef'] - d[k]['lo'] for k in ms], [d[k]['hi'] - d[k]['coef'] for k in ms]], fmt='o', color=c, label=lab_, capsize=2, lw=1.2)
ax[1].axhline(0, color=INK2, lw=.8); ax[1].set_xticks(range(len(ms)), [k[2:] for k in ms], rotation=50, fontsize=7.5); ax[1].set_title('Metrics: what the market pays for, and what it under-prices'); ax[1].legend(fontsize=7); ax[1].grid(axis='x', visible=False); savefig('p4b_hedonic')

# %% [markdown]
# ## B. Sliding doors: does moving to an elite club help a young player?
# Treated: a feeder (24 or under, non-elite club, 900+ minutes in season t) who makes a permanent move to a club that is elite in season t+1. Control: a feeder with no move of any kind that year. Twins are proposed by the Spark LSH index (same fine position and season, nearest in age, style-neutral profile, minutes, club table position and market value), then matched one-to-one on a propensity score. Outcomes two seasons after t: share of league minutes played, and the composite index of his **style-neutral** per-90 output, each as a change from season t.

# %%
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, BucketedRandomProjectionLSH
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import CausalForestDML
sp = spark()
CM = pd.read_parquet('lake/tm_club_minutes'); tot = CM.groupby(['player_id', 'season'])['minutes'].sum(); el = FT.groupby(['club_id', 'yr'])['elite_club'].first()
D = FT[FT['feeder'] & (FT['yr'] <= 2022) & FT['value'].gt(0)].copy(); mv = D[['player_id', 'yr']].merge(TC[['player_id', 'd', 'c', 'loan_out', 'loan_return']], on='player_id')
mv = mv[(mv['d'] >= pd.to_datetime((mv['yr'] + 1).astype(str) + '-06-01')) & (mv['d'] < pd.to_datetime((mv['yr'] + 2).astype(str) + '-06-01'))]; mv['to_elite'] = [bool(el.get((c, y + 1), False)) for c, y in zip(mv['c'], mv['yr'])]
early = mv[~mv['loan_out'] & ~mv['loan_return'] & mv['to_elite'] & (mv['d'] < pd.to_datetime((mv['yr'] + 2).astype(str) + '-02-01'))].drop_duplicates(['player_id', 'yr'])
D = D.merge(early[['player_id', 'yr', 'c']].rename(columns={'c': 'buyer'}), on=['player_id', 'yr'], how='left'); D['T'] = D['buyer'].notna().astype(int); anym = set(zip(mv['player_id'], mv['yr']))
D['stayed'] = [((p, y) not in anym) and (tot.get((p, y + 1), 0) > 0) for p, y in zip(D['player_id'], D['yr'])]; D = D[(D['T'] == 1) | D['stayed']].reset_index(drop=True)
D['share2'] = [min(tot.get((p, y + 2), 0) / 3420.0, 1.0) for p, y in zip(D['player_id'], D['yr'])]; D['d_min'] = D['share2'] - D['min_share'].clip(upper=1.0)
nxt = FT[['player_id', 'yr', 'composite_perf_index']].assign(yr=lambda d: d['yr'] - 2).rename(columns={'composite_perf_index': 'comp2'}); D = D.merge(nxt, on=['player_id', 'yr'], how='left'); D['d_out'] = D['comp2'] - D['composite_perf_index']
D['buyer_dom'] = [lake.STY.set_index(['club_id', 'yr'])['x_dom'].get((b, y + 1), np.nan) if pd.notna(b) else np.nan for b, y in zip(D['buyer'], D['yr'])]
COV = ['age_num', 'min_share', 'composite_perf_index', 'table_pos', 'lv'] + E.ZC; D[COV] = D[COV].fillna(D[COV].median()); Zs = (D[COV] - D[COV].mean()) / D[COV].std(); D['uid'] = np.arange(len(D))
sdf = VectorAssembler(inputCols=COV, outputCol='v').transform(sp.createDataFrame(pd.concat([D[['uid', 'T', 'yr', 'fine_pos']], Zs], axis=1))).select('uid', 'T', 'yr', 'fine_pos', 'v').cache()
t0 = time.time(); lm = BucketedRandomProjectionLSH(inputCol='v', outputCol='h', bucketLength=3.0, numHashTables=8, seed=SEED).fit(sdf)
tw = (lm.approxSimilarityJoin(sdf.filter('T = 1'), sdf.filter('T = 0'), 1e6, 'dist').filter('datasetA.fine_pos = datasetB.fine_pos and datasetA.yr = datasetB.yr').select(F.col('datasetA.uid').alias('a'), F.col('datasetB.uid').alias('b'), 'dist')).toPandas(); lsh_s = time.time() - t0
tw['r'] = tw.groupby('a')['dist'].rank(method='first'); Du = D.set_index('uid'); smd = lambda x1, x0: float((x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2))
def match(topk, calf, repl):
    """Propensity-score matching inside the LSH twin candidates. Returns the pairs and the worst covariate imbalance."""
    c_ = tw[tw['r'] <= topk]; S_ = pd.concat([D[D['T'] == 1], D[D['uid'].isin(c_['b'])]]); Zm = (S_[COV] - S_[COV].mean()) / S_[COV].std()
    lg_ = pd.Series(LogisticRegression(C=1.0, max_iter=2000).fit(Zm, S_['T']).decision_function(Zm), index=S_['uid'].to_numpy()); cal = calf * lg_.std(); used, pairs = set(), []
    for a in lg_[S_.loc[S_['T'] == 1, 'uid']].sort_values(ascending=False).index:
        c = c_[c_['a'] == a]; c = c if repl else c[~c['b'].isin(used)]
        if len(c):
            dlt = lg_.reindex(c['b']).to_numpy() - lg_[a]; j = int(np.argmin(np.abs(dlt)))
            if abs(dlt[j]) <= cal: used.add(int(c['b'].iloc[j])); pairs.append((a, int(c['b'].iloc[j])))
    pr = pd.DataFrame(pairs, columns=['a', 'b']); worst = max(abs(smd(Du.loc[pr['a'], c], Du.loc[pr['b'], c])) for c in COV) if len(pr) > 20 else 9.0
    return pr, worst, float(cal)
grid = [{'twins_per_treated': k, 'caliper_sd': c, 'with_replacement': r} for k in (10, 25, 50) for c in (0.05, 0.1, 0.2) for r in (False, True)]      # the setting is chosen on covariate balance alone; outcomes are not looked at
for g in grid: g['pr'], g['worst_smd'], g['cal'] = match(g['twins_per_treated'], g['caliper_sd'], g['with_replacement']); g['pairs'] = len(g['pr'])
best_g = min([g for g in grid if g['pairs'] >= 100], key=lambda g: g['worst_smd']); PR_, cal = best_g['pr'], best_g['cal']; TA_, CB_ = Du.loc[PR_['a']], Du.loc[PR_['b']]
BAL = pd.DataFrame({'before': {c: smd(D.loc[D['T'] == 1, c], D.loc[D['T'] == 0, c]) for c in COV}, 'after': {c: smd(TA_[c], CB_[c]) for c in COV}})
def att(col):
    d = TA_[col].to_numpy() - CB_[col].to_numpy(); d = d[~np.isnan(d)]; ci_ = boot_ci(d); sd0 = float(np.nanstd(D.loc[D['T'] == 0, col])); dz = d.mean() / sd0
    def ev(x): rr = np.exp(0.91 * abs(x)); return float(rr + np.sqrt(rr * (rr - 1)))
    near = 0.0 if ci_[0] <= 0 <= ci_[1] else min(abs(ci_[0]), abs(ci_[1])) / sd0
    return {'pairs': int(len(d)), 'att': float(d.mean()), 'ci95': ci_, 'p_sign_flip': float((np.abs((rng.choice([-1, 1], (20000, len(d))) * d).mean(1)) >= abs(d.mean())).mean()), 'treated_mean_change': float(TA_[col].mean()), 'twin_mean_change': float(CB_[col].mean()),
            'std_effect': float(dz), 'e_value': ev(dz), 'e_value_ci': (1.0 if near == 0 else ev(near))}
RB = {'minutes_share': att('d_min'), 'neutral_output_index': att('d_out')}
ag = lambda a: np.where(a <= 21, '21 or under', np.where(a <= 22, '22', '23-24')); TA_ = TA_.assign(grp=ag(TA_['age_num'].to_numpy())); sub = {}
for gname, msk in [(g, (TA_['grp'] == g).to_numpy()) for g in ('21 or under', '22', '23-24')] + [(g, (TA_['pos_group'] == g).to_numpy()) for g in ('DF', 'MF', 'FW')] + [('buyer less dominant', (TA_['buyer_dom'] <= TA_['buyer_dom'].median()).to_numpy()), ('buyer more dominant', (TA_['buyer_dom'] > TA_['buyer_dom'].median()).to_numpy())]:
    for col in ('d_min', 'd_out'):
        d = (TA_[col].to_numpy() - CB_[col].to_numpy())[msk]; d = d[~np.isnan(d)]
        if len(d) >= 8: sub[f'{gname} | {col}'] = {'pairs': int(len(d)), 'att': float(d.mean()), 'ci95': boot_ci(d)}
# causal forest on everyone (treated + all stayers), for the heterogeneity question
Xh = pd.concat([D[['age_num', 'min_share', 'composite_perf_index', 'table_pos', 'lv']], pd.get_dummies(D['pos_group']).astype(float)], axis=1); Wc = D[E.ZC]
cf = CausalForestDML(model_y=GradientBoostingRegressor(random_state=SEED), model_t=GradientBoostingClassifier(random_state=SEED), discrete_treatment=True, n_estimators=600, min_samples_leaf=15, cv=5, random_state=SEED).fit(D['d_min'], D['T'], X=Xh, W=Wc)
CF = {}
for gname, msk in [('all', np.ones(len(D), bool))] + [(g, ag(D['age_num'].to_numpy()) == g) for g in ('21 or under', '22', '23-24')] + [(g, (D['pos_group'] == g).to_numpy()) for g in ('DF', 'MF', 'FW')]:
    inf = cf.ate_inference(Xh[msk]); CF[gname] = {'n': int(msk.sum()), 'treated': int(D.loc[msk, 'T'].sum()), 'cate': float(inf.mean_point), 'ci95': [float(x) for x in inf.conf_int_mean()], 'p': float(inf.pvalue())}
fi = dict(zip(Xh.columns, map(float, cf.feature_importances_)))
M['B'] = {'feeders': len(D), 'treated': int(D['T'].sum()), 'stayers': int((D['T'] == 0).sum()), 'lsh_candidate_pairs': len(tw), 'matching_grid': [{k: v for k, v in g.items() if k not in ('pr', 'cal')} for g in grid], 'matching_chosen': {k: v for k, v in best_g.items() if k not in ('pr', 'cal')}, 'distinct_twins': int(PR_['b'].nunique()), 'lsh_seconds': round(lsh_s, 2), 'matched_pairs': len(PR_), 'caliper_logit': float(cal), 'balance': BAL.round(4).to_dict(), 'max_abs_smd_before': float(BAL['before'].abs().max()),
          'max_abs_smd_after': float(BAL['after'].abs().max()), 'covariates_over_0.1_after': BAL.index[BAL['after'].abs() > 0.1].tolist(), 'att': RB, 'subgroups_matched': sub, 'causal_forest_minutes': CF, 'forest_feature_importance': fi,
          'treated_minutes_share_t': float(TA_['min_share'].mean()), 'treated_minutes_share_t2': float(TA_['share2'].mean()), 'twin_minutes_share_t2': float(CB_['share2'].mean()), 'output_observed_pairs_share': float(RB['neutral_output_index']['pairs'] / len(PR_))}
print({k: v for k, v in M['B'].items() if k in ('feeders', 'treated', 'matched_pairs', 'matching_chosen', 'distinct_twins', 'max_abs_smd_before', 'max_abs_smd_after', 'covariates_over_0.1_after', 'att')}); print({k: (round(v['cate'], 3), [round(x, 3) for x in v['ci95']]) for k, v in CF.items()}); print({k: (v['pairs'], round(v['att'], 3), [round(x, 3) for x in v['ci95']]) for k, v in sub.items()})
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={'width_ratios': [1.15, 1, 1.25]}); o = BAL['before'].abs().sort_values().index
ax[0].scatter(BAL.loc[o, 'before'].abs(), range(len(o)), color=GREY, label='before matching', s=22); ax[0].scatter(BAL.loc[o, 'after'].abs(), range(len(o)), color=BLUE, label='after matching', s=22); ax[0].axvline(0.1, color=INK2, lw=.8, ls=':')
ax[0].set_yticks(range(len(o)), [c.replace('z_', '').replace('_num', '').replace('composite_perf_index', 'composite').replace('lv', 'log value') for c in o], fontsize=6.5); ax[0].set_xlabel('|standardised mean difference|'); ax[0].set_title('Balance of the twins'); ax[0].legend(fontsize=7, loc='lower right'); ax[0].grid(axis='y', visible=False)
gs = ['all', '21 or under', '22', '23-24', 'DF', 'MF', 'FW']; y = np.arange(len(gs))[::-1]
ax[1].errorbar([CF[g]['cate'] for g in gs], y, xerr=[[CF[g]['cate'] - CF[g]['ci95'][0] for g in gs], [CF[g]['ci95'][1] - CF[g]['cate'] for g in gs]], fmt='o', color=ORANGE, capsize=2, lw=1.2); ax[1].axvline(0, color=INK2, lw=.8)
ax[1].set_yticks(y, [f"{g} (n={CF[g]['treated']})" for g in gs], fontsize=7.5); ax[1].xaxis.set_major_formatter(lambda v, _: f'{v:+.0%}'); ax[1].set_xlabel('effect on share of league minutes, t to t+2'); ax[1].set_title('Causal forest: who pays the price?'); ax[1].grid(axis='y', visible=False)
lab2 = [('minutes_share', 'share of league\nminutes'), ('neutral_output_index', 'style-neutral\noutput index / 100')]; sc2 = [1, 100]
for i, ((k, lb), sc_) in enumerate(zip(lab2, sc2)):
    r = RB[k]; ax[2].bar([i - .2, i + .2], [r['treated_mean_change'] / sc_, r['twin_mean_change'] / sc_], width=.36, color=[ORANGE, GREY]); ax[2].text(i, max(r['treated_mean_change'], r['twin_mean_change'], 0) / sc_ + .012, f"difference {r['att'] / sc_:+.2f}\n(p {fmt_p(r['p_sign_flip'])})", ha='center', fontsize=7.5)
ax[2].bar([0], [0], color=ORANGE, label='moved to an elite club'); ax[2].bar([0], [0], color=GREY, label='matched twin who stayed'); ax[2].axhline(0, color=INK2, lw=.8); ax[2].set_xticks([0, 1], [l for _, l in lab2], fontsize=7.5); ax[2].set_title('Change from season t to t+2'); ax[2].legend(fontsize=7, loc='lower left'); ax[2].grid(axis='x', visible=False); savefig('p4b_sliding_doors')

# %% [markdown]
# ## C. Red-team the scout: can a player game the model?
# Four of the sixteen metrics are volume a player controls with little regard to quality: shots, take-on attempts, clearances and progressive carries (**cheap**). The other twelve need an outcome or a team-mate (goals, npxG, assists, key passes, shot-creating actions, passes into the box, progressive passes, carries into the box, tackles won, interceptions, blocks, aerial win rate: **costly**).
# For every player ranked 11 to 20 in an unseen replacement query, a greedy search raises his cheap metrics in steps of 5% of the position's 5th-95th percentile range, never above the 95th percentile, and re-scores him through the whole pipeline (style-neutral value, standardisation, translated similarity, trajectory, composite index, hazard). He "games" the model if he enters the top 10.

# %%
models = E.load_models(); HM = pickle.load(open('lake/hazard_models.pkl', 'rb')); HFe = HM['features']; PF = pd.read_parquet('lake/pairs.parquet'); TEST = (2020, 2021, 2022, 2023, 2024); K = 10
CHEAP = ['sh', 'take_on', 'clr', 'prgc']; CI_ = [E.MET.index(m) for m in CHEAP]; COSTLY = [i for i in range(len(E.MET)) if i not in CI_]; Fm = lake.F; yr_a, fp_a, pg_a = Fm['yr'].to_numpy(), Fm['fine_pos'].to_numpy(), Fm['pos_group'].to_numpy()
BND, SDN, SRT = {}, {}, {}
for t in TEST:
    for p in np.unique(fp_a): m = (yr_a <= t) & (fp_a == p); BND[(t, p)] = np.percentile(lake.RAWV[m], [5, 95], axis=0); SDN[(t, p)] = np.log(lake.NEU[m] + E.EPS).std(0)
    for g in np.unique(pg_a): SRT[(t, g)] = np.sort(lake.NEU[(yr_a == t) & (pg_a == g)], axis=0)
HX = Fm[[c for c in HFe if c in Fm.columns]].copy()
for c in HFe:
    if c.startswith('P_'): HX[c] = (Fm['fine_pos'] == c[2:]).astype(float)
    if c.startswith('L_'): HX[c] = (Fm['comp'] == LEAGUES[int(c[2:])]).astype(float)
HXa = HX[HFe].fillna(0).to_numpy(float); zpos = [HFe.index(z) for z in E.ZC]; cpos = HFe.index('composite_perf_index')
def rescore(row, raws, model):
    """Scores of one candidate under altered raw per-90 profiles (k x 16), pushed through every profile-dependent score."""
    xi, si, t, fp, pg = lake.idx[(row.player_id, row.t)], lake.idx[(row.S, row.t)], row.t, row.P, pg_a[lake.idx[(row.player_id, row.t)]]; k = len(raws); r0, n0 = lake.RAWV[xi], lake.NEU[xi]
    own = lake.translate(np.ones((1, len(E.MET))), [pg], Fm['x_dom'].iat[xi], Fm['x_press'].iat[xi], Fm['comp'].iat[xi], t)[0]; neu = np.where(r0 > 0, n0 * raws / np.where(r0 > 0, r0, 1), raws / own)
    z = lake.Z[xi] + (np.log(neu + E.EPS) - np.log(n0 + E.EPS)) / SDN[(t, fp)]
    tr = lake.translate(neu, [pg] * k, Fm['x_dom'].iat[si], Fm['x_press'].iat[si], Fm['comp'].iat[si], t); zS = lake.zraw(lake.RAWV[[si]], [fp], t)
    srt = SRT[(t, pg)]; pct = lambda v: np.array([[np.searchsorted(srt[:, j], v[i, j], side='right') / len(srt) * 100 for j in range(len(E.MET))] for i in range(len(v))])
    comp = row.composite_perf_index + (pct(neu) - pct(n0[None, :])).mean(1)
    X = pd.DataFrame({f: np.repeat(getattr(row, f), k) for f in E.FEATS}); X['sim_translated'] = E.rowcos(lake.zraw(tr, [fp] * k, t)[:, lake.dims], np.repeat(zS, k, 0)[:, lake.dims])
    X['traj_sim'] = [lake._traj(xi, si, t, z0=z[i])[0] for i in range(k)]; X['composite_perf_index'] = comp
    if t in HM:
        mu, sd, cf_, b = HM[t]; hx = np.repeat(HXa[[xi]], k, 0); hx[:, zpos] = np.nan_to_num(z); hx[:, cpos] = comp; X['hazard'] = 1 / (1 + np.exp(-(((hx - mu) / np.where(sd > 0, sd, 1)) @ cf_ + b)))
    return model.predict(X[E.FEATS])
qs = PF[PF['t'].isin(TEST)]['qid'].drop_duplicates(); qs = qs.sample(min(150, len(qs)), random_state=SEED); recs = []; t0 = time.time()
for q in qs:
    G = PF[PF['qid'] == q].copy(); mdl = models[int(G['t'].iat[0])]; G['score'] = mdl.predict(G[E.FEATS]); G = G.sort_values('score', ascending=False).reset_index(drop=True)
    if len(G) < 2 * K: continue
    for rk in range(K, 2 * K):
        row = G.iloc[rk]; thr = G['score'].iat[K - 1]; xi = lake.idx[(row.player_id, row.t)]; cur = lake.RAWV[xi].copy(); lo, hi = BND[(row.t, row.P)]; step = (hi - lo) / 20.0; sc = row.score; nstep = 0
        while sc <= thr and nstep < 40:
            opts = [j for j in CI_ if cur[j] + step[j] <= max(hi[j], lake.RAWV[xi][j]) + 1e-9]
            if not opts: break
            var = np.repeat(cur[None, :], len(opts), 0)
            for i, j in enumerate(opts): var[i, j] += step[j]
            s_ = rescore(row, var, mdl); b_ = int(np.argmax(s_))
            if s_[b_] <= sc + 1e-12: break
            cur = var[b_]; sc = s_[b_]; nstep += 1
        d = cur - lake.RAWV[xi]; sdr = np.exp(lake.mu[(row.t, row.P)]) * lake.sd[(row.t, row.P)]
        recs.append({'qid': q, 'rank': rk + 1, 'gamed': bool(sc > thr), 'steps': nstep, 'l1_range': float((d[CI_] / (hi - lo)[CI_]).sum()), **{'d_' + m: float(d[j]) for m, j in zip(CHEAP, CI_)}, **{'rel_' + m: float(d[j] / max(lake.RAWV[xi][j], 1e-9)) for m, j in zip(CHEAP, CI_)}})
RT_ = pd.DataFrame(recs); gm = RT_[RT_['gamed']]; lever = max(CHEAP, key=lambda m: (gm['d_' + m] > 0).mean()); only = {m: float(((gm[['d_' + x for x in CHEAP if x != m]].abs().sum(1) == 0) & (gm['d_' + m] > 0)).sum() / len(RT_)) for m in CHEAP}
qb = [RT_[RT_['qid'].isin(rng.choice(RT_['qid'].unique(), RT_['qid'].nunique()))]['gamed'].mean() for _ in range(2000)]
C = {'queries': int(RT_['qid'].nunique()), 'near_miss_players': len(RT_), 'cheap': CHEAP, 'costly': [E.MET[i] for i in COSTLY], 'gameability': float(RT_['gamed'].mean()), 'gameability_ci95': [float(x) for x in np.percentile(qb, [2.5, 97.5])],
     'median_steps': float(gm['steps'].median()), 'median_change_share_of_range': float(gm['l1_range'].median()), 'most_used_metric': lever, 'share_of_gamers_using_it': float((gm['d_' + lever] > 0).mean()),
     'median_added_per90': {m: float(gm.loc[gm['d_' + m] > 0, 'd_' + m].median()) for m in CHEAP}, 'median_relative_increase': {m: float(gm.loc[gm['d_' + m] > 0, 'rel_' + m].median()) for m in CHEAP}, 'gamed_with_one_metric_alone': only,
     'gameability_by_rank': RT_.groupby('rank')['gamed'].mean().round(3).to_dict(), 'search_seconds': round(time.time() - t0, 1)}
print({k: v for k, v in C.items() if k not in ('costly', 'gameability_by_rank')})

# %% [markdown]
# ### Hardening: efficiency-only profile
# The hardened model never sees the four cheap metrics: similarities and trajectory use the twelve costly metrics, the composite index is the mean costly percentile, and the hazard is refitted on costly metrics (walk-forward). Padding a cheap metric then changes nothing, so gameability through cheap metrics is zero by construction. The question is what that costs in recall.

# %%
lakeH = E.Lake(); lakeH.dims = np.array(COSTLY); cc = Fm[[E.AF[i] + '_pct' for i in COSTLY]].mean(axis=1).round(1)
PPd = pd.read_parquet('lake/person_period.parquet')[['player_id', 'yr', 'event']]; Hc = [c for c in HFe if c not in ['z_' + m for m in CHEAP]]; HXc = HX[HFe].fillna(0).copy(); HXc['composite_perf_index'] = cc.to_numpy(); HXc['player_id'] = Fm['player_id'].to_numpy(); HXc['yr'] = yr_a; hz = []
for t in range(2018, 2025):
    trn = PPd[PPd['yr'] <= t - 1].merge(HXc, on=['player_id', 'yr']); mu, sd = trn[Hc].mean(), trn[Hc].std().replace(0, 1); lr = LogisticRegression(C=1 / (0.05 * len(trn)), max_iter=3000).fit((trn[Hc] - mu) / sd, trn['event']); cur = HXc[HXc['yr'] == t]
    hz.append(pd.DataFrame({'player_id': cur['player_id'], 'yr': t, 'hazard': lr.predict_proba((cur[Hc] - mu) / sd)[:, 1]}))
lakeH.HZ = pd.concat(hz)
for fr in (lakeH.F, lakeH.pool, lakeH.pool_all): fr['composite_perf_index'] = fr[[E.AF[i] + '_pct' for i in COSTLY]].mean(axis=1).round(1)
EV = pd.read_parquet('lake/replacement_events'); EV['qid'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str); Q = EV[['qid', 'C', 't', 'S', 'P']].drop_duplicates('qid'); SA = pd.read_parquet('lake/signing_events'); QA = SA[['qid', 'C', 't', 'P', 'league']].drop_duplicates('qid')
def labelled(P, pos):
    P = P.merge(pos.assign(y=1), on=['qid', 'player_id'], how='left'); P['y'] = P['y'].fillna(0).astype(int); P['q_in_pool'] = P.groupby('qid')['y'].transform('max') == 1; return P
PFh = labelled(lakeH.pair_features(Q), EV[EV['N_in_pool']][['qid', 'N']].drop_duplicates().rename(columns={'N': 'player_id'})); PAh = labelled(lakeH.pair_features_club(QA), SA[['qid', 'N']].drop_duplicates().rename(columns={'N': 'player_id'})); PAh = PAh[PAh['q_in_pool']]
def rec(R):
    R = R[R.groupby('qid')['y'].transform('max') == 1].copy(); R['r'] = R.groupby('qid')['score'].rank(ascending=False, method='average'); return R[R['y'] == 1].groupby('qid')['r'].min()
rH = rec(E.walk_forward(PFh, E.FEATS, TEST, train_pairs=PAh, use_candidates=False)); r0 = rec(pd.read_parquet('lake/scores_lgbm_v2.parquet')).reindex(rH.index); bsq = rng.integers(0, len(rH), (4000, len(rH)))
h0, hH = (r0.to_numpy() <= K).astype(float), (rH.to_numpy() <= K).astype(float)
C['hardened'] = {'gameability_through_cheap_metrics': 0.0, 'test_queries': len(rH), 'recall10_before': float(h0.mean()), 'recall10_after': float(hH.mean()), 'recall10_change_ci95': [float(x) for x in np.percentile((hH[bsq] - h0[bsq]).mean(1), [2.5, 97.5])],
                 'recall25_before': float((r0 <= 25).mean()), 'recall25_after': float((rH <= 25).mean()), 'median_rank_before': float(r0.median()), 'median_rank_after': float(rH.median())}
M['C'] = C; print(C['hardened'])
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5)); byr = RT_.groupby('rank')['gamed'].mean()
ax[0].bar(byr.index, byr.values, width=.6, color=ORANGE); ax[0].axhline(RT_['gamed'].mean(), color=INK2, lw=.8, ls=':'); ax[0].set_xticks(list(byr.index)); ax[0].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[0].set_xlabel('rank before padding'); ax[0].set_title('Share who pad their way into the top 10'); ax[0].grid(axis='x', visible=False)
use = [(gm['d_' + m] > 0).mean() for m in CHEAP]; nm = {'sh': 'shots', 'take_on': 'take-on attempts', 'clr': 'clearances', 'prgc': 'progressive carries'}
ax[1].barh([nm[m] for m in CHEAP][::-1], use[::-1], height=.55, color=BLUE)
for i, m in enumerate(CHEAP[::-1]): ax[1].text(use[::-1][i] + .01, i, f"+{C['median_added_per90'][m]:.2f} per 90" if C['median_added_per90'][m] == C['median_added_per90'][m] else '', va='center', fontsize=7.5)
ax[1].xaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[1].set_xlim(0, max(use) * 1.35 + .05); ax[1].set_title('Which metric the successful padders raised'); ax[1].set_xlabel('share of successful padders (label: median amount added)'); ax[1].grid(axis='y', visible=False)
ax[2].bar([0, 1], [C['gameability'], 0.0], width=.5, color=[ORANGE, GREEN]); ax[2].bar([2.4, 3.4], [C['hardened']['recall10_before'], C['hardened']['recall10_after']], width=.5, color=[ORANGE, GREEN])
for x_, v in zip([0, 1, 2.4, 3.4], [C['gameability'], 0.0, C['hardened']['recall10_before'], C['hardened']['recall10_after']]): ax[2].text(x_, v + .01, f'{v:.0%}', ha='center', fontsize=8)
ax[2].set_xticks([0, 1, 2.4, 3.4], ['engine', 'hardened', 'engine', 'hardened'], fontsize=7.5); ax[2].text(.5, -0.13, 'gameability', ha='center', transform=ax[2].get_xaxis_transform(), fontsize=8, color=INK2); ax[2].text(2.9, -0.13, 'recall@10', ha='center', transform=ax[2].get_xaxis_transform(), fontsize=8, color=INK2)
ax[2].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[2].set_title('Hardening: what it removes and what it costs'); ax[2].grid(axis='x', visible=False); savefig('p4b_redteam')
_ = put_metrics('phase4b', M)
