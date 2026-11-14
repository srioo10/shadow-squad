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
# price-matched null: every pick is swapped for a random pool player of the same season and the same value decile, so a cheap portfolio is compared with other cheap portfolios
pool['dec'] = pool.groupby('yr')['value'].transform(lambda v: pd.qcut(v.rank(method='first'), 10, labels=False)); cells = {k: g[['ret', 'end', 'value']].to_numpy() for k, g in pool.groupby(['yr', 'dec'])}
Vd = V[['player_id', 'date']].copy(); st_ = pool[['player_id', 'yr']].assign(cut=lambda d: pd.to_datetime((d['yr'] + 1).astype(str) + '-05-31').astype('datetime64[ns]')); st_ = pd.merge_asof(st_.sort_values('cut'), Vd, left_on='cut', right_on='date', by='player_id', direction='backward')
pool = pool.merge(st_.assign(stale_days=(st_['cut'] - st_['date']).dt.days)[['player_id', 'yr', 'stale_days']], on=['player_id', 'yr'], how='left'); PMN = {}
for name in PORT:
    pk = picks[name].merge(pool[['player_id', 'yr', 'dec', 'stale_days']], on=['player_id', 'yr']); keys = list(zip(pk['yr'], pk['dec'])); nm_, nmed_, nvw_ = np.empty(5000), np.empty(5000), np.empty(5000)
    for i in range(5000):
        d = np.array([cells[k][rng.integers(len(cells[k]))] for k in keys]); nm_[i] = d[:, 0].mean(); nmed_[i] = np.median(d[:, 0]); nvw_[i] = d[:, 1].sum() / d[:, 2].sum() - 1
    PMN[name] = (nm_, nmed_, nvw_); res[name].update({'price_matched_mean': float(nm_.mean()), 'price_matched_median': float(nmed_.mean()), 'price_matched_value_weighted': float(nvw_.mean()), 'p_vs_price_matched_mean': float((nm_ >= res[name]['mean_return']).mean()),
        'p_vs_price_matched_median': float((nmed_ >= res[name]['median_return']).mean()), 'p_vs_price_matched_value_weighted': float((nvw_ >= res[name]['value_weighted_return']).mean()), 'stale_valuation_picks': int((pk['stale_days'] > 183).sum()), 'median_value_eur_m': float(pk['value'].median() / 1e6)})
fresh = pd.concat([g[g['stale_days'] <= 183].dropna(subset=['hed_resid']).sort_values('hed_resid').head(25) for _, g in pool.groupby('yr')]); res['Under-priced, fresh valuations only'] = {**stats_of(fresh), 'p_vs_random_mean': float((null_m >= fresh['ret'].mean()).mean())}
fk = list(zip(fresh['yr'], fresh['dec'])); fm_ = np.array([np.median([cells[k][rng.integers(len(cells[k]))][0] for k in fk]) for _ in range(5000)]); res['Under-priced, fresh valuations only'].update({'price_matched_median': float(fm_.mean()), 'p_vs_price_matched_median': float((fm_ >= fresh['ret'].median()).mean())})
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
print({k: (round(v['median_return'], 3), round(v.get('price_matched_median', np.nan), 3), fmt_p(v.get('p_vs_price_matched_median', np.nan)), v.get('stale_valuation_picks')) for k, v in res.items() if k != 'Random'}); print({k: (round(v['coef'], 3), fmt_p(v['p'])) for k, v in lg1.items()}, '| return model:', {k: (round(v['coef'], 3), fmt_p(v['p'])) for k, v in lg2.items()}); print('R2', round(m1.rsquared, 3), round(m2.rsquared, 3), '| residual vs return', round(rho_r, 3), fmt_p(p_r), byq['mean'].round(3).to_dict()); print('under-priced:', M['A']['hedonic_return']['most_underpriced_metrics'], 'over-priced:', M['A']['hedonic_return']['most_overpriced_metrics'])
order = ['Under-priced (hedonic)', 'Best young performer', 'Hazard', 'Shadow Squad', 'Market value']; fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
for j, (key, pmk, ttl) in enumerate((('median_return', 1, 'Median two-year return of a pick'), ('value_weighted_return', 2, 'Return on the money spent'))):
    for i, o in enumerate(order):
        lo_n, hi_n = np.percentile(PMN[o][pmk], [2.5, 97.5]); ax[j].bar(i, hi_n - lo_n, bottom=lo_n, width=.62, color='#eceae4', zorder=1); ax[j].hlines(PMN[o][pmk].mean(), i - .31, i + .31, color=GREY, lw=1.2, zorder=2)
        v = res[o][key]; ax[j].scatter(i, v, s=95, color=ORANGE if o.startswith('Under') else INK2, zorder=3); ax[j].text(i + .2, v, f'{v:+.0%}', va='center', fontsize=9, color=INK, fontweight='bold' if o.startswith('Under') else 'normal')
    ax[j].axhline(0, color=INK2, lw=.7); ax[j].set_xticks(range(len(order)), [o.replace(' (hedonic)', '').replace('Best young ', 'Best young\n').replace('Market ', 'Market\n').replace('Shadow ', 'Shadow\n') for o in order], fontsize=8); ax[j].yaxis.set_major_formatter(lambda v, _: f'{v:+.0%}'); ax[j].set_title(ttl); ax[j].grid(axis='x', visible=False)
ax[1].text(0.98, 0.97, 'dot: the portfolio\ngrey band: 95% of random portfolios\nbought at the same prices', transform=ax[1].transAxes, fontsize=7.5, color=INK2, va='top', ha='right'); savefig('p4b_portfolio')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1, 1.5]}); L4 = list(lg1)
for i, (d, lab_, c) in enumerate(((lg1, 'price today (value model)', BLUE), (lg2, 'return over the next 24 months', ORANGE))):
    ax[0].errorbar([d[k]['coef'] for k in L4], np.arange(len(L4)) + (i - .5) * .28, xerr=[[d[k]['coef'] - d[k]['lo'] for k in L4], [d[k]['hi'] - d[k]['coef'] for k in L4]], fmt='o', color=c, label=lab_, capsize=2, lw=1.2)
ax[0].axvline(0, color=INK2, lw=.8); ax[0].set_yticks(range(len(L4)), L4); ax[0].set_xlabel('log difference from the Premier League, like-for-like player'); ax[0].set_title('League: priced lower, and does it come back?'); ax[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1); ax[0].grid(axis='y', visible=False)
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
ax[0].set_yticks(range(len(o)), [c.replace('z_', '').replace('_num', '').replace('composite_perf_index', 'composite').replace('lv', 'log value') for c in o], fontsize=6.5); ax[0].set_xlabel('|standardised mean difference|'); ax[0].set_title('Balance of the twins'); ax[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2); ax[0].grid(axis='y', visible=False)
gs = ['all', '21 or under', '22', '23-24', 'DF', 'MF', 'FW']; y = np.arange(len(gs))[::-1]
ax[1].errorbar([CF[g]['cate'] for g in gs], y, xerr=[[CF[g]['cate'] - CF[g]['ci95'][0] for g in gs], [CF[g]['ci95'][1] - CF[g]['cate'] for g in gs]], fmt='o', color=ORANGE, capsize=2, lw=1.2); ax[1].axvline(0, color=INK2, lw=.8)
ax[1].set_yticks(y, [f"{g} (n={CF[g]['treated']})" for g in gs], fontsize=7.5); ax[1].xaxis.set_major_formatter(lambda v, _: f'{v:+.0%}'); ax[1].set_xlabel('effect on share of league minutes, t to t+2'); ax[1].set_title('Causal forest: who pays the price?'); ax[1].grid(axis='y', visible=False)
lab2 = [('minutes_share', 'share of league\nminutes'), ('neutral_output_index', 'style-neutral\noutput index / 100')]; sc2 = [1, 100]
for i, ((k, lb), sc_) in enumerate(zip(lab2, sc2)):
    r = RB[k]; ax[2].bar([i - .2, i + .2], [r['treated_mean_change'] / sc_, r['twin_mean_change'] / sc_], width=.36, color=[ORANGE, GREY]); ax[2].text(i, 0.008, f"difference {r['att'] / sc_:+.3f} (p {fmt_p(r['p_sign_flip'])})", ha='center', va='bottom', fontsize=7.5)
ax[2].bar([0], [0], color=ORANGE, label='moved to an elite club'); ax[2].bar([0], [0], color=GREY, label='matched twin who stayed'); ax[2].axhline(0, color=INK2, lw=.8); ax[2].set_xticks([0, 1], [l for _, l in lab2], fontsize=7.5); ax[2].set_ylim(min(RB['minutes_share']['treated_mean_change'], RB['minutes_share']['twin_mean_change']) * 1.35, 0.045); ax[2].set_title('Change from season t to t+2'); ax[2].legend(fontsize=7, loc='lower right'); ax[2].grid(axis='x', visible=False); savefig('p4b_sliding_doors')

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
PPd = pd.read_parquet('lake/person_period.parquet')[['player_id', 'yr', 'event']]; Hc = [c for c in HFe if c not in ['z_' + m for m in CHEAP]]; HXc = HX[HFe].fillna(0).copy(); HXc['composite_perf_index'] = cc.to_numpy(); HXc['player_id'] = Fm['player_id'].to_numpy(); HXc['yr'] = yr_a; hz = []; HZc = {}
for t in range(2018, 2025):
    trn = PPd[PPd['yr'] <= t - 1].merge(HXc, on=['player_id', 'yr']); mu, sd = trn[Hc].mean(), trn[Hc].std().replace(0, 1); lr = LogisticRegression(C=1 / (0.05 * len(trn)), max_iter=3000).fit((trn[Hc] - mu) / sd, trn['event']); HZc[t] = (mu.to_numpy(), sd.to_numpy(), lr.coef_[0], float(lr.intercept_[0])); cur = HXc[HXc['yr'] == t]
    hz.append(pd.DataFrame({'player_id': cur['player_id'], 'yr': t, 'hazard': lr.predict_proba((cur[Hc] - mu) / sd)[:, 1]}))
lakeH.HZ = pd.concat(hz)
for fr in (lakeH.F, lakeH.pool, lakeH.pool_all): fr['composite_perf_index'] = fr[[E.AF[i] + '_pct' for i in COSTLY]].mean(axis=1).round(1)
EV = pd.read_parquet('lake/replacement_events'); EV['qid'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str); Q = EV[['qid', 'C', 't', 'S', 'P']].drop_duplicates('qid'); SA = pd.read_parquet('lake/signing_events'); QA = SA[['qid', 'C', 't', 'P', 'league']].drop_duplicates('qid')
def labelled(P, pos):
    P = P.merge(pos.assign(y=1), on=['qid', 'player_id'], how='left'); P['y'] = P['y'].fillna(0).astype(int); P['q_in_pool'] = P.groupby('qid')['y'].transform('max') == 1; return P
PFh = labelled(lakeH.pair_features(Q), EV[EV['N_in_pool']][['qid', 'N']].drop_duplicates().rename(columns={'N': 'player_id'})); PAh = labelled(lakeH.pair_features_club(QA), SA[['qid', 'N']].drop_duplicates().rename(columns={'N': 'player_id'})); PAh = PAh[PAh['q_in_pool']]
def rec(R):
    R = R[R.groupby('qid')['y'].transform('max') == 1].copy(); R['r'] = R.groupby('qid')['score'].rank(ascending=False, method='average'); return R[R['y'] == 1].groupby('qid')['r'].min()
RH_, modelsH = E.walk_forward(PFh, E.FEATS, TEST, train_pairs=PAh, use_candidates=False, keep_models=True); rH = rec(RH_); r0 = rec(pd.read_parquet('lake/scores_lgbm_v2.parquet')).reindex(rH.index); bsq = rng.integers(0, len(rH), (4000, len(rH)))
h0, hH = (r0.to_numpy() <= K).astype(float), (rH.to_numpy() <= K).astype(float)
C['hardened'] = { 'test_queries': len(rH), 'recall10_before': float(h0.mean()), 'recall10_after': float(hH.mean()), 'recall10_change_ci95': [float(x) for x in np.percentile((hH[bsq] - h0[bsq]).mean(1), [2.5, 97.5])],
                 'recall25_before': float((r0 <= 25).mean()), 'recall25_after': float((rH <= 25).mean()), 'median_rank_before': float(r0.median()), 'median_rank_after': float(rH.median())}
print(C['hardened'])

# %% [markdown]
# ### How big is the padding, who can do it, and does an adaptive attacker beat the hardened model?
# (a) The padding is compared with how much each cheap metric moves from one season to the next for the same player. (b) The greedy search is repeated for players ranked 21 to 50. (c) A genetic-algorithm attacker (the canonical GA of the lectures: binary chromosome, roulette selection, single-point crossover, mutation) may raise any metric: a cheap step costs 1, a costly step costs 4. It attacks the engine and the hardened model at several budgets.

# %%
import ga as GA
prevF = Fm[['player_id', 'yr'] + E.RAW].assign(yr=lambda d: d['yr'] + 1); yo = Fm[['player_id', 'yr'] + E.RAW].merge(prevF, on=['player_id', 'yr'], suffixes=('', '_p')); nat = {m: float((yo[E.RAW[j]] - yo[E.RAW[j] + '_p']).std()) for m, j in zip(CHEAP, CI_)}
C['natural_yoy_sd'] = nat; C['median_added_in_yoy_sd'] = {m: float(C['median_added_per90'][m] / nat[m]) if C['median_added_per90'][m] == C['median_added_per90'][m] else None for m in CHEAP}
def greedy(row, mdl, thr):
    xi = lake.idx[(row.player_id, row.t)]; cur = lake.RAWV[xi].copy(); lo, hi = BND[(row.t, row.P)]; step = (hi - lo) / 20.0; sc = row.score; n = 0
    while sc <= thr and n < 60:
        opts = [j for j in CI_ if cur[j] + step[j] <= max(hi[j], lake.RAWV[xi][j]) + 1e-9]
        if not opts: break
        var = np.repeat(cur[None, :], len(opts), 0)
        for i, j in enumerate(opts): var[i, j] += step[j]
        s_ = rescore(row, var, mdl); b_ = int(np.argmax(s_))
        if s_[b_] <= sc + 1e-12: break
        cur = var[b_]; sc = s_[b_]; n += 1
    return sc > thr, n
far = []; q2 = qs.sample(min(60, len(qs)), random_state=SEED + 1)
for q in q2:
    G = PF[PF['qid'] == q].copy(); mdl = models[int(G['t'].iat[0])]; G['score'] = mdl.predict(G[E.FEATS]); G = G.sort_values('score', ascending=False).reset_index(drop=True)
    for rk in range(20, min(50, len(G))):
        ok_, n_ = greedy(G.iloc[rk], mdl, G['score'].iat[K - 1]); far.append({'rank': rk + 1, 'gamed': ok_, 'steps': n_})
FAR = pd.DataFrame(far); FAR['band'] = pd.cut(FAR['rank'], [20, 30, 40, 50], labels=['21-30', '31-40', '41-50']); C['by_rank_band'] = {'11-20': float(RT_['gamed'].mean()), **FAR.groupby('band', observed=True)['gamed'].mean().round(4).to_dict()}; C['far_players'] = len(FAR); C['median_steps_21_50'] = float(FAR[FAR['gamed']]['steps'].median()) if FAR['gamed'].any() else None
# rescoring under the hardened model: only the twelve costly metrics reach it
Hpos = [Hc.index(c) for c in Hc]; zc_names = ['z_' + E.MET[i] for i in COSTLY]; HXh = HXc[Hc].to_numpy(float)
def rescore_h(row, raws, model):
    xi, si, t, fp, pg = lake.idx[(row.player_id, row.t)], lake.idx[(row.S, row.t)], row.t, row.P, pg_a[lake.idx[(row.player_id, row.t)]]; k = len(raws); r0, n0 = lake.RAWV[xi], lake.NEU[xi]
    own = lake.translate(np.ones((1, len(E.MET))), [pg], Fm['x_dom'].iat[xi], Fm['x_press'].iat[xi], Fm['comp'].iat[xi], t)[0]; neu = np.where(r0 > 0, n0 * raws / np.where(r0 > 0, r0, 1), raws / own); z = lake.Z[xi] + (np.log(neu + E.EPS) - np.log(n0 + E.EPS)) / SDN[(t, fp)]
    tr = lake.translate(neu, [pg] * k, Fm['x_dom'].iat[si], Fm['x_press'].iat[si], Fm['comp'].iat[si], t); zS = lake.zraw(lake.RAWV[[si]], [fp], t); srt = SRT[(t, pg)]
    pct = lambda v: np.array([[np.searchsorted(srt[:, j], v[i, j], side='right') / len(srt) * 100 for j in COSTLY] for i in range(len(v))]); comp = row.composite_perf_index + (pct(neu) - pct(n0[None, :])).mean(1)
    X = pd.DataFrame({f: np.repeat(getattr(row, f), k) for f in E.FEATS}); X['sim_translated'] = E.rowcos(lake.zraw(tr, [fp] * k, t)[:, COSTLY], np.repeat(zS, k, 0)[:, COSTLY]); X['traj_sim'] = [lakeH._traj(xi, si, t, z0=z[i])[0] for i in range(k)]; X['composite_perf_index'] = comp
    if t in HZc:
        mu, sd, cf_, b = HZc[t]; hx = np.repeat(HXh[[xi]], k, 0)
        for zn, j in zip(zc_names, COSTLY): hx[:, Hc.index(zn)] = np.nan_to_num(z[:, j])
        hx[:, Hc.index('composite_perf_index')] = comp; X['hazard'] = 1 / (1 + np.exp(-(((hx - mu) / sd) @ cf_ + b)))
    return model.predict(X[E.FEATS])
BUD = (1, 2, 4, 8, 16); W_COSTLY = 4; NB = 2; cost_w = np.array([1 if j in CI_ else W_COSTLY for j in range(len(E.MET))])
def ga_attack(row, mdl, thr, budget, fn, seed):
    xi = lake.idx[(row.player_id, row.t)]; base = lake.RAWV[xi]; lo, hi = BND[(row.t, row.P)]; step = (hi - lo) / 20.0
    def fit(Pm):
        st = np.stack([GA.decode(ch, len(E.MET), NB) for ch in np.asarray(Pm)]); raws = np.minimum(base + st * step, np.maximum(hi, base)); cost = (st * cost_w).sum(1); sc = fn(row, raws, mdl); return np.where(cost <= budget, sc, sc - 5.0 * (cost - budget))
    best, bf, _ = GA.run(fit, len(E.MET) * NB, pop=24, gens=10, seed=seed, init=np.vstack([np.zeros((1, len(E.MET) * NB), int), np.random.default_rng(seed).integers(0, 2, (23, len(E.MET) * NB)) * (np.random.default_rng(seed + 1).random((23, len(E.MET) * NB)) < 0.15)]))
    return bf > thr
att = []; t0 = time.time(); nm_rows = []
for q in qs.sample(min(40, len(qs)), random_state=SEED + 2):
    G = PF[PF['qid'] == q].copy(); t_ = int(G['t'].iat[0]); G['score'] = models[t_].predict(G[E.FEATS]); G = G.sort_values('score', ascending=False).reset_index(drop=True); Gh = PFh[PFh['qid'] == q].copy(); Gh['score'] = modelsH[t_].predict(Gh[E.FEATS]); Gh = Gh.sort_values('score', ascending=False).reset_index(drop=True)
    for rk in (11, 15):
        if len(G) > rk and len(Gh) > rk:
            for b_ in BUD:
                att.append({'model': 'engine', 'budget': b_, 'gamed': bool(ga_attack(G.iloc[rk], models[t_], G['score'].iat[K - 1], b_, rescore, SEED + rk))}); att.append({'model': 'hardened', 'budget': b_, 'gamed': bool(ga_attack(Gh.iloc[rk], modelsH[t_], Gh['score'].iat[K - 1], b_, rescore_h, SEED + rk))})
AT = pd.DataFrame(att); curve = AT.groupby(['model', 'budget'])['gamed'].agg(['mean', 'size']).reset_index(); C['adaptive_attack'] = {'players_per_model': int(len(AT) / len(BUD) / 2), 'cost_of_costly_step': W_COSTLY, 'step': '5% of the position range', 'curve': curve.round(4).to_dict('records'), 'seconds': round(time.time() - t0, 1)}
M['C'] = C; print(C['by_rank_band'], C['median_added_in_yoy_sd']); display(curve.pivot(index='budget', columns='model', values='mean').round(3))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9)); piv = curve.pivot(index='budget', columns='model', values='mean')
for mdl_, col in (('engine', ORANGE), ('hardened', GREEN)):
    ax[0].plot(piv.index, piv[mdl_], 'o-', color=col, lw=2.4); ax[0].text(piv.index[-1] * 1.12, piv[mdl_].iloc[-1], 'engine' if mdl_ == 'engine' else 'hardened\n(costly metrics only)', color=col, fontsize=8.5, va='center', fontweight='bold')
ax[0].set_xscale('log', base=2); ax[0].set_xticks(list(BUD), [str(b_) for b_ in BUD]); ax[0].set_xlim(0.85, 34); ax[0].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[0].set_xlabel(f'attack budget (a cheap step costs 1, a costly step {W_COSTLY})'); ax[0].set_ylabel('near-miss players who reach the top 10'); ax[0].set_title('An adaptive attacker against both models')
bands = list(C['by_rank_band']); vals = [C['by_rank_band'][b_] for b_ in bands]; ax[1].bar(range(len(bands)), vals, width=.55, color=[ORANGE] + [GREY] * (len(bands) - 1))
for i, v in enumerate(vals): ax[1].text(i, v + .008, f'{v:.0%}', ha='center', fontsize=9)
ax[1].set_xticks(range(len(bands)), [f'ranked {b_}' for b_ in bands]); ax[1].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[1].set_title('Padding cheap stats: who can reach the top 10 (engine)'); ax[1].grid(axis='x', visible=False); savefig('p4b_redteam')

# %% [markdown]
# ## D. Budget squad rebuild: a genetic algorithm against the exact optimum
# A club that loses several starters in one summer has one budget for all of them. Each open position has its 16 best-ranked candidates; pick one per position to maximise the summed fit (rank percentile from the engine), without spending more than the departed starters were worth and without picking a player twice.
# The genetic algorithm encodes each position's choice in 4 bits. The fitness of a whole population is computed as one Spark job. Integer linear programming (PuLP with CBC, out of course) gives the true optimum to compare with.

# %%
sc = sp.sparkContext; LVs = pd.read_parquet('lake/scores_lgbm_v2.parquet').merge(PF[['qid', 'player_id', 'C', 'S', 'value', 'S_value', 'player', 'squad']], on=['qid', 'player_id']); LVs['q'] = LVs.groupby('qid')['score'].rank(pct=True); inst = []
for (C_, t_), g in LVs.groupby(['C', 't']):
    qs_ = g['qid'].unique()
    if 2 <= len(qs_) <= 6:
        opts = [g[g['qid'] == q].dropna(subset=['value']).nlargest(16, 'score') for q in qs_]
        if all(len(o) == 16 for o in opts): inst.append((C_, t_, np.stack([o['q'].to_numpy() for o in opts]), np.stack([o['value'].to_numpy() / 1e6 for o in opts]), np.stack([o['player_id'].to_numpy() for o in opts]), float(np.nansum([o['S_value'].iat[0] for o in opts]) / 1e6)))
rb = []
def spark_fit(f1):
    return lambda Pm: np.array(sc.parallelize(np.asarray(Pm).tolist(), 8).map(lambda ch: float(f1(np.array([ch]))[0])).collect())
for n_, (C_, t_, q_, pr_, pid_, bud) in enumerate(inst):
    bud = max(bud, float(np.sort(pr_, 1)[:, :2].mean(1).sum())); t0 = time.time(); pick, opt, ok_ = GA.squad_ilp(q_, pr_, pid_, bud); t_ilp = time.time() - t0
    if not ok_: continue
    f1 = GA.squad_fitness(q_, pr_, pid_, bud, 4); t0 = time.time(); best, bf, hist = GA.run(f1, 4 * len(q_), pop=60, gens=40, seed=SEED); t_ga = time.time() - t0; idx = np.minimum(GA.decode(best, len(q_), 4), 15); ar = np.arange(len(q_))
    feas = pr_[ar, idx].sum() <= bud + 1e-9 and len(set(pid_[ar, idx].tolist())) == len(q_); row = {'club': C_, 't': t_, 'slots': len(q_), 'budget_m': bud, 'ilp': opt, 'ga': float(q_[ar, idx].sum()) if feas else np.nan, 'feasible': bool(feas), 'ga_s': t_ga, 'ilp_s': t_ilp, 'gens_to_best': int(np.argmax([h[0] for h in hist]))}
    if n_ < 10: t0 = time.time(); _, bfs, _ = GA.run(spark_fit(f1), 4 * len(q_), pop=60, gens=40, seed=SEED); row['ga_spark_s'] = time.time() - t0; row['spark_equals_local'] = bool(abs(bfs - bf) < 1e-9)
    rb.append(row)
RB_ = pd.DataFrame(rb); RB_['ratio'] = RB_['ga'] / RB_['ilp']; display(RB_.head(8).round(3))
M['D'] = {'instances': len(RB_), 'mean_slots': float(RB_['slots'].mean()), 'ga_feasible_share': float(RB_['feasible'].mean()), 'mean_ratio_to_optimum': float(RB_['ratio'].mean()), 'min_ratio': float(RB_['ratio'].min()), 'share_reaching_optimum': float((RB_['ratio'] >= 1 - 1e-9).mean()),
          'mean_ga_seconds': float(RB_['ga_s'].mean()), 'mean_ilp_seconds': float(RB_['ilp_s'].mean()), 'mean_ga_spark_seconds': float(RB_['ga_spark_s'].mean()), 'spark_instances': int(RB_['ga_spark_s'].notna().sum()), 'spark_equals_local': bool(RB_['spark_equals_local'].dropna().all()),
          'median_generation_of_best': float(RB_['gens_to_best'].median()), 'search_space_per_slot': 16, 'population': 60, 'generations': 40}
print(M['D'])
fig, ax = plt.subplots(figsize=(6.2, 3.3)); r_ = RB_['ratio'].dropna().sort_values().to_numpy(); ax.bar(range(len(r_)), r_, color=np.where(r_ >= 1 - 1e-9, GREEN, GREY), width=.8); ax.axhline(1, color=INK2, lw=.8)
ax.set_ylim(min(0.8, r_.min() - .02), 1.02); ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax.set_xticks([]); ax.set_xlabel(f'{len(r_)} club-summers with 2 to 6 starters to replace, sorted'); ax.set_ylabel('GA result as a share of the ILP optimum'); ax.set_title(f"Genetic algorithm against the exact optimum: {(r_ >= 1 - 1e-9).mean():.0%} optimal"); ax.grid(axis='x', visible=False); savefig('p4b_rebuild')
_ = put_metrics('phase4b', M)
