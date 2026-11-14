# %% [markdown]
# # End-sem 03A. Post-mid-semester topics: closed and maximal frequent itemsets, DBSCAN
# Closed, maximal and longest frequent itemsets and DBSCAN, each implemented in PySpark or checked against a second algorithm for the same task. The genetic algorithm is in notebook 04B (it needs the engine's scores).
# Definitions follow the lecture slides: an itemset is **closed** if no immediate superset has the same support, and **maximal frequent** if no immediate superset is frequent.

# %%
import sys, os, time; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
import itemsets as IS, dbscan_dist as DD
sp = spark(); sc = sp.sparkContext; M = {}; MET = [f[:-4] for f in MID['active_features']]; HI = ['hi_' + m for m in MET]; POS = ['CB', 'FB', 'DM', 'CM', 'AM', 'W', 'ST']
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
FT = pd.read_parquet('lake/features'); SA = pd.read_parquet('lake/signing_events'); PA_ = pd.read_parquet('lake/pool_all')[['player_id', 'yr']].merge(FT, on=['player_id', 'yr'])
SG = SA[['N', 't', 'P']].rename(columns={'N': 'player_id', 't': 'yr'}).drop_duplicates().merge(FT[['player_id', 'yr', 'fine_pos'] + HI], on=['player_id', 'yr'])
mask = lambda d: (d[HI].to_numpy(int) * (1 << np.arange(len(HI)))).sum(1)
names = MET + ['pos=' + p for p in POS]; show = lambda m: '{' + ', '.join(names[i] for i in IS.bits(m)) + '}'
TX = (mask(SG) + np.array([1 << (len(HI) + POS.index(p)) for p in SG['fine_pos']])).tolist(); NI = len(names); rdd = sc.parallelize(TX, 8).cache()
print(f'{len(TX):,} transactions (one per signed player: his top-quartile traits and his position), {NI} items, mean length {np.mean([IS.pc(t) for t in TX]):.1f}')

# %% [markdown]
# ## 1. Closed frequent itemsets (CFI): A-Close in Spark, checked against CHARM
# **A-Close** (taught) finds the minimal generators level by level, counting every candidate with one `mapPartitions` + `reduceByKey` job per level, then closes each generator by intersecting the transactions that contain it. **CHARM** (new) keeps, for every itemset, the set of transactions it occurs in, and merges items whose transaction sets coincide or nest, so it never enumerates the non-closed sets.

# %%
rows, CL = [], {}; tid = IS.tidsets(rdd, NI)
for s in (0.10, 0.05, 0.02):
    ms = int(np.ceil(s * len(TX))); t0 = time.time(); ac, st_a = IS.aclose(rdd, NI, ms, len(TX)); ta = time.time() - t0; t0 = time.time(); ch, st_c = IS.charm(tid, ms); tc = time.time() - t0
    t0 = time.time(); pm, st_p = IS.pincer(rdd, NI, ms); tp = time.time() - t0; t0 = time.time(); gm, st_g = IS.genmax(tid, ms); tg = time.time() - t0; CL[s] = (ac, pm)
    mx_from_closed = {x for x in ac if not any(x != y and x & y == x for y in ac)}
    rows.append({'min_support': s, 'closed': len(ac), 'maximal': len(pm), 'aclose_equals_charm': ac == ch, 'pincer_equals_genmax': pm == gm, 'maximal_equals_max_of_closed': pm == mx_from_closed,
                 'aclose_passes': st_a['passes'], 'aclose_candidates': st_a['candidates'], 'aclose_s': round(ta, 2), 'charm_intersections': st_c['intersections'], 'charm_s': round(tc, 3),
                 'pincer_passes': st_p['passes'], 'pincer_candidates': st_p['candidates'], 'pincer_saved_by_mfs': st_p['saved_by_mfs'], 'pincer_s': round(tp, 2), 'genmax_intersections': st_g['intersections'], 'genmax_pruned': st_g['pruned_branches'], 'genmax_s': round(tg, 3)})
IT = pd.DataFrame(rows); display(IT.T)
assert IT[['aclose_equals_charm', 'pincer_equals_genmax', 'maximal_equals_max_of_closed']].all().all(), 'itemset algorithms disagree'
ac, pm = CL[0.05]; top_c = sorted([(x, s) for x, s in ac.items() if IS.pc(x) >= 3], key=lambda v: -v[1])[:6]; top_m = sorted(pm, key=lambda x: -IS.pc(x))[:6]
M['itemsets'] = {'transactions': len(TX), 'items': NI, 'by_support': IT.to_dict('records'), 'example_closed': [(show(x), int(s)) for x, s in top_c], 'example_maximal': [show(x) for x in top_m]}
print('largest closed sets with 3+ items at 5%:', M['itemsets']['example_closed'][:3]); print('longest maximal sets at 5%:', M['itemsets']['example_maximal'][:3])

# %% [markdown]
# ## 2. Maximal frequent itemsets (MFI): Pincer-Search in Spark, checked against GenMax
# **Pincer-Search** (taught) runs Apriori bottom-up and, in the same pass, counts the elements of the maximal frequent candidate set (MFCS) top-down; an MFCS element found frequent is maximal at once and its subsets need no counting. **GenMax** (new) is a depth-first search on transaction sets that cuts a branch as soon as the itemset plus everything it could still add already lies inside a known maximal set. The table above shows the two return identical sets at three support levels, and that the maximal sets are exactly the maximal elements of the closed sets.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4)); x = np.arange(len(IT)); lab = [f'{s:.0%}' for s in IT['min_support']]
for i, (c, nm, col) in enumerate((('closed', 'closed', BLUE), ('maximal', 'maximal', ORANGE))):
    ax[0].bar(x + (i - .5) * .36, IT[c], width=.34, color=col)
    for xx, v in zip(x + (i - .5) * .36, IT[c]): ax[0].text(xx, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
ax[0].text(x[-1] - .18, IT['closed'].iloc[-1] * .55, 'closed', color=BLUE, ha='center', fontsize=9, fontweight='bold', rotation=90, va='center'); ax[0].text(x[0] + .18, IT['maximal'].iloc[0] * 1.9 + 30, 'maximal', color=ORANGE, ha='center', fontsize=9, fontweight='bold')
ax[0].set_xticks(x, lab); ax[0].set_xlabel('minimum support'); ax[0].set_title('Closed and maximal itemsets of signed players'); ax[0].grid(axis='x', visible=False)
for i, (c, nm, col) in enumerate((('aclose_candidates', 'A-Close (Spark)', BLUE), ('pincer_candidates', 'Pincer-Search (Spark)', ORANGE))):
    ax[1].plot(x, IT[c], 'o-', color=col); ax[1].text(x[-1] + .06, IT[c].iloc[-1], nm, color=col, va='center', fontsize=8.5)
ax[1].set_xticks(x, lab); ax[1].set_xlim(-.2, len(IT) - .2 + .9); ax[1].set_yscale('log'); ax[1].set_xlabel('minimum support'); ax[1].set_title('Candidates counted by the Spark jobs'); savefig('p3a_itemsets')

# %% [markdown]
# ## 3. Longest frequent itemsets (LFI)
# An LFI is a frequent itemset of the greatest length. Every LFI is maximal, but most maximal sets are shorter. Taken from the maximal sets it is one line (keep the longest). Mined directly, a depth-first search can do far less work: once a frequent set of length L is known, any branch that cannot reach L is dropped.

# %%
lrows = []
for s in (0.10, 0.05, 0.02):
    ms = int(np.ceil(s * len(TX))); pm = CL[s][1]; L = max(IS.pc(x) for x in pm); from_max = {x for x in pm if IS.pc(x) == L}; t0 = time.time(); direct, Ld, st_l = IS.lfi(tid, ms); tl = time.time() - t0
    lrows.append({'min_support': s, 'longest_length': L, 'lfi_count': len(from_max), 'maximal_count': len(pm), 'direct_equals_longest_maximal': direct == from_max, 'intersections': st_l['intersections'], 'branches_cut_by_length': st_l['pruned_by_length'], 'seconds': round(tl, 3), 'example': show(sorted(direct)[0])})
LF = pd.DataFrame(lrows); display(LF); assert LF['direct_equals_longest_maximal'].all(); M['lfi'] = LF.to_dict('records')

# %% [markdown]
# ## 4. Using the itemsets in the engine: the pattern-match score
# For every season cut-off t and fine position, the closed trait sets of players signed in windows before t are mined (CHARM; identical to A-Close, as shown above). Each pattern gets a lift: its support among signed players divided by its support among all 900-minute players of that position in the same seasons. A candidate's **pattern-match score** is the sum of log-lift over the closed patterns his own traits satisfy. The maximal patterns he satisfies are listed in the dashboard ("matches 2 of the maximal profiles past signings at this position had").

# %%
t0 = time.time(); pats, pm_rows = [], []
for t in range(2018, 2026):
    for P in POS:
        sg = SG[(SG['yr'] <= t - 1) & (SG['fine_pos'] == P)]; pool = PA_[(PA_['yr'] <= t - 1) & (PA_['fine_pos'] == P)]; cur = FT[(FT['yr'] == t) & (FT['fine_pos'] == P)]
        if len(sg) < 30 or not len(cur): continue
        tx = mask(sg).tolist(); ms = max(5, int(np.ceil(0.06 * len(tx)))); td = {}
        for i, tr in enumerate(tx):
            for b in IS.bits(tr): td[1 << b] = td.get(1 << b, 0) | (1 << i)
        closed, _ = IS.charm(td, ms); pmk = mask(pool); mx = {x for x in closed if not any(x != y and x & y == x for y in closed)}; keep = []
        for x, s in closed.items():
            sp_ = int(((pmk & x) == x).sum()); lift = (s / len(tx)) / max(sp_ / len(pool), 1e-9)
            if lift > 1.05: keep.append((x, s, lift, x in mx)); pats.append({'cutoff': t, 'fine_pos': P, 'pattern': show(x), 'n_items': IS.pc(x), 'support': s, 'signed': len(tx), 'lift': lift, 'maximal': x in mx})
        cm = mask(cur)
        for pid, m_ in zip(cur['player_id'], cm):
            hit = [(x, lf, mxf) for x, s, lf, mxf in keep if (int(m_) & x) == x]; mxh = sorted([(lf, show(x)) for x, lf, mxf in hit if mxf], reverse=True)
            pm_rows.append({'player_id': pid, 'yr': t, 'pattern_match': float(sum(np.log(lf) for _, lf, _ in hit)), 'n_closed_matched': len(hit), 'n_maximal_matched': len(mxh), 'n_maximal_total': sum(1 for k in keep if k[3]), 'maximal_matched': ' | '.join(s_ for _, s_ in mxh[:3])})
PT = pd.DataFrame(pats); PM = pd.DataFrame(pm_rows); PT.to_parquet('lake/signing_patterns.parquet', index=False); PM.to_parquet('lake/pattern_match.parquet', index=False)
LB = pd.read_parquet('lake/labels_v2'); LB = LB[LB['feeder'] & LB['player_id'].notna() & LB['observed24']].assign(up=lambda d: (d['up_tier'] >= 1).astype(int)).merge(PM, on=['player_id', 'yr'])
sgn = SA[['N', 't']].drop_duplicates().rename(columns={'N': 'player_id', 't': 'yr'}).assign(signed=1); AL = PA_[['player_id', 'yr']].merge(PM, on=['player_id', 'yr']).merge(sgn, on=['player_id', 'yr'], how='left').fillna({'signed': 0})
from sklearn.metrics import roc_auc_score
M['pattern_match'] = {'patterns': len(PT), 'maximal_patterns': int(PT['maximal'].sum()), 'cutoffs': int(PT['cutoff'].nunique()), 'rows_scored': len(PM), 'seconds': round(time.time() - t0, 1), 'auc_signed_next_summer': float(roc_auc_score(AL['signed'], AL['pattern_match'])),
                      'auc_feeder_moves_up': float(roc_auc_score(LB['up'], LB['pattern_match'])), 'signed_rate_top_quintile': float(AL[AL['pattern_match'] >= AL['pattern_match'].quantile(.8)]['signed'].mean()), 'signed_rate_rest': float(AL[AL['pattern_match'] < AL['pattern_match'].quantile(.8)]['signed'].mean()),
                      'top_patterns_2024': PT[(PT['cutoff'] == 2024) & PT['maximal'] & (PT['support'] >= 10)].nlargest(6, 'lift')[['fine_pos', 'pattern', 'support', 'lift']].round(2).to_dict('records')}
print({k: v for k, v in M['pattern_match'].items() if k != 'top_patterns_2024'}); display(pd.DataFrame(M['pattern_match']['top_patterns_2024']))

# %% [markdown]
# ## 5. DBSCAN: grid-partitioned and distributed in Spark, checked against sklearn; HDBSCAN as the new algorithm
# **Distributed DBSCAN** (taught algorithm, made to scale): the points of a position are cut into slabs along the first principal axis. Each slab also gets a halo of width eps from its neighbours, so every point's eps-neighbourhood is complete in its home slab. Spark finds core points per slab, builds clusters per slab, and merges clusters that share a core point. **HDBSCAN** (new) needs no eps: it builds the hierarchy of density levels and keeps the most stable clusters.
# Data: the 16 standardised style-neutral metrics of each fine position, reduced to 5 principal components.

# %%
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
ZC = ['z_' + m for m in MET]; MINPTS = 8; drows, sens = [], []; t_all = time.time()
for P in POS:
    d = FT[FT['fine_pos'] == P].dropna(subset=ZC); X = PCA(5, random_state=SEED).fit_transform(d[ZC].to_numpy()); kd = np.sort(NearestNeighbors(n_neighbors=MINPTS).fit(X).kneighbors(X)[0][:, -1]); eps0 = float(np.quantile(kd, 0.80))
    t0 = time.time(); lab, core, st = DD.dist_dbscan(sp, X, eps0, MINPTS); td_ = time.time() - t0; t0 = time.time(); sk = DBSCAN(eps=eps0, min_samples=MINPTS).fit(X); ts = time.time() - t0; skc = np.zeros(len(X), bool); skc[sk.core_sample_indices_] = True
    hd = HDBSCAN(min_cluster_size=25, min_samples=MINPTS).fit(X)
    drows.append({'fine_pos': P, 'n': len(X), 'eps': eps0, 'clusters_spark': int(lab.max() + 1), 'clusters_sklearn': int(sk.labels_.max() + 1), 'noise_spark': float((lab == -1).mean()), 'noise_sklearn': float((sk.labels_ == -1).mean()), 'core_identical': bool((core == skc).all()),
                  'ari_spark_vs_sklearn': float(adjusted_rand_score(sk.labels_, lab)), 'ari_core_points': float(adjusted_rand_score(sk.labels_[skc], lab[skc])) if skc.sum() > 1 else 1.0, 'halo_share': st['halo_share'], 'merges': st['merges'], 'spark_s': round(td_, 2), 'sklearn_s': round(ts, 3),
                  'hdbscan_clusters': int(hd.labels_.max() + 1), 'hdbscan_noise': float((hd.labels_ == -1).mean()), 'ari_hdbscan_vs_dbscan': float(adjusted_rand_score(sk.labels_, hd.labels_))})
    for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        e = float(np.quantile(kd, q)); l2 = DBSCAN(eps=e, min_samples=MINPTS).fit_predict(X); sens.append({'fine_pos': P, 'kdist_quantile': q, 'eps': e, 'clusters': int(l2.max() + 1), 'noise': float((l2 == -1).mean())})
DT = pd.DataFrame(drows); SE = pd.DataFrame(sens); display(DT.round(3))
# walk-forward use in the engine's vocabulary: is a player "unique" (DBSCAN noise) among the players of his position seen up to his own season?
uq = []
for y in range(2017, 2025):
    for P in POS:
        d = FT[(FT['fine_pos'] == P) & (FT['yr'] <= y)].dropna(subset=ZC)
        if len(d) < 60: continue
        X = PCA(5, random_state=SEED).fit_transform(d[ZC].to_numpy()); kd = np.sort(NearestNeighbors(n_neighbors=MINPTS).fit(X).kneighbors(X)[0][:, -1]); l2 = DBSCAN(eps=float(np.quantile(kd, 0.80)), min_samples=MINPTS).fit_predict(X); cur = (d['yr'] == y).to_numpy()
        uq.append(pd.DataFrame({'player_id': d['player_id'].to_numpy()[cur], 'yr': y, 'unique': l2[cur] == -1, 'density_cluster': l2[cur]}))
UQ = pd.concat(uq); UQ.to_parquet('lake/density.parquet', index=False); LU = LB.merge(UQ, on=['player_id', 'yr']); u1, u0 = LU[LU['unique']], LU[~LU['unique']]
fp = stats.fisher_exact([[u1['up'].sum(), (1 - u1['up']).sum()], [u0['up'].sum(), (1 - u0['up']).sum()]])[1]; AU = PA_[['player_id', 'yr']].merge(UQ, on=['player_id', 'yr']).merge(sgn, on=['player_id', 'yr'], how='left').fillna({'signed': 0})
M['dbscan'] = {'min_pts': MINPTS, 'by_position': DT.round(4).to_dict('records'), 'sensitivity': SE.round(4).to_dict('records'), 'all_core_identical': bool(DT['core_identical'].all()), 'min_ari': float(DT['ari_spark_vs_sklearn'].min()), 'min_ari_core': float(DT['ari_core_points'].min()),
               'mean_noise': float(DT['noise_spark'].mean()), 'mean_hdbscan_noise': float(DT['hdbscan_noise'].mean()), 'mean_ari_hdbscan': float(DT['ari_hdbscan_vs_dbscan'].mean()), 'seconds': round(time.time() - t_all, 1),
               'unique_flag': {'feeders': len(LU), 'unique_share': float(LU['unique'].mean()), 'move_up_unique': float(u1['up'].mean()), 'move_up_rest': float(u0['up'].mean()), 'fisher_p': float(fp), 'signed_unique': float(AU[AU['unique']]['signed'].mean()), 'signed_rest': float(AU[~AU['unique']]['signed'].mean())}}
print({k: v for k, v in M['dbscan'].items() if k not in ('by_position', 'sensitivity')})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for P, g in SE.groupby('fine_pos'):
    ax[0].plot(g['kdist_quantile'], g['noise'], 'o-', color=GREY, lw=1.2, ms=3); ax[0].text(0.955, g['noise'].iloc[-1], P, fontsize=7, color=INK2, va='center')
ax[0].axvline(0.8, color=ORANGE, lw=1.2); ax[0].text(0.805, ax[0].get_ylim()[1] * .93, 'eps used', color=ORANGE, fontsize=8); ax[0].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[0].set_xlabel('eps, as a quantile of the 8th-nearest-neighbour distance'); ax[0].set_ylabel('points labelled noise'); ax[0].set_title('DBSCAN is sensitive to eps'); ax[0].set_xlim(.48, 1.0)
xx = np.arange(len(DT)); ax[1].bar(xx - .2, DT['noise_spark'], width=.38, color=BLUE); ax[1].bar(xx + .2, DT['hdbscan_noise'], width=.38, color=ORANGE)
ax[1].text(xx[0] - .2, DT['noise_spark'].iloc[0] + .01, 'DBSCAN\n(Spark)', color=BLUE, ha='center', fontsize=7.5); ax[1].text(xx[0] + .28, DT['hdbscan_noise'].iloc[0] + .01, 'HDBSCAN', color=ORANGE, ha='left', fontsize=7.5)
ax[1].set_xticks(xx, [f"{p}\nARI {a:.2f}" for p, a in zip(DT['fine_pos'], DT['ari_spark_vs_sklearn'])], fontsize=7.5); ax[1].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}'); ax[1].set_ylim(0, max(DT['noise_spark'].max(), DT['hdbscan_noise'].max()) * 1.3)
ax[1].set_title('Share of unique (noise) players; ARI = Spark DBSCAN vs sklearn'); ax[1].grid(axis='x', visible=False); savefig('p3a_dbscan')
_ = put_metrics('phase3a', M)
