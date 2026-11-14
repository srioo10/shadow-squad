# %% [markdown]
# # End-sem 03. Data-mining tasks in PySpark, one new algorithm per task
# None of these algorithms was used in the mid-sem notebook. Every section gives what the algorithm does, why it fits, the Spark code, the plan (once), the runtime, a metric and a figure.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np, time
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pyspark.sql import functions as F, Window as W
from pyspark.ml.feature import VectorAssembler, BucketedRandomProjectionLSH, MinHashLSH, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from common import *
sp = spark(packages='io.graphframes:graphframes-spark4_2.13:0.10.0'); RT, M = {}, {}
AF = MID['active_features']; MET = [f[:-4] for f in AF]; ZC = ['z_' + m for m in MET]; TC = ['hi_' + m for m in MET]; T0 = MID['TRAIN_LAST_T0']
POS = ['CB', 'FB', 'DM', 'CM', 'AM', 'W', 'ST']
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()
dist2 = lambda a, b: F.sqrt(F.aggregate(F.zip_with(a, b, lambda x, y: (x - y) * (x - y)), F.lit(0.0), lambda acc, x: acc + x))

# %% [markdown]
# ## 0. Feature table
# Style-neutral per-90 values (walk-forward style model from notebook 01) are logged and standardised inside each fine position, using only seasons up to the row's own. Binary traits mark a top-quartile percentile.

# %%
xr = sp.read.parquet('lake/player_xref').select('player', 'squad', 'yr', 'player_id', 'fine_pos', 'club_id')
base = sp.read.parquet('lake/fbref').drop('season', 'league').join(xr, ['player', 'squad', 'yr']).filter(F.col('fine_pos').isNotNull() & F.col('pos_group').isin('DF', 'MF', 'FW'))
w = W.partitionBy('fine_pos').orderBy('yr').rangeBetween(W.unboundedPreceding, 0)      # mean and SD from seasons up to the row's own season only
for f, m in zip(AF, MET):
    lg_ = F.log(F.col(f) + F.lit(MID['EPS_C'][f]))
    base = base.withColumn('z_' + m, (lg_ - F.mean(lg_).over(w)) / F.stddev(lg_).over(w)).withColumn('hi_' + m, (F.col(f + '_pct') >= 75).cast('double'))
feat = base.select('player_id', 'player', 'squad', 'comp', 'yr', 'club_id', 'fine_pos', 'pos_group', 'age_num', 'min_num', 'composite_perf_index', 'x_starter_share', 'x_dom', 'x_press', 'elite_club', 'feeder', 'table_pos',
                   *ZC, *TC, *AF, *[f.replace('_p90', '_p90raw') for f in AF], *[f + '_pct' for f in AF]).withColumn('za', F.array(*ZC)).withColumn('n_hi', sum(F.col(c) for c in TC))
feat.write.mode('overwrite').parquet('lake/features'); feat = sp.read.parquet('lake/features')
feat = VectorAssembler(inputCols=ZC, outputCol='zv').transform(feat); feat = VectorAssembler(inputCols=TC, outputCol='tv').transform(feat).cache()
print(f'feature rows: {feat.count():,}; by fine position: ' + str(dict(sorted(feat.groupBy("fine_pos").count().toPandas().values.tolist()))))

# %% [markdown]
# ## 1. Similarity search: locality-sensitive hashing
# **What it does.** `BucketedRandomProjectionLSH` projects each vector on random directions and cuts the line into buckets, so close vectors tend to share a bucket; only pairs that share one are compared. `MinHashLSH` does the same for sets under Jaccard distance.
# **Why here.** Every replacement event asks "who in the pool plays like the departed starter"; LSH answers that without comparing every pair. The data is small, so the exact join is also run to measure what LSH loses.

# %%
ev = sp.read.parquet('lake/replacement_events').withColumn('qid', F.concat_ws('_', 'C', 't', 'S'))
quer = (ev.select('qid', 'C', 't', 'S', 'P').distinct().join(feat.select(F.col('player_id').alias('S'), F.col('yr').alias('t'), F.col('za').alias('qa'), 'zv', 'tv', F.col('n_hi').alias('q_hi')), ['S', 't'])).cache()
cand = sp.read.parquet('lake/pool').select('player_id', 'yr').join(feat.select('player_id', 'yr', 'fine_pos', 'za', 'zv', 'tv', 'n_hi'), ['player_id', 'yr']).cache()
truth = ev.filter('N_in_pool').select('qid', F.col('N').alias('player_id')).distinct().toPandas()
t0 = time.time()
exact = (cand.join(quer.select('qid', 't', 'P', 'qa'), (F.col('yr') == F.col('t')) & (F.col('fine_pos') == F.col('P'))).withColumn('dist', dist2('za', 'qa'))
         .withColumn('rk', F.row_number().over(W.partitionBy('qid').orderBy('dist'))).select('qid', 'player_id', 'dist', 'rk')).cache()
n_exact = exact.count(); RT['exact same-position join'] = round(time.time() - t0, 2); EX = exact.toPandas()
top25 = set(map(tuple, EX.loc[EX['rk'] <= 25, ['qid', 'player_id']].values)); tru = set(map(tuple, truth.values))
def brp_join(T, B):
    m = BucketedRandomProjectionLSH(inputCol='zv', outputCol='h', bucketLength=B, numHashTables=T, seed=SEED).fit(cand)
    return (m.approxSimilarityJoin(cand, quer, 1e6, 'lsh_dist').filter('datasetA.yr = datasetB.t and datasetA.fine_pos = datasetB.P')
             .select(F.col('datasetB.qid').alias('qid'), F.col('datasetA.player_id').alias('player_id'), 'lsh_dist'))
grid = []
for B in (2.0, 4.0):
    for T in (1, 2, 4, 8):
        t0 = time.time(); J = brp_join(T, B).toPandas(); s = set(map(tuple, J[['qid', 'player_id']].values))
        grid.append({'bucketLength': B, 'numHashTables': T, 'pairs': len(s), 'set_share': len(s) / n_exact, 'recall_top25': len(s & top25) / len(top25), 'signing_recall': len(s & tru) / len(tru), 'seconds': round(time.time() - t0, 2)})
grid = pd.DataFrame(grid); display(grid.round(3))
ok = grid[grid['recall_top25'] >= 0.9]; best = (ok.sort_values('set_share') if len(ok) else grid.sort_values('recall_top25', ascending=False)).iloc[0]
brp = brp_join(int(best['numHashTables']), float(best['bucketLength'])); brp.explain()
BR = brp.toPandas().assign(in_brp=True)
t0 = time.time(); mh = MinHashLSH(inputCol='tv', outputCol='h', numHashTables=5, seed=SEED).fit(cand.filter('n_hi > 0'))
MH = (mh.approxSimilarityJoin(cand.filter('n_hi > 0'), quer.filter('q_hi > 0'), 0.8, 'jaccard_dist').filter('datasetA.yr = datasetB.t and datasetA.fine_pos = datasetB.P')
        .select(F.col('datasetB.qid').alias('qid'), F.col('datasetA.player_id').alias('player_id'), 'jaccard_dist')).toPandas().assign(in_minhash=True); RT['MinHash join'] = round(time.time() - t0, 2)
CA = EX.merge(BR, on=['qid', 'player_id'], how='left').merge(MH, on=['qid', 'player_id'], how='left'); CA[['in_brp', 'in_minhash']] = CA[['in_brp', 'in_minhash']].fillna(False).astype(bool)
CA['candidate'] = CA['in_brp'] | CA['in_minhash']; CA.to_parquet('lake/candidates.parquet', index=False)
key = lambda d: set(map(tuple, d[['qid', 'player_id']].values)); sb, sm, sc = key(CA[CA['in_brp']]), key(CA[CA['in_minhash']]), key(CA[CA['candidate']])
M['lsh'] = {'grid': grid.to_dict('records'), 'chosen': best.to_dict(), 'exact_pairs': n_exact, 'queries': int(EX['qid'].nunique()), 'mean_pool_per_query': float(n_exact / EX['qid'].nunique()),
            'minhash_pairs': len(sm), 'minhash_signing_recall': len(sm & tru) / len(tru), 'minhash_recall_top25': len(sm & top25) / len(top25), 'brp_minhash_jaccard': len(sb & sm) / len(sb | sm),
            'union_pairs': len(sc), 'union_set_share': len(sc) / n_exact, 'union_signing_recall': len(sc & tru) / len(tru), 'union_recall_top25': len(sc & top25) / len(top25), 'in_pool_signing_pairs': len(tru)}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in M['lsh'].items() if k not in ('grid', 'chosen')}); print('chosen:', best.round(3).to_dict())
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for B, g in grid.groupby('bucketLength'): ax.plot(g['set_share'], g['recall_top25'], 'o-', label=f'top-25 neighbours kept, bucket {B:g}'); ax.plot(g['set_share'], g['signing_recall'], 's--', label=f'real signings kept, bucket {B:g}')
ax.plot([0, 1], [0, 1], ':', color=GREY); ax.set_xlabel('share of same-position pairs compared'); ax.set_ylabel('recall'); ax.set_title('Euclidean LSH: work saved vs recall'); ax.legend(fontsize=7, loc='lower right'); savefig('p3_lsh_tradeoff')

# %% [markdown]
# ## 2. Sequential pattern mining: PrefixSpan
# **What it does.** PrefixSpan grows frequent sequences one item at a time and, for each prefix, searches only the part of the data that follows it (the projected database), so it never generates candidates that do not occur.
# **Why here.** A career is an ordered list of seasons. Each season is an itemset (club tier, league, role, form band); the question is which orders of seasons come before a move up.

# %%
from pyspark.ml.fpm import PrefixSpan
ab = F.create_map(*[F.lit(x) for kv in {'Premier League': 'PL', 'La Liga': 'LL', 'Bundesliga': 'BL', 'Serie A': 'SA', 'Ligue 1': 'L1'}.items() for x in kv])
rows = feat.select('player_id', 'yr', F.array(F.when(F.col('elite_club'), 'club=elite').when(F.col('table_pos') <= 12, 'club=mid').otherwise('club=low'), F.concat(F.lit('lg='), ab[F.col('comp')]),
                   F.when(F.col('min_num') >= 1800, 'role=starter').otherwise('role=rotation'),
                   F.when(F.col('composite_perf_index') >= 60, 'form=high').when(F.col('composite_perf_index') <= 40, 'form=low').otherwise('form=mid')).alias('items'))
lab = sp.read.parquet('lake/labels_v2').filter('feeder and player_id is not null and observed24').select('player_id', F.col('yr').alias('yr0'), (F.col('up_tier') >= 1).cast('int').alias('up'))
with timed('PrefixSpan', RT):
    seqs = (lab.join(rows, (lab.player_id == rows.player_id) & (rows.yr <= lab.yr0) & (rows.yr >= lab.yr0 - 3)).groupBy(lab.player_id, 'yr0', 'up')
               .agg(F.sort_array(F.collect_list(F.struct('yr', 'items'))).alias('s')).withColumn('sequence', F.col('s.items')).drop('s')).cache()
    ps = PrefixSpan(minSupport=0.03, maxPatternLength=4, maxLocalProjDBSize=32000000)
    pats = ps.findFrequentSequentialPatterns(seqs.filter(F.col('yr0') <= T0).select('sequence')).toPandas()
SQ = seqs.toPandas(); TR, TE = SQ[SQ['yr0'] <= T0], SQ[SQ['yr0'] > T0]
def contains(seq, pat):
    i = 0
    for its in seq:
        if i < len(pat) and set(pat[i]) <= set(its): i += 1
    return i == len(pat)
pats['pattern'] = pats['sequence'].map(lambda s: ' -> '.join('{' + ', '.join(sorted(x)) + '}' for x in s)); pats['n_items'] = pats['sequence'].map(lambda s: sum(len(x) for x in s)); pats['length'] = pats['sequence'].map(len)
hit_tr = np.array([[contains(s, p) for s in TR['sequence']] for p in pats['sequence']]); hit_te = np.array([[contains(s, p) for s in TE['sequence']] for p in pats['sequence']])
pats['n_train'] = hit_tr.sum(1); pats['conf_train'] = [TR['up'].values[h].mean() if h.any() else np.nan for h in hit_tr]; pats['lift_train'] = pats['conf_train'] / TR['up'].mean()
pats['n_test'] = hit_te.sum(1); pats['conf_test'] = [TE['up'].values[h].mean() if h.any() else np.nan for h in hit_te]; pats['lift_test'] = pats['conf_test'] / TE['up'].mean()
keep = pats[(pats['n_train'] >= 30) & (pats['lift_train'] >= 1.3) & (pats['n_items'] >= 2)].sort_values('lift_train', ascending=False)
flag = hit_te[keep.index].any(0) if len(keep) else np.zeros(len(TE), bool)
tab = [[int((flag & (TE['up'] == 1)).sum()), int((flag & (TE['up'] == 0)).sum())], [int((~flag & (TE['up'] == 1)).sum()), int((~flag & (TE['up'] == 0)).sum())]]
multi = keep[keep['length'] >= 2]
M['prefixspan'] = {'train_sequences': len(TR), 'test_sequences': len(TE), 'frequent_patterns': len(pats), 'patterns_len2plus': int((pats['length'] >= 2).sum()), 'kept_rules': len(keep), 'kept_multi_season': len(multi),
                   'base_rate_train': float(TR['up'].mean()), 'base_rate_test': float(TE['up'].mean()), 'flagged_test': int(flag.sum()), 'flagged_rate_test': float(TE['up'].values[flag].mean()) if flag.any() else None,
                   'oot_lift': float(TE['up'].values[flag].mean() / TE['up'].mean()) if flag.any() else None, 'fisher_p': float(stats.fisher_exact(tab, alternative='greater')[1]),
                   'rank_corr_train_test_lift': float(stats.spearmanr(keep['lift_train'], keep['lift_test'], nan_policy='omit')[0]) if len(keep) > 2 else None,
                   'top': keep.head(8)[['pattern', 'n_train', 'lift_train', 'n_test', 'lift_test']].round(2).to_dict('records'), 'top_multi_season': multi.head(8)[['pattern', 'n_train', 'lift_train', 'n_test', 'lift_test']].round(2).to_dict('records')}
print({k: v for k, v in M['prefixspan'].items() if not k.startswith('top')}); pd.set_option('display.max_colwidth', 120); display(multi.head(8)[['pattern', 'n_train', 'lift_train', 'n_test', 'lift_test']].round(2))
top = (multi if len(multi) >= 5 else keep).head(10).iloc[::-1]; fig, ax = plt.subplots(figsize=(8, 3.6)); y = np.arange(len(top))
ax.barh(y - .2, top['lift_train'], .4, label=f'train (t <= {T0})'); ax.barh(y + .2, top['lift_test'], .4, label='later seasons'); ax.axvline(1, color=GREY, ls=':'); ax.set_yticks(y, top['pattern'], fontsize=6); ax.set_xlabel('lift for moving up'); ax.legend(fontsize=7, loc='lower right'); ax.grid(axis='y', visible=False); ax.set_title('Career sequences that precede a move up'); savefig('p3_prefixspan')

# %% [markdown]
# ## 3. Clustering: bisecting k-means per fine position
# **What it does.** Bisecting k-means starts with one cluster and keeps splitting the cluster with the largest error in two with 2-means, which gives a hierarchy and is less sensitive to the starting points than plain k-means.
# **Why here.** Mid-sem archetypes were cut inside three coarse groups (DF/MF/FW). With Transfermarkt positions the split is made inside each of seven fine positions, so a full-back is never compared with a centre-back. The number of clusters is chosen on seasons up to 2019-20 and the model is refitted at every season cut-off on past seasons only.

# %%
from pyspark.ml.clustering import BisectingKMeans
sil, parts = [], []; ce = ClusteringEvaluator(featuresCol='zv'); kbest = {}
with timed('BisectingKMeans (k chosen on seasons <= 2019, then one refit per season cut-off)', RT):
    for P in POS:
        d = feat.filter(F.col('fine_pos') == P).select('player_id', 'yr', 'fine_pos', 'zv', *ZC).cache(); early = d.filter(F.col('yr') <= T0); best = (-1, None)
        for k in range(2, 7):
            s_ = ce.evaluate(BisectingKMeans(k=k, seed=SEED, featuresCol='zv', minDivisibleClusterSize=30).fit(early).transform(early)); sil.append({'fine_pos': P, 'k': k, 'silhouette': s_})
            if k >= 3 and s_ > best[0]: best = (s_, k)
        kbest[P] = best[1]
        for c in range(2017, 2025):      # the model of cut-off c is fitted on seasons <= c and labels every row, so later signings can be read in the vocabulary of that cut-off
            pr = BisectingKMeans(k=kbest[P], seed=SEED, featuresCol='zv', minDivisibleClusterSize=30).fit(d.filter(F.col('yr') <= c)).transform(d)
            parts.append(pr.select('player_id', 'yr', 'fine_pos', F.col('prediction').alias('cl'), *(ZC if c == 2024 else [])).toPandas().assign(cutoff=c))
        d.unpersist()
pr.explain()
sil = pd.DataFrame(sil); AW = pd.concat(parts); AW['archetype'] = AW['fine_pos'] + AW['cl'].astype(str); AW[['player_id', 'yr', 'fine_pos', 'cutoff', 'archetype']].to_parquet('lake/archetypes_wf.parquet', index=False)
AW[AW['yr'] == AW['cutoff']][['player_id', 'yr', 'fine_pos', 'archetype']].to_parquet('lake/archetypes.parquet', index=False)       # what the engine uses: a player's archetype as known at the end of his own season
AR = AW[AW['cutoff'] == 2024].copy()      # description only (table and figure below): the last model's labels for all seasons
cen = AR.groupby('archetype')[ZC].mean(); desc = cen.apply(lambda r: ', '.join(f"{c[2:]} {v:+.1f}" for c, v in r.sort_values(ascending=False).head(3).items()), axis=1)
LB = pd.read_parquet('lake/labels_v2'); LB = LB[LB['feeder'] & LB['player_id'].notna() & LB['observed24']].assign(up=lambda d: (d['up_tier'] >= 1).astype(int)).merge(AR[['player_id', 'yr', 'archetype']], on=['player_id', 'yr'])
ct = pd.crosstab(LB['archetype'], LB['up']); chi2, p_chi = stats.chi2_contingency(ct)[:2]; cv = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
arch = pd.DataFrame({'n': AR.groupby('archetype').size(), 'feeders': ct.sum(1), 'move_up_rate': (ct[1] / ct.sum(1)).round(3), 'defining traits (mean z)': desc}); display(arch)
M['bisecting'] = {'silhouette': sil.round(4).to_dict('records'), 'k_chosen': kbest, 'archetypes': int(AR['archetype'].nunique()), 'silhouette_chosen_mean': float(np.mean([sil[(sil.fine_pos == P) & (sil.k == k)]['silhouette'].iloc[0] for P, k in kbest.items()])),
                  'chi2': float(chi2), 'chi2_p': float(p_chi), 'cramers_v': cv, 'table': arch.reset_index().to_dict('records')}
print(f"{AR['archetype'].nunique()} archetypes {kbest}; archetype vs moving up: chi-square p = {p_chi:.4f}, Cramer's V = {cv:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1, 1.6]})
for P, g in sil.groupby('fine_pos'): ax[0].plot(g['k'], g['silhouette'], 'o-', label=P)
ax[0].set_xlabel('k'); ax[0].set_ylabel('silhouette (squared Euclidean)'); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title('Bisecting k-means: choosing k')
im = ax[1].imshow(cen.values.T, cmap=DIV, vmin=-1.5, vmax=1.5, aspect='auto'); ax[1].grid(False); ax[1].set_xticks(range(len(cen)), cen.index, rotation=90, fontsize=6); ax[1].set_yticks(range(len(MET)), MET, fontsize=6); ax[1].set_title('Archetype centroids (z within position)'); fig.colorbar(im, ax=ax[1]); savefig('p3_bisecting')

# %% [markdown]
# ## 4. Link analysis: PageRank on the transfer graph, and power-iteration clustering
# **What it does.** PageRank gives a node a high score when high-scoring nodes point to it. Here each permanent transfer is an edge; with edges pointing from buyer to seller, a club scores high when clubs that are themselves important buy from it. Power-iteration clustering (PIC) finds groups of clubs that trade mostly with each other from a few power iterations on the normalised affinity matrix.
# **Why here.** "Who sells to whom" is a graph question. The pipeline score of a feeder's club says how often and how high that club's players are bought. To avoid leakage the graph for season t only uses transfers dated before June of t+1.

# %%
from graphframes import GraphFrame
from pyspark.ml.clustering import PowerIterationClustering
tcl = (sp.read.parquet('lake/tm_transfers_clean').filter("not loan_out and not loan_return and not from_youth and f is not null and c is not null")
         .filter(~F.lower('fn').rlike('without club|retired|unknown|career break|^ban$') & ~F.lower('tn').rlike('without club|retired|unknown|career break|^ban$')).cache())
def senior(e, k=10):   # keep clubs that both buy and sell at least k players: drops academies, schools and one-off amateur clubs, which a reversed PageRank otherwise rewards
    okc = e.groupBy(F.col('c').alias('id')).count().filter(F.col('count') >= k).join(e.groupBy(F.col('f').alias('id')).count().filter(F.col('count') >= k), 'id').select('id')
    return e.join(okc.withColumnRenamed('id', 'f'), 'f').join(okc.withColumnRenamed('id', 'c'), 'c')
names = tcl.select(F.col('c').alias('id'), F.col('tn').alias('name')).union(tcl.select('f', 'fn')).groupBy('id').agg(F.first('name').alias('name')).toPandas().set_index('id')['name']
def pagerank(edges):
    v = edges.select(F.col('src').alias('id')).union(edges.select('dst')).distinct()
    return GraphFrame(v, edges).pageRank(resetProbability=0.15, maxIter=10).vertices
prs = []
with timed('GraphFrames PageRank (8 seasons x 2 directions)', RT):
    for t in range(2017, 2025):
        e = senior(tcl.filter(F.col('d') < F.make_date(F.lit(t + 1), F.lit(6), F.lit(1)))).cache()
        pipe = pagerank(e.select(F.col('c').alias('src'), F.col('f').alias('dst'))).withColumnRenamed('pagerank', 'pr_pipe')        # buyer -> seller: rank flows to clubs whose players are bought
        dest = pagerank(e.select(F.col('f').alias('src'), F.col('c').alias('dst'))).withColumnRenamed('pagerank', 'pr_dest')        # seller -> buyer: destination prestige
        deg = e.groupBy(F.col('f').alias('id')).agg(F.count('*').alias('sales'))
        prs.append(pipe.join(dest, 'id').join(deg, 'id', 'left').fillna(0).toPandas().assign(t=t))
pipe.explain()
PR = pd.concat(prs).rename(columns={'id': 'club_id'}); PR['club'] = PR['club_id'].map(names); PR.to_parquet('lake/club_pagerank.parquet', index=False)
from sklearn.metrics import roc_auc_score
LBp = LB.assign(up=(LB['up_tier'] >= 1).astype(int)).merge(PR, left_on=['club_id', 'yr'], right_on=['club_id', 't'], how='left'); ok = LBp['pr_pipe'].notna()
cl = LBp[ok].groupby(['club_id', 'yr']).agg(n=('up', 'size'), rate=('up', 'mean'), pr=('pr_pipe', 'first'), sales=('sales', 'first')).query('n >= 3')
top_pipe = PR[PR['t'] == 2024].nlargest(15, 'pr_pipe')[['club', 'pr_pipe', 'pr_dest', 'sales']]; display(top_pipe.round(2))
with timed('PowerIterationClustering', RT):
    und = (tcl.select(F.least('f', 'c').alias('src'), F.greatest('f', 'c').alias('dst')).groupBy('src', 'dst').agg(F.count('*').cast('double').alias('weight')))
    dg = und.select(F.col('src').alias('id'), 'weight').union(und.select('dst', 'weight')).groupBy('id').agg(F.sum('weight').alias('deg')).filter('deg >= 30')
    und = und.join(dg.select(F.col('id').alias('src')), 'src').join(dg.select(F.col('id').alias('dst')), 'dst').withColumn('src', F.col('src').cast('long')).withColumn('dst', F.col('dst').cast('long'))
    und = und.cache(); U = und.toPandas(); m2 = 2 * U['weight'].sum(); degs = pd.concat([U.groupby('src')['weight'].sum(), U.groupby('dst')['weight'].sum()]).groupby(level=0).sum()
    def modul(pc):
        cm = pc.set_index('id')['cluster']; same = U['src'].map(cm) == U['dst'].map(cm); return float(U.loc[same, 'weight'].sum() * 2 / m2 - ((degs.groupby(cm).sum() / m2) ** 2).sum())
    pic_grid = {}      # PIC stops early on purpose: too many iterations collapse every node onto the trivial eigenvector, so the iteration count is chosen by modularity (no labels involved)
    for it in (3, 5, 8, 12, 20):
        pc = PowerIterationClustering(k=8, maxIter=it, initMode='degree', weightCol='weight').assignClusters(und).toPandas(); pic_grid[it] = (modul(pc), pc)
    best_it = max(pic_grid, key=lambda i: pic_grid[i][0]); modularity, pic = pic_grid[best_it]
clubs_csv = pd.read_csv('data/transfermarkt/clubs.csv').set_index('club_id')['domestic_competition_id']; pic['country'] = pic['id'].map(clubs_csv); known = pic.dropna(subset=['country'])
comp_tab = pd.crosstab(known['cluster'], known['country']); purity = float(comp_tab.max(1).sum() / comp_tab.values.sum())
M['graph'] = {'edges': int(tcl.count()), 'clubs': int(len(names)), 'auc_pipeline_rank_alone': float(roc_auc_score(LBp.loc[ok, 'up'], LBp.loc[ok, 'pr_pipe'])), 'auc_sales_count_alone': float(roc_auc_score(LBp.loc[ok, 'up'], LBp.loc[ok, 'sales'])),
              'auc_dest_rank_alone': float(roc_auc_score(LBp.loc[ok, 'up'], LBp.loc[ok, 'pr_dest'])), 'feeders_scored': int(ok.sum()), 'club_level_spearman': float(stats.spearmanr(cl['pr'], cl['rate'])[0]), 'club_level_spearman_p': float(stats.spearmanr(cl['pr'], cl['rate'])[1]),
              'club_seasons': len(cl), 'top_pipeline_2023': top_pipe.round(3).to_dict('records'), 'pic_clubs': len(pic), 'pic_k': 8, 'pic_maxIter_chosen': best_it, 'pic_modularity_by_maxIter': {str(k): v[0] for k, v in pic_grid.items()}, 'pic_modularity': modularity, 'pic_country_purity': purity, 'pic_known_clubs': len(known),
              'pic_sizes': pic['cluster'].value_counts().sort_index().to_dict()}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in M['graph'].items() if k != 'top_pipeline_2023'})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); tp = top_pipe.iloc[::-1]; ax[0].barh(tp['club'], tp['pr_pipe']); ax[0].set_title('Talent-pipeline PageRank, transfers up to June 2025'); ax[0].tick_params(labelsize=7)
im = ax[1].imshow(np.log1p(comp_tab.values), cmap=SEQ, aspect='auto'); ax[1].grid(False); ax[0].grid(axis='y', visible=False); ax[1].set_xticks(range(comp_tab.shape[1]), comp_tab.columns, rotation=90, fontsize=6); ax[1].set_yticks(range(len(comp_tab)), [f'community {i}' for i in comp_tab.index], fontsize=7)
ax[1].set_title(f'Transfer communities (PIC) by league, modularity {modularity:.2f}'); savefig('p3_graph')

# %% [markdown]
# ## 5. Recommendation: ALS on the club x archetype matrix
# **What it does.** Alternating least squares factorises the (club, archetype) count matrix into low-rank club and archetype vectors, treating a count as confidence that the club likes that archetype (implicit feedback).
# **Why here.** Clubs have taste: some keep buying ball-playing centre-backs. The factor model also gives a score for archetypes a club has not bought yet, from clubs with similar buying history. One model per season cut-off, trained only on signings before it.

# %%
from pyspark.ml.recommendation import ALS
AWs = sp.createDataFrame(AW[['player_id', 'yr', 'archetype', 'fine_pos', 'cutoff']]).cache()
cxs = sp.read.parquet('lake/club_xref').select(F.col('club_id').alias('c'), 'squad')
sign = (tcl.join(cxs, 'c').join(AWs, (tcl.player_id == AWs.player_id) & (AWs.yr == tcl.season - 1)).select(F.col('c').cast('int').alias('club'), 'archetype', 'fine_pos', 'season', 'cutoff')).cache()
a_idx = {a: i for i, a in enumerate(sorted(AW['archetype'].unique()))}; a_df = sp.createDataFrame(pd.DataFrame({'archetype': list(a_idx), 'item': list(a_idx.values())}))
def als_scores(t):      # signings up to season t, with archetypes as labelled by the clustering model of cut-off t
    tr = sign.filter((F.col('cutoff') == t) & (F.col('season') <= t)).join(a_df, 'archetype').groupBy('club', 'item').agg(F.count('*').cast('float').alias('n'))
    mdl = ALS(userCol='club', itemCol='item', ratingCol='n', implicitPrefs=True, rank=6, maxIter=15, regParam=0.1, alpha=20.0, seed=SEED, coldStartStrategy='drop').fit(tr)
    return mdl.transform(tr.select('club').distinct().crossJoin(a_df.select('item'))), tr
with timed('ALS (7 cut-offs)', RT):
    taste = []
    for t in range(2018, 2025):   # the first archetype season is 2017-18, so the first cut-off with any past signing is 2018
        sc_, tr_ = als_scores(t); taste.append(sc_.toPandas().assign(t=t))
sc_.explain()
inv = {v: k for k, v in a_idx.items()}; TA = pd.concat(taste); TA['archetype'] = TA['item'].map(inv); TA = TA.rename(columns={'club': 'club_id', 'prediction': 'taste'})[['club_id', 't', 'archetype', 'taste']]; TA.to_parquet('lake/club_taste.parquet', index=False)
SG = sign.filter(F.col('cutoff') == T0).toPandas(); trn, tst = SG[SG['season'] <= T0], SG[SG['season'] > T0]
pop = trn['archetype'].value_counts(); own = trn.groupby(['club', 'archetype']).size(); als0 = TA[TA['t'] == T0].set_index(['club_id', 'archetype'])['taste']
def rr(row, how):
    opts = [a for a in a_idx if a.startswith(row.fine_pos) and a[len(row.fine_pos):].isdigit()]
    if how == 'pop': sc = {a: pop.get(a, 0) for a in opts}
    elif how == 'own': sc = {a: own.get((row.club, a), 0) + 1e-6 * pop.get(a, 0) for a in opts}
    else: sc = {a: als0.get((row.club, a), np.nan) for a in opts}; sc = sc if not any(np.isnan(v) for v in sc.values()) else {a: pop.get(a, 0) for a in opts}
    return 1 / (1 + sum(v > sc[row.archetype] for v in sc.values()))
res = {h: np.array([rr(r, h) for r in tst.itertuples()]) for h in ('pop', 'own', 'als')}
rng = np.random.default_rng(SEED); bs = rng.integers(0, len(tst), (2000, len(tst)))
M['als'] = {'train_signings': len(trn), 'test_signings': len(tst), 'clubs': int(SG['club'].nunique()), 'archetypes': len(a_idx),
            **{f'mrr_{h}': float(v.mean()) for h, v in res.items()}, **{f'top1_{h}': float((v == 1).mean()) for h, v in res.items()},
            'mrr_als_minus_pop_ci95': [float(x) for x in np.percentile((res['als'][bs] - res['pop'][bs]).mean(1), [2.5, 97.5])], 'mrr_als_minus_own_ci95': [float(x) for x in np.percentile((res['als'][bs] - res['own'][bs]).mean(1), [2.5, 97.5])]}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in M['als'].items()})
fig, ax = plt.subplots(figsize=(4.6, 3.2)); lab_ = {'pop': 'league-wide popularity', 'own': "club's own past counts", 'als': 'ALS factors'}
ax.bar([lab_[h] for h in res], [res[h].mean() for h in res], yerr=[1.96 * res[h].std() / np.sqrt(len(tst)) for h in res], capsize=3); ax.set_ylabel('mean reciprocal rank of the bought archetype'); ax.tick_params(axis='x', labelsize=7)
ax.set_title(f'Which archetype will the club buy? (signings after {T0})', fontsize=9); savefig('p3_als')

# %% [markdown]
# ## 6. Classification: linear SVM and multilayer perceptron
# **What they do.** `LinearSVC` finds the separating hyperplane with the largest margin (hinge loss). `MultilayerPerceptronClassifier` is a feed-forward neural network trained by back-propagation, so it can bend the boundary.
# **Why here.** Same question as the mid-sem classifiers (will this feeder move up?), same walk-forward split (train on seasons up to Y-2, test on Y), so the numbers can be put next to the mid-sem leaderboard. Spark logistic regression is run only as the like-for-like reference.

# %%
from pyspark.ml.classification import LinearSVC, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
FE_COLS = [f + '_pct' for f in AF] + ['age_num', 'composite_perf_index', 'starter_share'] + ['P_' + p for p in POS] + ['L_' + str(i) for i in range(1, 5)]
labs = sp.read.parquet('lake/labels_v2').filter('feeder and player_id is not null and observed24').select('player_id', 'yr', (F.col('up_tier') >= 1).cast('double').alias('y_v2'), (F.col('best_tier') >= 1).cast('double').alias('y_v1'), 'labeled')
D = feat.join(labs, ['player_id', 'yr']).withColumn('starter_share', F.coalesce('x_starter_share', F.lit(0.0)))
for p in POS: D = D.withColumn('P_' + p, (F.col('fine_pos') == p).cast('double'))
for i, L_ in enumerate(LEAGUES[1:], start=1): D = D.withColumn('L_' + str(i), (F.col('comp') == L_).cast('double'))
D = D.fillna(0, FE_COLS).cache()
def models(d):
    return {'LinearSVC': LinearSVC(maxIter=100, regParam=0.1, weightCol='w', labelCol='y'), 'MLP (16-8)': MultilayerPerceptronClassifier(layers=[d, 16, 8, 2], maxIter=200, seed=SEED, labelCol='y'),
            'Logistic regression (reference)': LogisticRegression(regParam=0.05, weightCol='w', labelCol='y')}
bce = BinaryClassificationEvaluator(labelCol='y', rawPredictionCol='rawPrediction'); rowsF, scores = [], []
with timed('LinearSVC + MLP + LR walk-forward', RT):
    for lab_name, ycol, seasons in (('mid-sem label', 'y_v1', (2019, 2020, 2021)), ('transfer label', 'y_v2', (2019, 2020, 2021, 2022, 2023))):
        Dl = D.withColumn('y', F.col(ycol)).filter('labeled' if ycol == 'y_v1' else 'true')
        for Y in seasons:
            tr = Dl.filter(F.col('yr') <= Y - 2); te = Dl.filter(F.col('yr') == Y); pos_rate = tr.agg(F.mean('y')).first()[0]
            tr = tr.withColumn('w', F.when(F.col('y') == 1, 0.5 / pos_rate).otherwise(0.5 / (1 - pos_rate)))
            for name, est in models(len(FE_COLS)).items():
                pm = Pipeline(stages=[VectorAssembler(inputCols=FE_COLS, outputCol='raw'), StandardScaler(inputCol='raw', outputCol='features', withMean=True), est]).fit(tr); pr = pm.transform(te)
                out = pr.select('player_id', 'yr', 'y', vector_to_array('rawPrediction')[1].alias('score')).toPandas()
                rowsF.append({'label': lab_name, 'season': Y, 'model': name, 'n_test': len(out), 'AUC': bce.evaluate(pr), 'hits@25': int(out.nlargest(25, 'score')['y'].sum()), 'base_rate': float(out['y'].mean())})
                if ycol == 'y_v2': scores.append(out.assign(model=name))
pr.select('player_id', 'rawPrediction').explain()
CF = pd.DataFrame(rowsF); pd.concat(scores).to_parquet('lake/clf_scores.parquet', index=False)
summ = CF.groupby(['label', 'model']).agg(mean_AUC=('AUC', 'mean'), hits=('hits@25', 'sum'), picks=('season', lambda s: 25 * len(s)), base_rate=('base_rate', 'mean')).reset_index(); summ['precision@25'] = summ['hits'] / summ['picks']; summ['lift'] = summ['precision@25'] / summ['base_rate']; display(summ.round(3))
BASE = json.load(open('outputs/baseline_metrics.json')); mid_lb = {r['method']: {'mean_AUC': r['mean_AUC'], 'hits': r['hits@K'], 'picks': r['picks']} for r in BASE['leaderboard']}
M['classification'] = {'n_features': len(FE_COLS), 'by_season': CF.round(4).to_dict('records'), 'summary': summ.round(4).to_dict('records'), 'midsem_leaderboard': mid_lb}
fig, ax = plt.subplots(figsize=(6.4, 3.4)); s1 = summ[summ['label'] == 'mid-sem label']; allm = [(m, v['mean_AUC'], 'mid-sem (sklearn)') for m, v in mid_lb.items()] + [(r.model, r.mean_AUC, 'end-sem (Spark)') for r in s1.itertuples()]
allm.sort(key=lambda x: x[1]); ax.barh([a[0] for a in allm], [a[1] for a in allm], color=[ORANGE if a[2].startswith('end') else GREY for a in allm]); ax.axvline(0.5, color=GREY, ls=':'); ax.set_xlim(0.45, 0.75)
ax.set_xlabel('mean walk-forward AUC, mid-sem label, test seasons 2019-2021'); ax.tick_params(labelsize=7); ax.set_title('New Spark classifiers (orange) next to the mid-sem leaderboard', fontsize=9); savefig('p3_classification')

# %% [markdown]
# ## 7. Outlier detection: distance-based (k-nearest-neighbour) score
# **What it does.** A point is an outlier when even its nearest neighbours are far away; the score is the mean distance to its k = 10 nearest neighbours (Ramaswamy-style distance-based outliers), computed by a Spark self-join inside each archetype.
# **Why here.** A "hidden gem" is a player whose profile is unusual for his archetype and good. The flag is the top decile of the score inside the archetype combined with an above-median composite index.

# %%
with timed('kNN outlier self-join', RT):
    A = feat.select('player_id', 'yr', 'za', 'composite_perf_index').join(AWs.select('player_id', 'yr', 'archetype', 'cutoff'), ['player_id', 'yr']).cache()
    a_, b_ = A.filter('yr = cutoff').alias('a'), A.alias('b')      # a player of season y is compared only with players of seasons <= y, in the archetypes of the season-y model
    knn = (a_.join(b_, (F.col('a.cutoff') == F.col('b.cutoff')) & (F.col('a.archetype') == F.col('b.archetype')) & (F.col('b.yr') <= F.col('a.yr')) & (F.col('a.player_id') != F.col('b.player_id')))
             .select(F.col('a.player_id').alias('player_id'), F.col('a.yr').alias('yr'), dist2('a.za', 'b.za').alias('d'))
             .withColumn('r', F.row_number().over(W.partitionBy('player_id', 'yr').orderBy('d'))).filter('r <= 10').groupBy('player_id', 'yr').agg(F.mean('d').alias('knn_dist')))
    out = (A.filter('yr = cutoff').join(knn, ['player_id', 'yr']).withColumn('knn_pct', F.percent_rank().over(W.partitionBy('yr', 'archetype').orderBy('knn_dist')))
            .withColumn('gem', (F.col('knn_pct') >= 0.9) & (F.col('composite_perf_index') >= F.percentile_approx('composite_perf_index', 0.5).over(W.partitionBy('yr', 'archetype')))))
    OUT = out.select('player_id', 'yr', 'archetype', 'knn_dist', 'knn_pct', 'gem').toPandas()
knn.explain(); OUT.to_parquet('lake/outliers.parquet', index=False)
LO = LB.merge(OUT, on=['player_id', 'yr']); rho, p_rho = stats.spearmanr(LO['knn_pct'], LO['up']); LO['decile'] = pd.qcut(LO['knn_pct'], 10, labels=False, duplicates='drop') + 1; dec = LO.groupby('decile')['up'].agg(['mean', 'size'])
g1, g0 = LO[LO['gem']], LO[~LO['gem']]; fp = stats.fisher_exact([[g1['up'].sum(), (1 - g1['up']).sum()], [g0['up'].sum(), (1 - g0['up']).sum()]], alternative='greater')[1]
mid_l = pd.read_parquet('lake/midsem/lake.parquet', columns=['player', 'squad', 'yr', 'anom_if', 'anom_lof']).merge(pd.read_parquet('lake/player_xref')[['player', 'squad', 'yr', 'player_id']], on=['player', 'squad', 'yr']).merge(OUT, on=['player_id', 'yr'])
M['outlier'] = {'rows_scored': len(OUT), 'feeders': len(LO), 'spearman_score_vs_up': float(rho), 'spearman_p': float(p_rho), 'gems_flagged': int(len(g1)), 'gem_move_up_rate': float(g1['up'].mean()), 'rest_move_up_rate': float(g0['up'].mean()), 'gem_fisher_p': float(fp),
                'top_decile_rate': float(dec['mean'].iloc[-1]), 'bottom_decile_rate': float(dec['mean'].iloc[0]), 'spearman_vs_midsem_iforest': float(stats.spearmanr(mid_l['knn_dist'], mid_l['anom_if'])[0]), 'spearman_vs_midsem_lof': float(stats.spearmanr(mid_l['knn_dist'], mid_l['anom_lof'])[0])}
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M['outlier'].items()})
fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar(dec.index, dec['mean']); ax.axhline(LO['up'].mean(), color=GREY, ls=':', label='feeder base rate'); ax.set_xlabel('decile of k-NN outlier score inside the archetype'); ax.set_ylabel('move-up rate'); ax.legend(fontsize=7); ax.set_title('Do unusual profiles get bought?', fontsize=9); savefig('p3_outlier')

# %% [markdown]
# ## 8. Stream mining: Structured Streaming replay with a Count-Min Sketch
# **What it does.** The 2023-24 Understat matches are replayed one week per micro-batch. A stateful streaming aggregation keeps each team's running deep completions for and against, which gives a live dominance value. A Count-Min Sketch counts deep completions per team in fixed memory (it can only over-count, by at most eps x total with the stated confidence) and is merged batch by batch.
# **Why here.** Team style is the context every player number is read in. The stream shows how early in a season that context is known.

# %%
import shutil, glob
LGU = {'EPL': 'Premier League', 'La_Liga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie_A': 'Serie A', 'Ligue_1': 'Ligue 1'}; SDIR = 'lake/_stream_in'; shutil.rmtree(SDIR, ignore_errors=True); os.makedirs(SDIR)
mi = pd.concat([pd.read_csv(f'data/understat/understats/{k}/match_info.csv', sep=None, engine='python').assign(comp=v) for k, v in LGU.items()]); mi = mi[mi['season'] == 2023]; mi['date'] = pd.to_datetime(mi['date'])
long = pd.concat([mi[['comp', 'date', 'team_h', 'h_deep', 'a_deep']].set_axis(['comp', 'date', 'team', 'deep', 'deep_allowed'], axis=1), mi[['comp', 'date', 'team_a', 'a_deep', 'h_deep']].set_axis(['comp', 'date', 'team', 'deep', 'deep_allowed'], axis=1)])
long[['deep', 'deep_allowed']] = long[['deep', 'deep_allowed']].astype(int); long['week'] = (long['date'] - long['date'].min()).dt.days // 7
for i, (wk, g) in enumerate(sorted(long.groupby('week'), key=lambda x: x[0])):
    fn = f'{SDIR}/week_{int(wk):02d}.csv'; g.drop(columns='date').to_csv(fn, index=False); os.utime(fn, (1700000000 + i, 1700000000 + i))      # modification times fix the replay order
stream = lambda: sp.readStream.schema('comp string, team string, deep int, deep_allowed int, week int').option('header', True).option('maxFilesPerTrigger', 1).csv(SDIR)
snaps, CMS = [], {}
def on_state(df, bid): snaps.append(df.toPandas().assign(batch=bid))
def on_raw(df, bid):
    e = df.filter('deep > 0').select('team', F.explode(F.sequence(F.lit(1), F.col('deep'))).alias('i'))
    for eps in (0.1, 0.01):
        s = e._jdf.stat().countMinSketch('team', eps, 0.95, SEED); CMS[eps] = s if eps not in CMS else CMS[eps].mergeInPlace(s)
with timed('Structured Streaming replay', RT):
    agg = stream().groupBy('comp', 'team').agg(F.sum('deep').alias('deep'), F.sum('deep_allowed').alias('deep_allowed'), F.count('*').alias('matches'))
    q1 = agg.writeStream.outputMode('complete').foreachBatch(on_state).trigger(availableNow=True).start(); q2 = stream().writeStream.foreachBatch(on_raw).trigger(availableNow=True).start(); q1.awaitTermination(); q2.awaitTermination()
agg.explain()
ctx = pd.read_parquet('lake/midsem/ctx.parquet'); fin = ctx[ctx['yr'] == 2023].set_index(['comp', 'title'])['x_dom']; rs = []
for s in snaps:
    s = s.copy(); sh = (s['deep'] + 0.5) / (s['deep'] + s['deep_allowed'] + 1); s['x'] = np.log(sh / (1 - sh)); s['x'] -= s.groupby('comp')['x'].transform('mean'); s['fin'] = [fin.get((c, t), np.nan) for c, t in zip(s['comp'], s['team'])]; s = s.dropna(subset=['fin'])
    rs.append({'batch': int(s['batch'].iloc[0]) + 1, 'teams': len(s), 'mean_matches': float(s['matches'].mean()), 'r_with_final': float(np.corrcoef(s['x'], s['fin'])[0, 1])})
RS = pd.DataFrame(rs); exact_c = long.groupby('team')['deep'].sum()
cms_err = {str(eps): {'width': int(c.width()), 'depth': int(c.depth()), 'mean_rel_error': float(np.mean([(c.estimateCount(t) - v) / v for t, v in exact_c.items()])), 'max_rel_error': float(np.max([(c.estimateCount(t) - v) / v for t, v in exact_c.items()]))} for eps, c in CMS.items()}
first90 = RS[RS['r_with_final'] >= 0.9]
M['stream'] = {'micro_batches': len(RS), 'match_rows': len(long), 'teams': int(long['team'].nunique()), 'teams_matched_to_final': int(RS['teams'].iloc[-1]), 'r_after_5_batches': float(RS['r_with_final'].iloc[4]), 'r_after_10_batches': float(RS['r_with_final'].iloc[9]), 'r_final': float(RS['r_with_final'].iloc[-1]),
               'first_batch_r_ge_0.9': int(first90['batch'].iloc[0]) if len(first90) else None, 'mean_matches_at_r_0.9': float(first90['mean_matches'].iloc[0]) if len(first90) else None, 'cms': cms_err, 'total_deep_events': int(exact_c.sum())}
print(M['stream'])
fig, ax = plt.subplots(figsize=(5.4, 3.2)); ax.plot(RS['mean_matches'], RS['r_with_final'], 'o-', ms=3); ax.axhline(0.9, color=GREY, ls=':'); ax.set_xlabel('mean matches played per team so far'); ax.set_ylabel('correlation with end-of-season x_dom'); ax.set_title('Streaming team dominance converges to the season value', fontsize=9); savefig('p3_stream')
_ = put_metrics('phase3', {**M, 'runtime_s': RT})
