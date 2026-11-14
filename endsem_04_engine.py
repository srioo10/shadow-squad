# %% [markdown]
# # End-sem 04. The Shadow Squad engine
# Style translation and its check on real movers, the candidate scores, and the learning-to-rank fusion.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np, pickle, time
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pyspark.sql import functions as F, Window as W
from common import *
import engine as E
sp = spark(); RT, M = {}, {}
def savefig(name): plt.tight_layout(); plt.savefig(f'outputs/figs/{name}.png'); plt.close()

# %% [markdown]
# ## 1. Market value at t and the hazard score (inputs the engine reads from the lake)
# Value = last Transfermarkt valuation dated before 1 June after season t, the day the transfer window used for the events opens, so no valuation can react to the signing it is used to predict. Hazard = discrete-time survival model: Spark logistic regression on person-period rows, refitted for every season t on rows whose 12-month outcome was already known (seasons up to t-1).

# %%
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
with timed('value at t', RT):
    ft = sp.read.parquet('lake/features').select('player_id', 'yr')
    v = sp.read.parquet('lake/tm_valuations').select('player_id', 'date', F.col('market_value_in_eur').cast('double').alias('value'))
    val = (ft.join(v, (ft.player_id == v.player_id) & (v.date < F.make_date(ft.yr + 1, F.lit(6), F.lit(1)))).select(ft.player_id, 'yr', 'date', 'value')
             .withColumn('r', F.row_number().over(W.partitionBy('player_id', 'yr').orderBy(F.desc('date')))).filter('r = 1').drop('r', 'date'))
    val.toPandas().to_parquet('lake/value_at_t.parquet', index=False)
lake = E.Lake(); FT = lake.F
HF = E.ZC + ['dur', 'age_num', 'composite_perf_index', 'min_num'] + ['P_' + p for p in sorted(FT['fine_pos'].unique())] + ['L_' + str(i) for i in range(1, 5)]
X = FT[['player_id', 'yr', 'fine_pos', 'comp', 'dur', 'age_num', 'composite_perf_index', 'min_num'] + E.ZC].copy()
for p in sorted(FT['fine_pos'].unique()): X['P_' + p] = (X['fine_pos'] == p).astype(float)
for i, L_ in enumerate(LEAGUES[1:], start=1): X['L_' + str(i)] = (X['comp'] == L_).astype(float)
PP = pd.read_parquet('lake/person_period.parquet')[['player_id', 'yr', 'event']]
xs = sp.createDataFrame(X[['player_id', 'yr'] + HF].fillna(0)).cache(); pp = sp.createDataFrame(PP).join(xs, ['player_id', 'yr']).cache()
pipe = lambda: Pipeline(stages=[VectorAssembler(inputCols=HF, outputCol='raw'), StandardScaler(inputCol='raw', outputCol='features', withMean=True), LogisticRegression(labelCol='event', regParam=0.05)])
hz, HM = [], {'features': HF}
with timed('discrete-time hazard, Spark LR, 7 refits', RT):
    for t in range(2018, 2025):
        mdl = pipe().fit(pp.filter(F.col('yr') <= t - 1)); HM[t] = (mdl.stages[1].mean.toArray(), mdl.stages[1].std.toArray(), mdl.stages[2].coefficients.toArray(), float(mdl.stages[2].intercept)); hz.append(mdl.transform(xs.filter(F.col('yr') == t)).select('player_id', 'yr', vector_to_array('probability')[1].alias('hazard')).toPandas())
mdl.transform(xs).select('probability').explain()
HZ = pd.concat(hz); HZ.to_parquet('lake/hazard.parquet', index=False); pickle.dump(HM, open('lake/hazard_models.pkl', 'wb'))      # coefficients kept so the red-team notebook can re-score altered profiles; lake = E.Lake(); FT = lake.F
print(f"value known for {FT['value'].notna().sum():,} of {len(FT):,} player-seasons; hazard scored for {len(HZ):,} player-seasons (2018-2024; no earlier season to learn from for 2017)")

# %% [markdown]
# ## 2. Style translation, checked on players who really changed club
# For every player with >= 900 minutes in two consecutive seasons at different clubs, his per-90 numbers at the new club are predicted three ways: (a) last season's raw numbers unchanged, (b) his style-neutral numbers, (c) his neutral numbers translated into the new club. The new club's style is taken from the season **before** he arrives, which is what a scout knows; clubs with no top-flight row that season get the neutral value.

# %%
A = FT.copy(); B = FT[['player_id', 'yr', 'squad', 'comp', 'x_dom', 'x_press', 'pos_group'] + E.RAW].copy(); B['yr'] -= 1
mvr = A.merge(B, on=['player_id', 'yr'], suffixes=('', '_n')); mvr = mvr[(mvr['squad'] != mvr['squad_n']) & (mvr['pos_group'] == mvr['pos_group_n'])].reset_index(drop=True)
prev = FT.groupby(['comp', 'squad', 'yr'])[['x_dom', 'x_press']].first(); known = [prev.index.isin([k])[0] if False else (k in prev.index) for k in zip(mvr['comp_n'], mvr['squad_n'], mvr['yr'])]
xd = np.array([prev.loc[k, 'x_dom'] if ok else 0.0 for k, ok in zip(zip(mvr['comp_n'], mvr['squad_n'], mvr['yr']), known)]); xp = np.array([prev.loc[k, 'x_press'] if ok else 0.0 for k, ok in zip(zip(mvr['comp_n'], mvr['squad_n'], mvr['yr']), known)])
act = mvr[[r + '_n' for r in E.RAW]].to_numpy(); pa = mvr[E.RAW].to_numpy(); pb = mvr[E.AF].to_numpy()
pc = lake.translate(pb, mvr['pos_group'].to_numpy(), xd, xp, mvr['comp_n'].to_numpy(), mvr['yr'].to_numpy()); pc_or = lake.translate(pb, mvr['pos_group'].to_numpy(), mvr['x_dom_n'].to_numpy(), mvr['x_press_n'].to_numpy(), mvr['comp_n'].to_numpy(), mvr['yr'].to_numpy())
scale = FT[E.RAW].std().to_numpy(); err = {k: np.abs(p - act) / scale for k, p in (('a_raw', pa), ('b_neutral', pb), ('c_translated', pc), ('c_oracle_style', pc_or))}
tabs = []
for g in ('DF', 'MF', 'FW', 'all'):
    msk = np.ones(len(mvr), bool) if g == 'all' else (mvr['pos_group'] == g).to_numpy()
    for i, m in enumerate(E.MET): tabs.append({'pos': g, 'metric': m, 'n': int(msk.sum()), **{k: float(e[msk, i].mean()) for k, e in err.items()}})
TV = pd.DataFrame(tabs); TV['c_beats_a'] = TV['c_translated'] < TV['a_raw']; TV.to_csv('outputs/translation_validation.csv', index=False)
rng = np.random.default_rng(SEED); bs = rng.integers(0, len(mvr), (2000, len(mvr))); pe = {k: e.mean(1) for k, e in err.items()}
ci = lambda d: [float(x) for x in np.percentile(d[bs].mean(1), [2.5, 97.5])]
cross = (mvr['comp'] != mvr['comp_n']).to_numpy(); big = (np.abs(xd - mvr['x_dom'].to_numpy()) > 0.5)
M['translation'] = {'movers': len(mvr), 'cross_league_movers': int(cross.sum()), 'new_club_style_known': int(sum(known)), 'mean_scaled_mae': {k: float(v.mean()) for k, v in pe.items()},
    'c_minus_a_ci95': ci(pe['c_translated'] - pe['a_raw']), 'c_minus_b_ci95': ci(pe['c_translated'] - pe['b_neutral']), 'b_minus_a_ci95': ci(pe['b_neutral'] - pe['a_raw']), 'oracle_minus_a_ci95': ci(pe['c_oracle_style'] - pe['a_raw']),
    'cells_c_beats_a': int(TV[TV['pos'] != 'all']['c_beats_a'].sum()), 'cells_total': int((TV['pos'] != 'all').sum()), 'metrics_c_beats_a_all_positions': TV[(TV['pos'] == 'all') & TV['c_beats_a']]['metric'].tolist(),
    'cross_league_only': {k: float(v[cross].mean()) for k, v in pe.items()}, 'big_style_change_only': {'n': int(big.sum()), **{k: float(v[big].mean()) for k, v in pe.items()}},
    'by_pos': {g: {k: float(TV[TV['pos'] == g][k].mean()) for k in err} for g in ('DF', 'MF', 'FW')}}
print({k: v for k, v in M['translation'].items()})
piv = TV[TV['pos'] == 'all'].set_index('metric'); fig, ax = plt.subplots(figsize=(8, 3.2)); x = np.arange(len(piv))
for j, (k, lab_) in enumerate((('a_raw', '(a) raw, unchanged'), ('b_neutral', '(b) style-neutral'), ('c_translated', '(c) translated into the new club'))): ax.bar(x + (j - 1) * .27, piv[k], .27, label=lab_)
ax.set_xticks(x, piv.index, rotation=60, fontsize=7); ax.set_ylabel('MAE at the new club (in SD of the metric)'); ax.legend(fontsize=7); ax.set_title(f'Predicting a mover\'s next-season per-90s ({len(mvr)} real club changes)', fontsize=9); savefig('p4_translation')

# %% [markdown]
# ## 3. Candidate scores for every replacement event
# One row per (event, same-position pool player). `sim_translated` compares the candidate translated into the buying club with the departed starter's actual numbers there. `traj_sim` compares the last seasons, age-aligned where the starter's seasons at those ages are in the data.

# %%
EV = pd.read_parquet('lake/replacement_events'); EV['qid'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str)
Q = EV.sort_values(['qid', 'N'])[['qid', 'C', 't', 'S', 'P', 'club', 'S_name']].drop_duplicates('qid')
with timed('pair features', RT): PF = lake.pair_features(Q)
CA = pd.read_parquet('lake/candidates.parquet')[['qid', 'player_id', 'dist', 'candidate', 'in_brp', 'in_minhash']]; PF = PF.merge(CA, on=['qid', 'player_id'], how='left'); PF['candidate'] = PF['candidate'].fillna(False).astype(bool)
pos_ = EV[EV['N_in_pool']][['qid', 'N']].drop_duplicates().assign(y=1).rename(columns={'N': 'player_id'}); PF = PF.merge(pos_, on=['qid', 'player_id'], how='left'); PF['y'] = PF['y'].fillna(0).astype(int)
PF['q_in_pool'] = PF.groupby('qid')['y'].transform('max') == 1; PF.to_parquet('lake/pairs.parquet', index=False)
M['value_cutoff'] = 'last valuation before 1 June after season t'
M['pairs'] = {'rows': len(PF), 'queries': int(PF['qid'].nunique()), 'in_pool_queries': int(PF[PF['q_in_pool']]['qid'].nunique()), 'positives': int(PF['y'].sum()), 'positives_in_candidate_set': int(PF[PF['candidate']]['y'].sum()),
              'traj_available': float(PF['traj_sim'].notna().mean()), 'traj_age_aligned_share': float(PF['traj_age_aligned'].mean()), 'value_missing': float(PF['value'].isna().mean()), 'hazard_missing': float(PF['hazard'].isna().mean()),
              'mean_feature_pos_vs_neg': {f: [float(PF[PF['q_in_pool'] & (PF['y'] == 1)][f].mean()), float(PF[PF['q_in_pool'] & (PF['y'] == 0)][f].mean())] for f in E.FEATS + ['sim_raw', 'sim_neutral']}}
print({k: v for k, v in M['pairs'].items() if k != 'mean_feature_pos_vs_neg'}); display(pd.DataFrame(M['pairs']['mean_feature_pos_vs_neg'], index=['real signing', 'other pool players']).T.round(3))

# %% [markdown]
# ### Trajectory: LSH on concatenated season vectors (Spark) and DTW as the out-of-course comparison
# Players with two consecutive seasons get a 32-dimensional vector [z(t), z(t-1)]; Euclidean LSH finds near trajectories. Dynamic time warping, which allows seasons to stretch, is run on the in-pool events and compared with the simple aligned cosine.

# %%
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from dtaidistance import dtw_ndim
two = FT.merge(FT[['player_id', 'yr'] + E.ZC].assign(yr=lambda d: d['yr'] + 1), on=['player_id', 'yr'], suffixes=('', '_prev')); cols2 = E.ZC + [c + '_prev' for c in E.ZC]
tw = VectorAssembler(inputCols=cols2, outputCol='v').transform(sp.createDataFrame(two[['player_id', 'yr', 'fine_pos'] + cols2])).select('player_id', 'yr', 'fine_pos', 'v').cache()
with timed('trajectory LSH self-join', RT):
    lm = BucketedRandomProjectionLSH(inputCol='v', outputCol='h', bucketLength=2.0, numHashTables=2, seed=SEED).fit(tw)
    lj = lm.approxSimilarityJoin(tw, tw, 1e6, 'd').filter('datasetA.fine_pos = datasetB.fine_pos and datasetA.player_id != datasetB.player_id').select(F.col('datasetA.player_id').alias('a'), F.col('datasetA.yr').alias('ay'), F.col('datasetB.player_id').alias('b'), F.col('datasetB.yr').alias('by'), 'd')
    n_lsh = lj.count(); near = lj.withColumn('r', F.row_number().over(W.partitionBy('a', 'ay').orderBy('d'))).filter('r <= 10').count()
V2 = two[cols2].to_numpy(); fp2 = two['fine_pos'].to_numpy(); rs = rng.choice(len(two), 300, replace=False); hit = tot = 0; pairs_exact = sum(int((fp2 == p).sum()) ** 2 for p in np.unique(fp2))
LJ = lj.toPandas(); lset = set(zip(LJ['a'], LJ['ay'], LJ['b'], LJ['by']))
for i in rs:
    same = np.flatnonzero((fp2 == fp2[i]) & (two['player_id'].to_numpy() != two['player_id'].iat[i])); nn = same[np.argsort(np.linalg.norm(V2[same] - V2[i], axis=1))[:10]]
    hit += sum((two['player_id'].iat[i], two['yr'].iat[i], two['player_id'].iat[j], two['yr'].iat[j]) in lset for j in nn); tot += len(nn)
seq = lambda pid, t: np.array([lake.Z[lake.idx[(pid, t - k)]] for k in (2, 1, 0) if (pid, t - k) in lake.idx])
IP = PF[PF['q_in_pool']].copy()
with timed('DTW on in-pool events', RT): IP['dtw'] = [dtw_ndim.distance(seq(a, t), seq(b, t)) / max(len(seq(a, t)), len(seq(b, t))) for a, b, t in zip(IP['player_id'], IP['S'], IP['t'])]
rk = lambda col, asc: IP.assign(r=IP.groupby('qid')[col].rank(ascending=asc, method='average'))[IP['y'] == 1]['r']
M['trajectory'] = {'two_season_vectors': len(two), 'lsh_pairs': int(n_lsh), 'lsh_pair_share': float(n_lsh / pairs_exact), 'lsh_recall_top10': float(hit / tot), 'spearman_traj_vs_dtw': float(stats.spearmanr(IP['traj_sim'], -IP['dtw'], nan_policy='omit')[0]),
                   'median_rank_true_signing_traj_sim': float(rk('traj_sim', False).median()), 'median_rank_true_signing_dtw': float(rk('dtw', True).median()), 'recall10_traj_sim': float((rk('traj_sim', False) <= 10).mean()), 'recall10_dtw': float((rk('dtw', True) <= 10).mean())}
print(M['trajectory'])

# %% [markdown]
# ### The large event set: every summer signing by a five-league club
# Query = (club, season, position). The reference profile is what the club fielded at that position that season (minutes-weighted), and the candidates are all 900-minute players of that position at other clubs.

# %%
SA = pd.read_parquet('lake/signing_events').sort_values(['qid', 'N']); QA = SA[['qid', 'C', 't', 'P', 'league', 'club']].drop_duplicates('qid')
with timed('pair features, all signings', RT): PA = lake.pair_features_club(QA)
PA = PA.merge(SA[['qid', 'N']].drop_duplicates().assign(y=1).rename(columns={'N': 'player_id'}), on=['qid', 'player_id'], how='left'); PA['y'] = PA['y'].fillna(0).astype(int); PA['q_in_pool'] = PA.groupby('qid')['y'].transform('max') == 1
PA = PA[PA['q_in_pool']].drop(columns=[c for c in PA.columns if c.startswith('tr_')]); PA.to_parquet('lake/pairs_all.parquet', index=False)
M['pairs_all'] = {'rows': len(PA), 'queries': int(PA['qid'].nunique()), 'positives': int(PA['y'].sum()), 'mean_candidates': float(PA.groupby('qid').size().mean()), 'reference_profile_missing': float(PA['sim_translated'].isna().mean())}; print(M['pairs_all'])

# %% [markdown]
# ## 4. Fusion: learning to rank
# One query per replacement event, label 1 for the player actually signed, 0 for the other LSH candidates. Walk-forward: the model that ranks events of season Y is trained on events up to Y-1. In-course: Spark `GBTClassifier` (pointwise). Out of course: LightGBM `LGBMRanker` (LambdaMART), which optimises the order inside each query.

# %%
from pyspark.ml.classification import GBTClassifier
TEST = (2020, 2021, 2022, 2023, 2024)
with timed('LGBMRanker walk-forward', RT): LG, models = E.walk_forward(PF, E.FEATS, TEST, keep_models=True)
allq = PF[PF['candidate'] & PF['q_in_pool']]; models['all'] = E.fit_ranker(allq, E.FEATS); pickle.dump(models, open('lake/models_replacement_only.pkl', 'wb'))
with timed('LGBMRanker trained on all signings, walk-forward', RT):
    LA, models_a = E.walk_forward(PA, E.FEATS, TEST, keep_models=True, use_candidates=False)                       # all-signings events ranked by a model trained on earlier all-signings events
    LV, models_v = E.walk_forward(PF, E.FEATS, TEST, keep_models=True, train_pairs=PA, use_candidates=False)       # replacement queries ranked by that same kind of model: far more training events
models_v['all'] = E.fit_ranker(PA, E.FEATS); pickle.dump(models_v, open('lake/models.pkl', 'wb')); models_rep, models = models, models_v
LA.to_parquet('lake/scores_all_signings.parquet', index=False); LV.to_parquet('lake/scores_lgbm_v2.parquet', index=False)
sdf = sp.createDataFrame(PF[['qid', 'player_id', 't', 'y', 'candidate', 'q_in_pool'] + E.FEATS].fillna(-1.0)).cache(); gb = []
with timed('Spark GBTClassifier walk-forward', RT):
    for Y in TEST:
        tr = sdf.filter((F.col('t') <= Y - 1) & F.col('candidate') & F.col('q_in_pool')); pr_ = tr.agg(F.mean('y')).first()[0]; tr = tr.withColumn('w', F.when(F.col('y') == 1, 0.5 / pr_).otherwise(0.5 / (1 - pr_)))
        gm = Pipeline(stages=[VectorAssembler(inputCols=E.FEATS, outputCol='features'), GBTClassifier(labelCol='y', weightCol='w', maxIter=60, maxDepth=3, stepSize=0.1, seed=SEED)]).fit(tr)
        gb.append(gm.transform(sdf.filter(F.col('t') == Y)).select('qid', 'player_id', 't', 'y', 'candidate', vector_to_array('probability')[1].alias('score')).toPandas())
gm.transform(sdf).select('probability').explain()
GB = pd.concat(gb); GB['score'] = np.where(GB['candidate'], GB['score'], -1e9)
def quick(R):
    R = R[R.groupby('qid')['y'].transform('max') == 1].copy(); R['r'] = R.groupby('qid')['score'].rank(ascending=False, method='average'); r = R[R['y'] == 1].groupby('qid')['r'].min()
    return {'queries': len(r), 'median_rank': float(r.median()), **{f'recall@{k}': float((r <= k).mean()) for k in (5, 10, 25)}}
LG.to_parquet('lake/scores_lgbm.parquet', index=False); GB.drop(columns='candidate').to_parquet('lake/scores_gbt.parquet', index=False)
imp = pd.Series(np.mean([m_.booster_.feature_importance('gain') for m_ in models['all'].ms], axis=0), index=E.FEATS); imp = (imp / imp.sum()).sort_values(ascending=False)
M['fusion'] = {'test_seasons': list(TEST), 'train_queries_first_fold': int(PF[(PF['t'] <= TEST[0] - 1) & PF['q_in_pool']]['qid'].nunique()), 'lgbm': quick(LG), 'lgbm_trained_on_all_signings': quick(LV), 'all_signings_task': quick(LA), 'gbt_spark': quick(GB), 'train_queries_first_fold_all_signings': int(PA[PA['t'] <= TEST[0] - 1]['qid'].nunique()), 'gain_share_all_events_model': imp.round(4).to_dict(), 'features': E.FEATS}
print(M['fusion'])
ex = EV[(EV['N_in_pool']) & (EV['t'] >= 2020)].iloc[0]; top = E.rank(lake, models, ex['C'], ex['t'], ex['S'], ex['P'], top=10)
print(f"example: {ex['club']} {ex['t']}-{str(ex['t'] + 1)[2:]}, replacing {ex['S_name']} ({ex['P']}); real signing {ex['N_name']}; model {top['model_used'][0]}"); display(top[['player', 'squad', 'age_num', 'value', 'sim_translated', 'traj_sim', 'hazard', 'score']].round(3))
_ = put_metrics('phase4', {**M, 'runtime_s': RT})
