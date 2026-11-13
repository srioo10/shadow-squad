# %% [markdown]
# # End-sem 01. Data lake (PySpark + Parquet) and player entity resolution
# Sources: FBref (mid-sem tables, exported unchanged by `src/run_baseline.py`), Understat team style, Transfermarkt (`davidcariboo/player-scores`).
# Everything is written as Parquet partitioned by `season` and `league`. FBref players are then linked to Transfermarkt ids.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from pyspark.sql import functions as F, Window as W
from pyspark.sql.types import DoubleType, StringType
from common import *
sp = spark(); RT = {}
TM = 'data/transfermarkt'
rd = lambda name: sp.read.csv(f'{TM}/{name}.csv', header=True, inferSchema=True, multiLine=True, escape='"')
lg = F.create_map(*[F.lit(x) for kv in TM_LEAGUE.items() for x in kv])
season_of = lambda d: F.when(F.month(d) >= 7, F.year(d)).otherwise(F.year(d) - 1)
def write(df, name, parts=('season', 'league')):
    df.write.mode('overwrite').partitionBy(*parts).parquet(f'lake/{name}'); n = sp.read.parquet(f'lake/{name}').count(); print(f'lake/{name}: {n:,} rows'); return n

# %% [markdown]
# ## 1. Write every source as Parquet

# %%
rows = {}
with timed('write lake', RT):
    fb = sp.read.parquet('lake/midsem/lake.parquet').withColumn('season', F.col('yr')).withColumn('league', F.col('comp'))
    rows['fbref'] = write(fb, 'fbref')
    raw = sp.read.parquet('lake/midsem/df_raw.parquet').withColumn('season', F.col('start_year').cast('int')).withColumn('league', F.col('comp'))
    rows['fbref_raw'] = write(raw, 'fbref_raw')
    rows['understat_team'] = write(sp.read.parquet('lake/midsem/ctx.parquet').withColumn('season', F.col('yr')).withColumn('league', F.col('comp')), 'understat_team')
    clubs = rd('clubs').select('club_id', F.col('name').alias('club_name'), F.coalesce(lg[F.col('domestic_competition_id')], F.lit('other')).alias('club_league'))
    tr = (rd('transfers').withColumn('transfer_date', F.to_date('transfer_date')).withColumn('season', (F.lit(2000) + F.substring('transfer_season', 1, 2).cast('int')))
          .filter(F.col('season').between(2010, 2026))
          .join(F.broadcast(clubs.select(F.col('club_id').alias('to_club_id'), F.col('club_league').alias('league'))), 'to_club_id', 'left').fillna({'league': 'other'}))
    rows['tm_transfers'] = write(tr, 'tm_transfers')
    val = (rd('player_valuations').withColumn('date', F.to_date('date')).withColumn('season', season_of('date')).filter(F.col('season') >= 2010)
           .withColumn('league', F.coalesce(lg[F.col('player_club_domestic_competition_id')], F.lit('other'))))
    rows['tm_valuations'] = write(val, 'tm_valuations')
    pl = (rd('players').withColumn('birth_year', F.year(F.to_date('date_of_birth'))).withColumn('season', F.col('last_season'))
          .withColumn('league', F.coalesce(lg[F.col('current_club_domestic_competition_id')], F.lit('other'))).drop('image_url', 'url', 'agent_name'))
    rows['tm_players'] = write(pl, 'tm_players')
    # player x club x season league minutes (needed for blocking and for "did the starter still play next season")
    app = (rd('appearances').filter(F.col('competition_id').isin(list(TM_LEAGUE))).withColumn('season', season_of(F.to_date('date')))
           .groupBy('player_id', F.col('player_club_id').alias('club_id'), 'season', lg[F.col('competition_id')].alias('league'))
           .agg(F.sum('minutes_played').alias('minutes'), F.count('*').alias('games')))
    rows['tm_club_minutes'] = write(app, 'tm_club_minutes')
    # final league tables from match results (covers 2024-25 and 2025-26, which the Understat table does not)
    g = rd('games').filter(F.col('competition_id').isin(list(TM_LEAGUE))).select('season', lg[F.col('competition_id')].alias('league'), 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals')
    side = lambda me, op, gf, ga: g.select('season', 'league', F.col(me).alias('club_id'), F.col(gf).alias('gf'), F.col(ga).alias('ga'))
    tab = (side('home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals').unionByName(side('away_club_id', 'home_club_id', 'away_club_goals', 'home_club_goals'))
           .groupBy('season', 'league', 'club_id').agg(F.count('*').alias('matches'), F.sum(F.when(F.col('gf') > F.col('ga'), 3).when(F.col('gf') == F.col('ga'), 1).otherwise(0)).alias('pts'), F.sum(F.col('gf') - F.col('ga')).alias('gd'))
           .withColumn('tm_pos', F.row_number().over(W.partitionBy('season', 'league').orderBy(F.desc(F.col('pts') / F.col('matches')), F.desc('gd')))))
    rows['tm_tables'] = write(tab, 'tm_tables')
sp.read.parquet('lake/fbref').filter("season = 2023 and league = 'Ligue 1'").select('player', 'squad', 'min_num').explain()

# %% [markdown]
# ## 2. Entity resolution: FBref player-season rows to Transfermarkt player ids
# **Pass 1 (learn the club alias map).** Join on (season, league, birth year, exact name key) and take, for every FBref club, the Transfermarkt club most of its exactly-matched players belong to.
# **Pass 2 (the match).** Block on (season, club). Inside a block compare name keys with Spark `levenshtein` and a Jaro-Winkler pandas UDF, add birth year and position-group agreement, keep mutual best pairs.

# %%
from rapidfuzz.distance import JaroWinkler
pkey_u = F.udf(pkey, StringType())
@F.pandas_udf(DoubleType())
def jw(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series([JaroWinkler.similarity(x, y) for x, y in zip(a, b)])
@F.pandas_udf(DoubleType())
def subset(a: pd.Series, b: pd.Series) -> pd.Series:   # "Rodri" vs "Rodrigo Hernandez Rodri": every token of the shorter name appears in the longer one
    return pd.Series([float(bool(x) and bool(y) and (set(x.split()) <= set(y.split()) or set(y.split()) <= set(x.split()))) for x, y in zip(a, b)])

with timed('entity resolution', RT):
    raw = sp.read.parquet('lake/fbref_raw').filter(~F.upper('pos').startswith('GK'))
    fbp = (raw.select('player', 'squad', F.col('league').alias('comp'), F.col('season').alias('yr'), F.col('born').cast('int').alias('born'), F.upper(F.substring('pos', 1, 2)).alias('pos2'), 'min_num')
           .withColumn('k_fb', pkey_u('player')).dropDuplicates(['player', 'squad', 'comp', 'yr']).cache())
    tmp = (sp.read.parquet('lake/tm_club_minutes').join(sp.read.parquet('lake/tm_players').select('player_id', 'name', 'birth_year', 'sub_position', 'position'), 'player_id')
           .filter(F.col('position') != 'Goalkeeper').withColumnRenamed('season', 'yr').withColumnRenamed('league', 'comp').withColumn('k_tm', pkey_u('name')).cache())
    # pass 1: club alias map by majority vote of exact matches
    votes = (fbp.join(tmp, (fbp.yr == tmp.yr) & (fbp.comp == tmp.comp) & (fbp.born == tmp.birth_year) & (fbp.k_fb == tmp.k_tm)).groupBy(fbp.comp, 'squad', 'club_id').count()
             .withColumn('r', F.row_number().over(W.partitionBy('comp', 'squad').orderBy(F.desc('count')))).withColumn('share', F.col('count') / F.sum('count').over(W.partitionBy('comp', 'squad'))).filter('r = 1').drop('r'))
    club_x = votes.join(F.broadcast(clubs.select('club_id', 'club_name')), 'club_id').toPandas()
    all_sq = fbp.select('comp', 'squad').distinct().toPandas()
    assert len(club_x) == len(all_sq) and club_x['club_id'].is_unique, 'club map is not one-to-one / complete'
    sp.createDataFrame(club_x).write.mode('overwrite').parquet('lake/club_xref')
    # pass 2: block on (season, club)
    cx = F.broadcast(sp.createDataFrame(club_x[['comp', 'squad', 'club_id']]))
    cand = (fbp.join(cx, ['comp', 'squad']).join(tmp.drop('comp'), ['yr', 'club_id'])
            .withColumn('lev', 1 - F.levenshtein('k_fb', 'k_tm') / F.greatest(F.length('k_fb'), F.length('k_tm'), F.lit(1)))
            .withColumn('name_sim', F.greatest('lev', jw('k_fb', 'k_tm'), F.lit(0.9) * subset('k_fb', 'k_tm')))
            .withColumn('by', F.when(F.col('born') == F.col('birth_year'), 1.0).when(F.abs(F.col('born') - F.col('birth_year')) == 1, 0.5).otherwise(0.0))
            .withColumn('pos_ok', (((F.col('pos2') == 'DF') & (F.col('position') == 'Defender')) | ((F.col('pos2') == 'MF') & (F.col('position') == 'Midfield')) | ((F.col('pos2') == 'FW') & (F.col('position') == 'Attack'))).cast('double'))
            .withColumn('confidence', F.round(0.6 * F.col('name_sim') + 0.3 * F.col('by') + 0.1 * F.col('pos_ok'), 3))
            .filter((F.col('name_sim') >= 0.80) & (F.col('by') > 0) | (F.col('name_sim') >= 0.95)))
    best = (cand.withColumn('r1', F.row_number().over(W.partitionBy('player', 'squad', 'yr').orderBy(F.desc('confidence'), F.desc('minutes'))))
                .withColumn('r2', F.row_number().over(W.partitionBy('player_id', 'club_id', 'yr').orderBy(F.desc('confidence'), F.desc('min_num')))).filter('r1 = 1 and r2 = 1'))
    fine = F.create_map(*[F.lit(x) for kv in FINE_POS.items() for x in kv])
    xref = best.select('player', 'squad', 'comp', 'yr', 'born', 'player_id', F.col('name').alias('tm_name'), 'club_id', 'birth_year', 'sub_position', fine[F.col('sub_position')].alias('fine_pos'),
                       'name_sim', 'by', 'pos_ok', 'confidence', F.col('minutes').alias('tm_minutes'), 'min_num')
    xref.withColumn('season', F.col('yr')).withColumn('league', F.col('comp')).write.mode('overwrite').partitionBy('season', 'league').parquet('lake/player_xref')
cand.select('player', 'tm_name' if False else 'name', 'confidence').explain()
X = sp.read.parquet('lake/player_xref').toPandas(); FB = fbp.toPandas()
print(f'{len(club_x)} FBref clubs mapped one-to-one to Transfermarkt clubs (lowest vote share {club_x["share"].min():.2f}); {len(X):,} of {len(FB):,} outfield player-club-seasons linked')

# %% [markdown]
# ## 3. Checks: match rate per league-season, a random sample to eyeball, and the unmatched feeders

# %%
FBm = FB.merge(X[['player', 'squad', 'yr', 'player_id', 'confidence', 'fine_pos']], on=['player', 'squad', 'yr'], how='left'); FBm['matched'] = FBm['player_id'].notna()
rate_all = FBm.pivot_table(index='comp', columns='yr', values='matched', aggfunc='mean').round(3)
rate_900 = FBm[FBm['min_num'] >= 900].pivot_table(index='comp', columns='yr', values='matched', aggfunc='mean').round(3)
print('match rate, all outfield rows'); display(rate_all); print('match rate, rows with >= 900 minutes (the modelling population)'); display(rate_900)
print('30 random matches for a manual check:')
display(X.sample(30, random_state=SEED)[['player', 'tm_name', 'squad', 'yr', 'born', 'birth_year', 'sub_position', 'name_sim', 'confidence']])
print('20 lowest-confidence matches:'); display(X.nsmallest(20, 'confidence')[['player', 'tm_name', 'squad', 'yr', 'born', 'birth_year', 'sub_position', 'name_sim', 'confidence']])
L = pd.read_parquet('lake/midsem/lake.parquet', columns=['player', 'squad', 'comp', 'yr', 'feeder', 'age_num', 'min_num'])
Lf = L[L['feeder']].merge(X[['player', 'squad', 'yr', 'player_id']], on=['player', 'squad', 'yr'], how='left')
unm = Lf[Lf['player_id'].isna()].sort_values(['yr', 'comp', 'squad']); unm.to_csv('outputs/unmatched_feeders.csv', index=False)
print(f'unmatched feeders: {len(unm)} of {len(Lf)} feeder-seasons (full list in outputs/unmatched_feeders.csv)'); display(unm.head(40))
same_id = X.groupby('player_id')['player'].nunique(); same_name = X.groupby('player')['player_id'].nunique()
print(f'Transfermarkt ids carrying more than one FBref spelling: {(same_id > 1).sum()}; FBref names that are really several players: {(same_name > 1).sum()}')

# %%
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 2.6)); im = ax.imshow(rate_900.values, vmin=0.9, vmax=1, cmap='Blues'); ax.set_xticks(range(rate_900.shape[1]), rate_900.columns); ax.set_yticks(range(len(rate_900)), rate_900.index)
for i in range(rate_900.shape[0]):
    for j in range(rate_900.shape[1]): ax.text(j, i, f'{rate_900.values[i, j]:.0%}', ha='center', va='center', fontsize=8, color='white' if rate_900.values[i, j] > .96 else 'black')
ax.set_title('FBref to Transfermarkt match rate (players with >= 900 min)'); fig.colorbar(im, shrink=.8); os.makedirs('outputs/figs', exist_ok=True); fig.savefig('outputs/figs/p1_match_rate.png', dpi=160, bbox_inches='tight'); plt.show()
put_metrics('phase1', {'lake_rows': rows, 'runtime_s': RT, 'clubs_mapped': len(club_x), 'club_vote_share_min': float(club_x['share'].min()), 'rows_linked': len(X), 'rows_total': len(FB),
                       'match_rate_all': float(FBm['matched'].mean()), 'match_rate_900': float(FBm.loc[FBm['min_num'] >= 900, 'matched'].mean()), 'match_rate_900_min_cell': float(np.nanmin(rate_900.values)),
                       'feeders_total': len(Lf), 'feeders_unmatched': len(unm), 'mean_confidence': float(X['confidence'].mean()), 'ids_multi_spelling': int((same_id > 1).sum()), 'names_multi_id': int((same_name > 1).sum())})
