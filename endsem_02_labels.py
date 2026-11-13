# %% [markdown]
# # End-sem 02. Labels v2 and replacement events
# 1. Outcome from real transfers (Transfermarkt) instead of "which club row does he have two seasons later".
# 2. Person-period (survival) rows. 3. Replacement events: an elite club lost a starter and signed someone for that position.

# %%
import sys, os; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from pyspark.sql import functions as F, Window as W
from common import *
sp = spark(); RT = {}
POOL_MAX_AGE, POOL_MIN_MIN, STARTER_MIN, GONE_MAX_MIN, SIGNING_MIN_MIN = 26, 900, MID['ELITE_MIN_MIN'], 450, 450
DATA_END = '2026-06-01'      # last date with complete transfer records in this Transfermarkt dump
for n in ('fbref', 'player_xref', 'tm_transfers', 'tm_club_minutes', 'tm_tables', 'tm_players', 'club_xref'): sp.read.parquet(f'lake/{n}').createOrReplaceTempView(n)
sp.read.parquet('lake/midsem/league_table.parquet').createOrReplaceTempView('mid_table')
sp.createDataFrame(pd.DataFrame({'league': list(MID['LEAGUE_RANK']), 'lrank': list(MID['LEAGUE_RANK'].values())})).createOrReplaceTempView('lrank')
sp.createDataFrame(pd.DataFrame({'squad': sorted({c for v in MID['GIANTS'].values() for c in v})})).createOrReplaceTempView('giants')

# %% [markdown]
# ## 1. Club tier per season and clean transfers
# Mid-sem table positions are used where they exist (2017-2023); positions rebuilt from Transfermarkt match results cover 2024 and 2025. Loans are found by walking each player's moves in date order (grouped pandas UDF): a move that undoes the open previous move within 18 months closes a loan and both legs are removed. A loan that became permanent counts as a move from the day the loan started.

# %%
with timed('club tiers + clean transfers', RT):
    cs = sp.sql("""
      SELECT t.club_id, t.season, t.league, x.squad, coalesce(m.position, t.tm_pos) AS pos, m.position AS mid_pos, t.tm_pos, (g.squad IS NOT NULL) AS giant, r.lrank
      FROM tm_tables t LEFT JOIN club_xref x ON x.club_id = t.club_id LEFT JOIN mid_table m ON m.squad = x.squad AND m.comp = t.league AND m.yr = t.season
      LEFT JOIN giants g ON g.squad = x.squad JOIN lrank r ON r.league = t.league WHERE t.season >= 2016""").withColumn('elite_club', F.col('giant') | (F.col('pos') <= MID['ELITE_TOP_N'])).cache()
    cs.createOrReplaceTempView('club_season')
    t = (sp.table('tm_transfers').filter((F.col('from_club_id') != F.col('to_club_id')) & (F.col('transfer_date') <= F.lit(DATA_END)))
         .select('player_id', F.col('transfer_date').alias('d'), F.col('from_club_id').cast('int').alias('f'), F.col('to_club_id').cast('int').alias('c'), F.col('from_club_name').alias('fn'), F.col('to_club_name').alias('tn'),
                 F.col('transfer_fee').cast('double').alias('fee'), F.col('market_value_in_eur').cast('double').alias('mv')).dropDuplicates(['player_id', 'd', 'f', 'c']))
    def mark_loans(g):   # walk one player's moves in date order; a move that undoes the open previous move within 18 months closes a loan
        g = g.sort_values('d').reset_index(drop=True); out = np.zeros(len(g), bool); ret = np.zeros(len(g), bool); fee = g['fee'].to_numpy(copy=True); open_i = None; i = 0
        days = lambda x, y: (g.d[x] - g.d[y]).days
        while i < len(g):
            if open_i is not None and g.f[i] == g.c[open_i] and g.c[i] == g.f[open_i] and days(i, open_i) <= 550:
                if days(i, open_i) <= 7: open_i = i                    # bought and loaned straight back: the purchase stands, this leg is the loan
                elif i + 1 < len(g) and g.f[i + 1] == g.f[open_i] and g.c[i + 1] == g.c[open_i] and days(i + 1, i) <= 62:
                    ret[i] = ret[i + 1] = True; fee[open_i] = fee[i + 1]; open_i = None; i += 1   # loan made permanent: the move counts from the loan start, with the later fee
                else: out[open_i] = ret[i] = True; open_i = None
            else: open_i = i
            i += 1
        return g.assign(loan_out=out, loan_return=ret, fee=fee)[['player_id', 'd', 'f', 'c', 'fn', 'tn', 'fee', 'mv', 'loan_out', 'loan_return']]
    marked = t.groupBy('player_id').applyInPandas(mark_loans, 'player_id int, d date, f int, c int, fn string, tn string, fee double, mv double, loan_out boolean, loan_return boolean')
    youth = F.col('fn').rlike(r'(U\d\d| B| II| C|Yth\.?|Youth|Primavera|Castilla|Atlètic|Reserves|Res\.)$')
    tc = (marked.withColumn('from_youth', youth).withColumn('season', F.when(F.month('d') >= 6, F.year('d')).otherwise(F.year('d') - 1)).cache())
    tc.createOrReplaceTempView('tc'); tc.write.mode('overwrite').parquet('lake/tm_transfers_clean')
ov = cs.filter('mid_pos IS NOT NULL').toPandas()
tab_agree = float((ov['mid_pos'] == ov['tm_pos']).mean()); top6_agree = float(((ov['mid_pos'] <= 6) == (ov['tm_pos'] <= 6)).mean())
n_t = tc.count(); n_loan = tc.filter('loan_out').count(); n_ret = tc.filter('loan_return').count()
print(f'{n_t:,} transfers; {n_loan:,} loan departures and {n_ret:,} loan returns removed. Table rebuilt from results agrees with the mid-sem table on {tab_agree:.1%} of club-seasons ({top6_agree:.1%} on top-6 membership).')

# %% [markdown]
# ## 2. Transfer-based outcome for every feeder-season, compared with the mid-sem label

# %%
with timed('transfer outcome', RT):
    fs = sp.sql("""SELECT f.player, f.squad, f.comp, f.yr, f.age_num, f.min_num, f.pos_group, f.feeder, f.elite_club, f.labeled, f.best_tier, f.composite_perf_index, x.player_id, x.club_id, x.fine_pos, r.lrank AS lrank0
                   FROM fbref f LEFT JOIN player_xref x ON x.player = f.player AND x.squad = f.squad AND x.yr = f.yr JOIN lrank r ON r.league = f.comp""")
    fs.createOrReplaceTempView('fs')
    mv = sp.sql(f"""
      SELECT s.player_id, s.yr, t.d, t.c, t.tn AS dest_name, t.fee, months_between(t.d, make_date(s.yr + 1, 6, 1)) AS months, k.league AS dest_league,
             CASE WHEN k.club_id IS NULL THEN 0 WHEN k.giant OR k.pos <= {MID['CL_TOP_N']} THEN 3 WHEN k.pos <= {MID['ELITE_TOP_N']} THEN 2 WHEN k.lrank < s.lrank0 THEN 1 ELSE 0 END AS tier
      FROM fs s JOIN tc t ON t.player_id = s.player_id AND t.d >= make_date(s.yr + 1, 6, 1) AND t.d < make_date(s.yr + 3, 6, 1) AND NOT t.loan_out AND NOT t.loan_return
      LEFT JOIN club_season k ON k.club_id = t.c AND k.season = t.season""")
    first = mv.withColumn('r', F.row_number().over(W.partitionBy('player_id', 'yr').orderBy('d'))).filter('r = 1').drop('r')
    first_up = (mv.filter('tier >= 1').withColumn('r', F.row_number().over(W.partitionBy('player_id', 'yr').orderBy('d'))).filter('r = 1')
                .select('player_id', 'yr', F.col('tier').alias('up_tier'), F.col('months').alias('up_months'), F.col('dest_name').alias('up_dest'), F.col('fee').alias('up_fee')))
    lab = (fs.join(first.select('player_id', 'yr', F.col('tier').alias('tier_tm'), F.col('dest_name').alias('first_dest'), F.col('fee').alias('first_fee'), F.col('months').alias('first_months')), ['player_id', 'yr'], 'left')
             .join(first_up, ['player_id', 'yr'], 'left')
             .withColumn('moved', F.col('tier_tm').isNotNull()).fillna({'tier_tm': 0, 'up_tier': 0})
             .withColumn('observed24', F.expr(f"make_date(yr + 3, 6, 1) <= DATE'{DATA_END}'")))
    lab.write.mode('overwrite').parquet('lake/labels_v2')
mv.explain()
LB = pd.read_parquet('lake/labels_v2'); FE = LB[LB['feeder'] & LB['player_id'].notna()].copy()
has_tr = set(tc.select('player_id').distinct().toPandas()['player_id']); FE['has_transfer_record'] = FE['player_id'].isin(has_tr)
print(f"feeder-seasons whose player has at least one Transfermarkt transfer record: {FE['has_transfer_record'].mean():.1%} ({(~FE['has_transfer_record']).sum()} without any; their label is 'no move')")
cmp_ = FE[FE['labeled']]
conf = pd.crosstab(cmp_['best_tier'].rename('mid-sem tier (club row in t+1..t+2)'), cmp_['tier_tm'].rename('transfer tier (first permanent move, 24 months)')); display(conf)
conf_up = pd.crosstab((cmp_['best_tier'] >= 1).rename('mid-sem moved up'), (cmp_['up_tier'] >= 1).rename('transfer moved up (any permanent move in 24 months)')); display(conf_up)
only_old = cmp_[(cmp_['best_tier'] >= 1) & (cmp_['up_tier'] == 0)]; only_new = cmp_[(cmp_['best_tier'] == 0) & (cmp_['up_tier'] >= 1)]
print(f"labelled feeders compared: {len(cmp_)}; agreement on moved-up yes/no: {((cmp_['best_tier'] >= 1) == (cmp_['up_tier'] >= 1)).mean():.1%}; mid-sem only: {len(only_old)}, transfer only: {len(only_new)}")
print('mid-sem "moves" with no permanent transfer (loan returns, loans, same-name players):'); display(only_old.sample(min(12, len(only_old)), random_state=SEED)[['player', 'squad', 'yr', 'best_tier', 'first_dest', 'has_transfer_record']])
print(f"of the {len(only_old)} mid-sem-only moves, {(~only_old['has_transfer_record']).sum()} belong to players with no transfer record at all (a gap in the Transfermarkt dump, not a real disagreement)")
rate_v2 = FE.groupby('yr').agg(feeders=('player', 'size'), moved_up=('up_tier', lambda s: (s >= 1).mean()), elite=('up_tier', lambda s: (s >= 2).mean()), observed=('observed24', 'all')).round(3); display(rate_v2)

# %% [markdown]
# ## 3. Survival format: one row per feeder per season at risk
# `event` = a permanent move up in the 12 months after the season. `dur` counts his seasons as a feeder so far. A row is censored when its 12-month window is not fully observed; a player who never moves is right-censored at his last feeder season. Seasons 2022-23 and 2023-24 are usable here, which the mid-sem label could not do.

# %%
PP = FE.sort_values(['player_id', 'yr']).copy()
PP['event'] = ((PP['up_tier'] >= 1) & (PP['up_months'] < 12)).astype(int)
PP['observed12'] = pd.to_datetime((PP['yr'] + 2).astype(str) + '-06-01') <= pd.Timestamp(DATA_END)
PP['dur'] = PP.groupby('player_id').cumcount() + 1
PP['prior_event'] = PP.groupby('player_id')['event'].cumsum() - PP['event']
PP = PP[(PP['prior_event'] == 0) & PP['observed12']].drop(columns='prior_event')      # leave the risk set after the first event
PP.to_parquet('lake/person_period.parquet', index=False)
haz = PP.groupby('dur').agg(at_risk=('event', 'size'), events=('event', 'sum')); haz['hazard'] = (haz['events'] / haz['at_risk']).round(3); display(haz)
print(f"person-period rows: {len(PP)} for {PP['player_id'].nunique()} players; events: {int(PP['event'].sum())}; rows from 2022 and 2023 (unusable under the mid-sem label): {int((PP['yr'] >= 2022).sum())}")

# %% [markdown]
# ## 4. Replacement events
# Elite club C, season t: starter S (>= 1,350 min) at fine position P plays < 450 league minutes for C in t+1, and C signs N (same fine position, >= 450 minutes for C in t+1) in the summer window between the two seasons.

# %%
with timed('replacement events', RT):
    sp.sql(f"""SELECT s.player_id, s.player, s.squad, s.comp, s.yr, s.age_num, s.min_num, s.fine_pos, s.club_id FROM fs s
               WHERE NOT s.elite_club AND s.age_num <= {POOL_MAX_AGE} AND s.min_num >= {POOL_MIN_MIN} AND s.player_id IS NOT NULL AND s.fine_pos IS NOT NULL""").createOrReplaceTempView('pool')
    sp.table('pool').write.mode('overwrite').parquet('lake/pool')
    ev = sp.sql(f"""
      WITH gone AS (
        SELECT s.club_id AS C, s.squad AS club, s.comp AS league, s.yr AS t, s.player_id AS S, s.player AS S_name, s.fine_pos AS P, s.min_num AS S_min, coalesce(m.minutes, 0) AS S_min_next
        FROM fs s JOIN club_season k ON k.club_id = s.club_id AND k.season = s.yr + 1
        LEFT JOIN tm_club_minutes m ON m.player_id = s.player_id AND m.club_id = s.club_id AND m.season = s.yr + 1
        WHERE s.elite_club AND s.min_num >= {STARTER_MIN} AND s.fine_pos IS NOT NULL AND coalesce(m.minutes, 0) < {GONE_MAX_MIN}),
      sign AS (
        SELECT t.c AS C, year(t.d) - 1 AS t, t.player_id AS N, p.name AS N_name, t.fn AS N_from, t.f AS N_from_id, t.fee, t.mv AS N_value, t.d AS N_date, t.loan_out AS N_is_loan, m.minutes AS N_min_next,
               CASE p.sub_position WHEN 'Centre-Back' THEN 'CB' WHEN 'Left-Back' THEN 'FB' WHEN 'Right-Back' THEN 'FB' WHEN 'Defensive Midfield' THEN 'DM' WHEN 'Central Midfield' THEN 'CM' WHEN 'Attacking Midfield' THEN 'AM'
                    WHEN 'Centre-Forward' THEN 'ST' WHEN 'Second Striker' THEN 'ST' WHEN 'Goalkeeper' THEN NULL ELSE 'W' END AS P
        FROM tc t JOIN tm_players p ON p.player_id = t.player_id JOIN tm_club_minutes m ON m.player_id = t.player_id AND m.club_id = t.c AND m.season = year(t.d)
        WHERE month(t.d) BETWEEN 6 AND 10 AND NOT (month(t.d) = 10 AND day(t.d) > 10) AND NOT t.loan_return AND NOT t.from_youth AND m.minutes >= {SIGNING_MIN_MIN})
      SELECT g.*, s.N, s.N_name, s.N_from, s.N_from_id, s.fee, s.N_value, s.N_date, s.N_is_loan, s.N_min_next, (q.player_id IS NOT NULL) AS N_in_pool, q.squad AS N_pool_club, q.comp AS N_pool_league, q.age_num AS N_age,
             fk.league AS N_from_league
      FROM gone g JOIN sign s ON s.C = g.C AND s.t = g.t AND s.P = g.P
      LEFT JOIN pool q ON q.player_id = s.N AND q.yr = g.t
      LEFT JOIN club_season fk ON fk.club_id = s.N_from_id AND fk.season = g.t""")
    ev.write.mode('overwrite').parquet('lake/replacement_events')
ev.explain()
EV = pd.read_parquet('lake/replacement_events'); EV['query'] = EV['C'].astype(str) + '_' + EV['t'].astype(str) + '_' + EV['S'].astype(str)
sg = EV.drop_duplicates(['C', 't', 'N'])
print(f"events (C, t, S, N): {len(EV)}; distinct queries (C, t, S): {EV['query'].nunique()}; distinct signings: {len(sg)}")
print(f"signings inside the candidate pool: {sg['N_in_pool'].mean():.1%} ({int(sg['N_in_pool'].sum())} of {len(sg)}); from a five-league club at all: {sg['N_from_league'].notna().mean():.1%}")
cov = sg.groupby('t').agg(signings=('N', 'size'), in_pool=('N_in_pool', 'mean'), from_top5=('N_from_league', lambda s: s.notna().mean())).round(3); display(cov)
why = sg[~sg['N_in_pool']].assign(reason=lambda d: np.where(d['N_from_league'].isna(), 'club outside the five leagues', 'five-league club, but elite / too old / under 900 min / unmatched'))['reason'].value_counts(); display(why)
print('20 events for a manual check against known transfers:')
pd.set_option('display.width', 250)
display(EV.sample(20, random_state=SEED).sort_values(['t', 'club'])[['club', 't', 'P', 'S_name', 'S_min', 'S_min_next', 'N_name', 'N_from', 'fee', 'N_in_pool']])
inq = EV[EV['N_in_pool']]
print(f"in-pool events: {len(inq)}; in-pool queries: {inq['query'].nunique()}; by season: {inq.groupby('t')['query'].nunique().to_dict()}; by position: {inq.groupby('P')['query'].nunique().to_dict()}")

# %%
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
c = conf.reindex(index=range(4), columns=range(4), fill_value=0); im = ax[0].imshow(np.log1p(c.values), cmap='Blues')
for i in range(4):
    for j in range(4): ax[0].text(j, i, int(c.values[i, j]), ha='center', va='center', fontsize=9)
ax[0].set_xlabel('transfer-based tier'); ax[0].set_ylabel('mid-sem tier'); ax[0].set_xticks(range(4)); ax[0].set_yticks(range(4)); ax[0].set_title('Label v1 vs v2 (labelled feeders)')
ax[1].bar(cov.index - .2, cov['from_top5'], .4, label='from a five-league club'); ax[1].bar(cov.index + .2, cov['in_pool'], .4, label='inside candidate pool'); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8); ax[1].set_title('Coverage of real replacement signings'); ax[1].set_xlabel('season t')
fig.tight_layout(); fig.savefig('outputs/figs/p2_labels_events.png', dpi=160); 
_ = put_metrics('phase2', {'runtime_s': RT, 'transfers_clean': n_t, 'loan_out_removed': n_loan, 'loan_return_removed': n_ret, 'table_agreement': tab_agree, 'top6_agreement': top6_agree,
    'feeders_with_transfer_record': float(FE['has_transfer_record'].mean()), 'label_midsem_only_no_record': int((~only_old['has_transfer_record']).sum()), 'label_compare_n': len(cmp_), 'label_agreement_up': float(((cmp_['best_tier'] >= 1) == (cmp_['up_tier'] >= 1)).mean()), 'label_midsem_only': len(only_old), 'label_transfer_only': len(only_new),
    'confusion_tier': conf.to_dict(), 'moved_up_rate_v1': float((cmp_['best_tier'] >= 1).mean()), 'moved_up_rate_v2_same_rows': float((cmp_['up_tier'] >= 1).mean()), 'moved_up_rate_v2_all_seasons': float((FE['up_tier'] >= 1).mean()),
    'rate_v2_by_season': rate_v2.reset_index().to_dict('records'), 'person_period_rows': len(PP), 'person_period_players': int(PP['player_id'].nunique()), 'person_period_events': int(PP['event'].sum()),
    'person_period_rows_2022_23': int((PP['yr'] >= 2022).sum()), 'hazard_by_duration': haz.reset_index().to_dict('records'),
    'events': len(EV), 'queries': int(EV['query'].nunique()), 'signings': len(sg), 'signings_in_pool': int(sg['N_in_pool'].sum()), 'in_pool_share': float(sg['N_in_pool'].mean()),
    'from_top5_share': float(sg['N_from_league'].notna().mean()), 'out_of_pool_reasons': why.to_dict(), 'in_pool_events': len(inq), 'in_pool_queries': int(inq['query'].nunique()),
    'in_pool_queries_by_season': {int(k): int(v) for k, v in inq.groupby('t')['query'].nunique().items()}, 'in_pool_queries_by_pos': inq.groupby('P')['query'].nunique().to_dict(),
    'params': {'POOL_MAX_AGE': POOL_MAX_AGE, 'POOL_MIN_MIN': POOL_MIN_MIN, 'STARTER_MIN': STARTER_MIN, 'GONE_MAX_MIN': GONE_MAX_MIN, 'SIGNING_MIN_MIN': SIGNING_MIN_MIN, 'DATA_END': DATA_END}})
