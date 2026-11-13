"""Phase 7: write report/endsem.tex from outputs/metrics.json (no number is typed by hand), compile it, and put it after the unchanged mid-sem PDF."""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
m = json.load(open(f'{ROOT}/outputs/metrics.json')); b = json.load(open(f'{ROOT}/outputs/baseline_metrics.json'))
p1, p2, p3, p4, p5, p6 = (m[f'phase{i}'] for i in range(1, 7))
pc = lambda x, d=1: f'{100 * x:.{d}f}\\%'
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; n = lambda x: f'{int(x):,}'
ci = lambda c, d=3: f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'
def tex(s): return str(s).replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('->', '$\\rightarrow$')
fig = lambda name, cap, w=1.0: f'\\begin{{figure}}[H]\\centering\\includegraphics[width={w}\\linewidth]{{{ROOT}/outputs/{name}}}\\caption{{{cap}}}\\end{{figure}}\n'
T = {r['method']: r for r in p5['main']['table']}; SS = 'Shadow Squad (LambdaMART, all scores)'; BY = 'Best young performer (composite index)'; MV = 'Highest market value'; RN = 'Random order of the pool'; HZ = 'Single score: hazard'
lsh, ps, bk, gr, als, cf, ol, st, tr, fu, sv, cn = p3['lsh'], p3['prefixspan'], p3['bisecting'], p3['graph'], p3['als'], p3['classification'], p3['outlier'], p3['stream'], p4['translation'], p4['fusion'], p5['survival'], p5['conformal']
cs = {(r['label'], r['model']): r for r in cf['summary']}; mid = cf['midsem_leaderboard']; rt = p3['runtime_s']
mm = tr['mean_scaled_mae']; nq = p5['main']['test_queries']

def master_rows():
    order = [RN, MV, 'Raw cosine similarity to the starter', BY, HZ, 'Single score: translated similarity', 'Shadow Squad fusion with Spark GBT (pointwise)', SS] + [k for k in T if k.startswith('Ablation')]
    out = []
    for k in order:
        r = T[k]; lo, hi = r.get('recall@10_lo'), r.get('recall@10_hi'); pv = r.get('p_vs_best_young@10')
        name = ('\\textbf{' + tex(k) + '}') if k == SS else tex(k)
        out.append(f"{name} & {r['median_rank']:.1f} & {pc(r['recall@5'], 0)} & {pc(r['recall@10'], 0)} [{100 * lo:.0f}, {100 * hi:.0f}] & {pc(r['recall@25'], 0)} & {pc(r['recall@50'], 0)} & {'' if pv is None or pv != pv else f2(pv)} \\\\")
        if k in (BY, 'Single score: translated similarity'): out.append('\\midrule')
    return '\n'.join(out)

def case_text(c):
    pcts = c['signed_feature_percentile_in_pool']; nice = {'log_value': 'market value', 'value_gap': 'value gap to the starter', 'sim_translated': 'translated similarity', 'traj_sim': 'trajectory similarity', 'pipeline_rank': 'club pipeline rank', 'club_taste': 'club taste', 'hazard': 'move-up hazard', 'age_num': 'age', 'composite_perf_index': 'composite index'}
    srt = sorted(pcts.items(), key=lambda kv: kv[1]); pick = srt[:3] if c['kind'] == 'miss' else srt[-3:][::-1]
    desc = ', '.join(f"{nice[k]} (higher than {100 * v:.0f}\\% of the pool)" for k, v in pick)
    fee = 'fee not recorded' if c['fee'] is None else ('free or undisclosed' if c['fee'] == 0 else f"fee EUR {c['fee'] / 1e6:.1f}m")
    head = f"\\paragraph{{{tex(c['club'])} {c['season']}-{str(c['season'] + 1)[2:]}: {tex(c['starter'])} ({c['position']}) replaced by {tex(c['signed'])} from {tex(c['from'])} ({fee}).}} "
    if c['kind'] == 'hit': body = f"The engine ranked him {c['rank']} of {c['pool']} pool players at that position. His strongest scores were {desc}. The first name on the list was {tex(c['top5'][0]['player'])} ({tex(c['top5'][0]['squad'])})."
    else: body = f"The engine ranked him {c['rank']} of {c['pool']}, close to last. His weakest scores were {desc}. His market value was EUR {c['signed_value'] / 1e6:.1f}m against EUR {c['starter_value'] / 1e6:.1f}m for the starter he replaced. The ranker has learned that elite clubs mostly buy established, highly valued players, so a cheap squad signing with a thin record is exactly the kind of move it cannot see coming."
    return head + body + '\n' + fig(f"screens/{c['tag']}.png", f"Dashboard for this event: ranked list, translated radar and SHAP waterfall for {tex(c['signed'])}.", 0.92)

cases = p6['cases']; hits = [c for c in cases if c['kind'] == 'hit']; miss = [c for c in cases if c['kind'] == 'miss']
cm = {r['target']: r for r in cn['movers']}; cr = {r['target']: r for r in cn['replacement']}
top_pat = ps['top_multi_season'][0] if ps['top_multi_season'] else ps['top'][0]
doc = rf'''\documentclass[11pt,a4paper]{{article}}
\usepackage{{fontspec}}\usepackage[margin=2.2cm]{{geometry}}\usepackage{{graphicx,booktabs,float,tikz,hyperref,parskip,caption}}
\usetikzlibrary{{arrows.meta,positioning}}\captionsetup{{font=small}}\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\title{{Shadow Squad: who replaces whom?\\[4pt]\large End-semester report, Player Replacement Scouting\\Analysis and Systems of Big Data, IIITDM Kancheepuram}}\date{{}}
\begin{{document}}\sloppy\maketitle
\noindent This part follows the mid-semester report, which is included unchanged before it. Every number below is read from \texttt{{outputs/metrics.json}}, which the notebooks \texttt{{endsem\_01}} to \texttt{{endsem\_06}} write when they run.

\section{{What changed since the mid-semester version, and why}}
The mid-sem project asked which young players at smaller clubs will move up. Its main result was that no method clearly beat ranking the best young performers, and its cosine shortlist was worse than that baseline. Three things were limiting it, and the end-sem version changes all three.

\textbf{{The question.}} A club does not buy "a player who will move up". It buys a replacement for a specific starter who left. The end-sem system takes an elite club, a season and a departing starter, and returns a ranked list of candidates. The test is whether the player the club really signed is near the top of that list.

\textbf{{The labels.}} Mid-sem outcomes were inferred from the club a player's name appeared next to one or two seasons later. They are now taken from Transfermarkt transfer records. On the {n(p2['label_compare_n'])} feeder-seasons that have both labels, the two agree on "moved up" for {pc(p2['label_agreement_up'])}. The old rule called {n(p2['label_midsem_only'])} moves that have no matching permanent transfer; most are loan spells and loan returns, and {n(p2['label_midsem_only_no_record'])} belong to players with no transfer record in the Transfermarkt dump. On the same rows the move-up rate falls from {pc(p2['moved_up_rate_v1'])} to {pc(p2['moved_up_rate_v2_same_rows'])}. The transfer label also covers 2022-23 and 2023-24, which the old rule could not label.

\textbf{{The data.}} Transfermarkt (\texttt{{davidcariboo/player-scores}}) adds transfers, fees, market values, fine positions and a stable player id. FBref removed its advanced statistics in January 2026, so the seasons stay 2017-18 to 2023-24. A Kaggle scrape with 2024-25 exists (\texttt{{emrey3lmaz}}) but was not merged; Transfermarkt already supplies the 2024-25 minutes and transfers that the outcomes need.

The mid-sem notebook was re-run unchanged first. The minutes column it uses is named \texttt{{avg\_mins\_per\_match}} but holds season totals (maximum {b['min_num_describe']['max']:.0f}), so no fix was needed. Its baseline on the refined labels: feeder base rate {pc(b['pool_hit_rate'])}; best walk-forward model, logistic regression, {mid['Logistic regression']['hits']} hits in {mid['Logistic regression']['picks']} picks; best-composite baseline {mid['Best composite (baseline)']['hits']} in {mid['Best composite (baseline)']['picks']}.

\section{{Data lake and PySpark pipeline}}
\begin{{figure}}[H]\centering\resizebox{{\linewidth}}{{!}}{{\begin{{tikzpicture}}[node distance=5mm and 7mm, b/.style={{draw,rounded corners,align=center,font=\footnotesize,minimum height=9mm,inner sep=3pt}}, >=Stealth]
\node[b](a){{Kaggle CSV\\FBref, Understat,\\Transfermarkt}};\node[b,right=of a](l){{Parquet lake\\season / league}};\node[b,right=of l](x){{Entity\\resolution}};\node[b,right=of x](e){{Labels v2,\\replacement events}};
\node[b,below=of e](mi){{Spark mining\\8 tasks}};\node[b,left=of mi](en){{Engine: translation,\\scores, ranker}};\node[b,left=of en](ev){{Evaluation}};\node[b,left=of ev](ap){{Streamlit app\\and report}};
\draw[->](a)--(l);\draw[->](l)--(x);\draw[->](x)--(e);\draw[->](e)--(mi);\draw[->](mi)--(en);\draw[->](en)--(ev);\draw[->](ev)--(ap);
\end{{tikzpicture}}}}\caption{{Pipeline. Every box after the first reads and writes Parquet under \texttt{{lake/}}.}}\end{{figure}}
All sources are written as Parquet partitioned by season and league: \texttt{{fbref}} ({n(p1['lake_rows']['fbref'])} player-seasons), \texttt{{fbref\_raw}} ({n(p1['lake_rows']['fbref_raw'])}), \texttt{{understat\_team}} ({n(p1['lake_rows']['understat_team'])} club-seasons), \texttt{{tm\_transfers}} ({n(p1['lake_rows']['tm_transfers'])}), \texttt{{tm\_valuations}} ({n(p1['lake_rows']['tm_valuations'])}), \texttt{{tm\_players}} ({n(p1['lake_rows']['tm_players'])}), plus two derived tables: league minutes per player, club and season, and league tables rebuilt from match results. The rebuilt tables agree with the mid-sem table on {pc(p2['table_agreement'])} of club-seasons. Writing the lake takes {p1['runtime_s']['write lake']:.0f} s in local mode. A query for one season and league reads one partition (the plan in notebook 01 shows the partition filter).

\textbf{{Entity resolution.}} FBref has no player id. Pass one learns the club alias map from the data: join on season, league, birth year and exact name key, then give each FBref club the Transfermarkt club most of its players belong to. All {p1['clubs_mapped']} clubs map one to one. Pass two blocks on (season, club) and compares accent-insensitive, token-sorted name keys with Spark \texttt{{levenshtein}} and a Jaro-Winkler pandas UDF, adds birth year and position group, and keeps mutual best pairs. It links {n(p1['rows_linked'])} of {n(p1['rows_total'])} outfield player-club-seasons ({pc(p1['match_rate_all'])}), and {pc(p1['match_rate_900'])} of those with 900 minutes or more; the worst league-season is {pc(p1['match_rate_900_min_cell'])}. {p1['feeders_unmatched']} of {n(p1['feeders_total'])} feeder-seasons stay unmatched and are listed in \texttt{{outputs/unmatched\_feeders.csv}}.
{fig('figs/p1_match_rate.png', 'Match rate per league and season for players with at least 900 minutes.', 0.7)}
\textbf{{Replacement events.}} An event is an elite club whose starter (1,350 minutes or more) plays under 450 league minutes for it the next season, and which signs a player of the same fine position in the summer between, who then plays at least 450 minutes. Loans are removed by walking each player's moves in date order ({n(p2['loan_out_removed'])} loan departures found). This gives {n(p2['events'])} events from {n(p2['signings'])} signings. Only {pc(p2['in_pool_share'])} of those signings ({p2['signings_in_pool']}) were inside the candidate pool (age 26 or under, 900 minutes, non-elite club in the five leagues). {pc(1 - p2['from_top5_share'])} came from clubs outside the five leagues, such as Benfica or Feyenoord, and the rest were older, at another elite club, or had played too little. The engine can only be right about the {p2['in_pool_queries']} events whose signing was in the pool, and all results below are restricted to them.
{fig('figs/p2_labels_events.png', 'Left: mid-sem label against the transfer label. Right: share of real replacement signings that came from a five-league club and that were inside the candidate pool.', 0.9)}

\section{{Post-mid-semester topics: one new algorithm per task, in PySpark}}
None of these was used before the mid-sem. Each runs in Spark (local mode) with its plan shown once in notebook 03.

\begin{{table}}[H]\centering\footnotesize\begin{{tabular}}{{p{{2.6cm}}p{{4.0cm}}p{{6.6cm}}r}}\toprule
Task & Algorithm (Spark) & Result & Time (s)\\\midrule
Similarity search & Euclidean LSH, MinHash LSH & {pc(lsh['chosen']['recall_top25'], 0)} of true top-25 neighbours kept while comparing {pc(lsh['chosen']['set_share'], 0)} of pairs; both views together keep {pc(lsh['union_signing_recall'], 0)} of real signings & {lsh['chosen']['seconds']:.1f}\\
Sequential patterns & PrefixSpan & {ps['frequent_patterns']} frequent career sequences, {ps['kept_rules']} with lift $\geq$ 1.3; flagged players move up {pc(ps['flagged_rate_test'])} vs {pc(ps['base_rate_test'])} out of time & {rt['PrefixSpan']:.1f}\\
Clustering & Bisecting k-means per fine position & {bk['archetypes']} archetypes, mean silhouette {f2(bk['silhouette_chosen_mean'])}; Cramer's V with moving up {f2(bk['cramers_v'])} & {rt['BisectingKMeans (7 positions x k=2..6)']:.0f}\\
Link analysis & GraphFrames PageRank; power-iteration clustering & pipeline rank alone AUC {f3(gr['auc_pipeline_rank_alone'])} (sales count {f3(gr['auc_sales_count_alone'])}); communities with modularity {f2(gr['pic_modularity'])} & {rt['GraphFrames PageRank (7 seasons x 2 directions)']:.0f}\\
Recommendation & ALS, implicit feedback & MRR {f3(als['mrr_als'])} vs popularity {f3(als['mrr_pop'])}: ALS loses & {rt['ALS (7 cut-offs)']:.0f}\\
Classification & LinearSVC, multilayer perceptron & LinearSVC AUC {f3(cs[('mid-sem label', 'LinearSVC')]['mean_AUC'])}, MLP {f3(cs[('mid-sem label', 'MLP (16-8)')]['mean_AUC'])} (mid-sem best {f3(mid['Logistic regression']['mean_AUC'])}) & {rt['LinearSVC + MLP + LR walk-forward']:.0f}\\
Outlier detection & k-NN distance score (self-join) & hidden-gem flag: {pc(ol['gem_move_up_rate'])} move up vs {pc(ol['rest_move_up_rate'])} & {rt['kNN outlier self-join']:.1f}\\
Stream mining & Structured Streaming, Count-Min Sketch & live team dominance reaches r = 0.9 with the season value after about {st['mean_matches_at_r_0.9']:.0f} matches & {rt['Structured Streaming replay']:.0f}\\\bottomrule
\end{{tabular}}\caption{{The eight Spark mining tasks.}}\end{{table}}

\textbf{{Similarity search (LSH).}} Random-projection LSH hashes vectors so that close ones tend to share a bucket, and only bucket-mates are compared. It generates the candidates for each event from style-neutral vectors; MinHash on top-quartile trait sets is the second view. The honest reading is that the pool is small (about {lsh['mean_pool_per_query']:.0f} same-position players per event), so LSH saves little here: the chosen setting ({int(lsh['chosen']['numHashTables'])} hash tables, bucket length {lsh['chosen']['bucketLength']:g}) still compares {pc(lsh['chosen']['set_share'], 0)} of pairs. It is in the pipeline because the same code scales to a pool of every league in the world.
{fig('figs/p3_lsh_tradeoff.png', 'LSH: share of pairs compared against recall of the exact top-25 neighbours and of the real signings.', 0.55)}
\textbf{{Sequential patterns (PrefixSpan).}} A career is an ordered list of seasons, each an itemset of club tier, league, role and form band. PrefixSpan mines the frequent orders on feeder-seasons up to 2019. The strongest multi-season pattern is {tex(top_pat['pattern'])} (lift {top_pat['lift_train']:.2f} in training, {top_pat['lift_test']:.2f} later). Out of time the flagged group moves up at {ps['oot_lift']:.2f} times the base rate (Fisher p = {ps['fisher_p']:.1e}): a real but small signal, and the train and test lifts of individual patterns correlate only {f2(ps['rank_corr_train_test_lift'])}.
{fig('figs/p3_prefixspan.png', 'Top career sequences by lift for moving up, in training and in later seasons.', 0.85)}
\textbf{{Clustering (bisecting k-means).}} Bisecting k-means splits the worst cluster in two repeatedly. It is run inside each of the seven Transfermarkt fine positions, which the mid-sem could not do with three coarse groups. It finds {bk['archetypes']} archetypes; archetype and moving up are associated (chi-square p = {bk['chi2_p']:.1e}, Cramer's V {f2(bk['cramers_v'])}). Creative archetypes are bought more often than defensive ones at every position.
{fig('figs/p3_bisecting.png', 'Silhouette by k for each position, and the archetype centroids.', 0.95)}
\textbf{{Link analysis (PageRank, PIC).}} Each permanent transfer is an edge ({n(gr['edges'])} in all). With edges pointing from buyer to seller, PageRank is high for clubs whose players are bought by important clubs. Academies and one-off amateur clubs had to be removed first, because the reversed graph rewards them. The top of the 2023 list is {', '.join(tex(r['club']) for r in gr['top_pipeline_2023'][:5])}. For feeders, the club's pipeline rank alone gives AUC {f3(gr['auc_pipeline_rank_alone'])}, which is no better than simply counting the club's past sales ({f3(gr['auc_sales_count_alone'])}). Power-iteration clustering on {n(gr['pic_clubs'])} clubs gives transfer communities with modularity {f2(gr['pic_modularity'])}; the iteration count has to be chosen (here by modularity), because too many iterations collapse every club into one group.
{fig('figs/p3_graph.png', 'Left: clubs with the highest talent-pipeline PageRank. Right: transfer communities by league.', 0.95)}
\textbf{{Recommendation (ALS).}} ALS factorises the club by archetype matrix of past signings. This did not work. On {n(als['test_signings'])} later signings, ALS ranks the bought archetype worse than league-wide popularity (MRR {f3(als['mrr_als'])} against {f3(als['mrr_pop'])}; difference 95\% interval {ci(als['mrr_als_minus_pop_ci95'])}). The training matrix has only {n(als['train_signings'])} signings across {als['clubs']} clubs, too few to learn a club's taste. The score is kept in the engine so the ablation can measure it.
{fig('figs/p3_als.png', 'Mean reciprocal rank of the archetype a club actually bought.', 0.45)}
\textbf{{Classification (LinearSVC, MLP).}} Same question and same walk-forward split as the mid-sem classifiers. LinearSVC reaches AUC {f3(cs[('mid-sem label', 'LinearSVC')]['mean_AUC'])} with {cs[('mid-sem label', 'LinearSVC')]['hits']} hits in {cs[('mid-sem label', 'LinearSVC')]['picks']} picks, level with the mid-sem logistic regression ({f3(mid['Logistic regression']['mean_AUC'])}, {mid['Logistic regression']['hits']} hits). The perceptron overfits the 700 to 1,900 training rows and is clearly worse (AUC {f3(cs[('mid-sem label', 'MLP (16-8)')]['mean_AUC'])}). On the transfer label over five test seasons LinearSVC gets AUC {f3(cs[('transfer label', 'LinearSVC')]['mean_AUC'])} and {cs[('transfer label', 'LinearSVC')]['hits']} hits in {cs[('transfer label', 'LinearSVC')]['picks']}.
{fig('figs/p3_classification.png', 'New Spark classifiers next to the mid-sem leaderboard (same label, same test seasons).', 0.62)}
\textbf{{Outlier detection (k-NN distance).}} The score is the mean distance to the ten nearest neighbours inside the archetype. Unusual players are bought slightly more often (Spearman {f3(ol['spearman_score_vs_up'])}, p = {ol['spearman_p']:.3f}). The hidden-gem flag (top decile of the score and above-median composite index) marks {ol['gems_flagged']} feeders who moved up at {pc(ol['gem_move_up_rate'])} against {pc(ol['rest_move_up_rate'])} for the rest (Fisher p = {ol['gem_fisher_p']:.1e}).
{fig('figs/p3_outlier.png', 'Move-up rate by decile of the k-NN outlier score.', 0.5)}
\textbf{{Stream mining.}} The 2023-24 Understat matches are replayed one week per micro-batch ({st['micro_batches']} batches). A stateful aggregation keeps each team's running dominance; after five weeks it correlates {f2(st['r_after_5_batches'])} with the end-of-season value and passes 0.9 at about {st['mean_matches_at_r_0.9']:.0f} matches per team. A Count-Min Sketch counts deep completions per team in fixed memory: at width {st['cms']['0.01']['width']} the mean relative error is {pc(st['cms']['0.01']['mean_rel_error'])}, and at width {st['cms']['0.1']['width']} it over-counts by {st['cms']['0.1']['mean_rel_error']:.1f} times the true count, which shows what the width buys.
{fig('figs/p3_stream.png', 'Streaming team dominance against the end-of-season value.', 0.5)}

\section{{Techniques beyond the course}}
\textbf{{Survival analysis.}} Feeder-seasons become person-period rows ({n(p2['person_period_rows'])} rows, {n(p2['person_period_events'])} events), which lets the last two seasons be used ({n(p2['person_period_rows_2022_23'])} rows). The discrete-time hazard is a Spark logistic regression refitted each season on past rows only; its walk-forward AUC is {f3(sv['discrete_time_auc_walk_forward'])} and its predicted and observed rates differ by {pc(sv['calibration_mean_abs_gap'])} on average across deciles. On {n(sv['cox_test_players'])} later entrants its concordance index is {f3(sv['c_index_discrete_time_test'])}; a Cox proportional-hazards model (\texttt{{lifelines}}) fitted as a check gets {f3(sv['c_index_cox_test'])}.

\textbf{{Learning to rank.}} LightGBM \texttt{{LGBMRanker}} (LambdaMART) with one query per replacement event, against a pointwise Spark \texttt{{GBTClassifier}}. Section 6 has the results.

\textbf{{Conformal prediction.}} Split-conformal turns a score into a list with a stated coverage. For movers: to catch 90\% of next season's real movers the hazard list must hold {pc(cm[0.9]['list_share_of_pool'], 0)} of all feeders, and it then caught {pc(cm[0.9]['coverage'])} on later seasons; a 50\% promise needs {pc(cm[0.5]['list_share_of_pool'], 0)} of feeders and caught {pc(cm[0.5]['coverage'])}. For replacement events, calibrated on {cr[0.9]['calibration_events']} events and checked on {cr[0.9]['test_events']}: a list that contains the real signing 90\% of the time needs {pc(cr[0.9]['list_share_of_pool'], 0)} of the pool (about {cr[0.9]['list_length_for_mean_pool']:.0f} players) and covered {pc(cr[0.9]['coverage'])}; a 50\% promise needs about {cr[0.5]['list_length_for_mean_pool']:.0f} players and covered {pc(cr[0.5]['coverage'])}. The guarantees hold approximately, and they show that a short list cannot promise much.
{fig('figs/p5_calibration_coverage.png', 'Hazard calibration, conformal coverage on later seasons, and the list size a guarantee costs.', 1.0)}
\textbf{{SHAP}} (TreeExplainer on the ranker) explains each candidate's rank, and \textbf{{dynamic time warping}} is compared with the simple trajectory similarity: the two correlate {f2(p4['trajectory']['spearman_traj_vs_dtw'])} and put the real signing in the top 10 equally often ({pc(p4['trajectory']['recall10_dtw'], 0)} and {pc(p4['trajectory']['recall10_traj_sim'], 0)}), so DTW adds nothing here. Only {pc(p4['pairs']['traj_age_aligned_share'], 0)} of pairs could be age-aligned, because the starter's seasons at the candidate's age are usually before 2017.

\section{{Style translation and its check on real movers}}
The mid-sem removed team style from a player's numbers. Translation runs it the other way, into the buying club:
\[\mathrm{{stat\ in\ }}C = \mathrm{{stat}}_{{\mathrm{{neutral}}}} \times \exp\Big(\sum_{{\mathrm{{supported\ terms}}}} \beta_{{\mathrm{{pos,metric,term}}}}\, x_{{\mathrm{{term}}}}(C,t)\Big) \times \mathrm{{league\ factor}}(C)\]
using the mid-sem elasticities and league offsets unchanged. The check uses {n(tr['movers'])} players who changed club between consecutive seasons ({n(tr['cross_league_movers'])} across leagues). Their per-90 numbers at the new club are predicted from (a) last season's raw numbers, (b) style-neutral numbers, and (c) neutral numbers translated into the new club, using the club's style from the season before the move.

Mean absolute error, in standard deviations of each metric: (a) {f3(mm['a_raw'])}, (b) {f3(mm['b_neutral'])}, (c) {f3(mm['c_translated'])}. Translation beats raw numbers by a small amount (difference {ci(tr['c_minus_a_ci95'])}, 95\% bootstrap) and in {tr['cells_c_beats_a']} of {tr['cells_total']} position-by-metric cells. It does not beat the plain style-neutral numbers (difference {ci(tr['c_minus_b_ci95'])}). Most of the gain is from removing the old club's style; adding the new club's style contributes nothing measurable, even with the new club's true style that season ({f3(mm['c_oracle_style'])}). So translation is kept as a display feature (the radar) and as one ranking score, with the gain over raw numbers stated at about {pc((mm['a_raw'] - mm['c_translated']) / mm['a_raw'])} of the error.
{fig('figs/p4_translation.png', 'Error in predicting a mover\'s next-season per-90s, by metric.', 0.85)}

\section{{Results}}
For each event the engine scores every same-position pool player on translated similarity, trajectory similarity, the selling club's pipeline PageRank, the buying club's ALS taste, the survival hazard, and market value (level, and gap to the departing starter), with age and composite index. A LambdaMART ranker combines them. It is trained walk-forward: events of season $Y$ are ranked by a model that saw only events up to $Y-1$. The test set is {nq} events from 2020-21 to 2023-24, with about {p5['main']['mean_pool_size']:.0f} candidates each.

\begin{{table}}[H]\centering\footnotesize\resizebox{{\linewidth}}{{!}}{{\begin{{tabular}}{{lrrrrrr}}\toprule
Method & Median rank & R@5 & R@10 [95\% CI] & R@25 & R@50 & p vs best young\\\midrule
{master_rows()}
\bottomrule\end{{tabular}}}}\caption{{Replacement recall@K and median rank of the real signing on {nq} unseen in-pool events. Intervals are bootstrap over events; the last column is a paired permutation test on recall@10 against the best-young-performer baseline.}}\end{{table}}

\textbf{{What worked.}} Shadow Squad puts the real signing in the top 10 for {pc(T[SS]['recall@10'], 0)} of events. A random order gets {pc(T[RN]['recall@10'], 0)}, the mid-sem method (raw cosine similarity) {pc(T['Raw cosine similarity to the starter']['recall@10'], 0)}, and the mid-sem result to beat (best young performer) {pc(T[BY]['recall@10'], 0)}. That is the highest recall@10 in the table.

\textbf{{What did not.}} With {nq} events the difference from the best-young-performer baseline is not statistically significant (p = {f2(T[SS]['p_vs_best_young@10'])}; difference in recall@10 {ci(p5['main']['full_minus_best_young_recall10_ci95'], 2)}). Two single scores do about as well as the full engine: the highest market value has a better median rank ({T[MV]['median_rank']:.1f} against {T[SS]['median_rank']:.1f}) and better recall@25 ({pc(T[MV]['recall@25'], 0)} against {pc(T[SS]['recall@25'], 0)}), and the hazard alone reaches {pc(T[HZ]['recall@10'], 0)} at 10 and {pc(T[HZ]['recall@25'], 0)} at 25. Elite clubs mostly sign the players everyone already values. The pointwise Spark GBT is worse than LambdaMART ({pc(T['Shadow Squad fusion with Spark GBT (pointwise)']['recall@10'], 0)} at 10).

\textbf{{Ablation.}} Removing one score at a time changes recall@10 from {pc(T[SS]['recall@10'], 0)} to {pc(T['Ablation: no translation']['recall@10'], 0)} without translation, {pc(T['Ablation: no trajectory']['recall@10'], 0)} without trajectory, {pc(T['Ablation: no PageRank']['recall@10'], 0)} without PageRank and {pc(T['Ablation: no ALS taste']['recall@10'], 0)} without ALS taste. Every interval overlaps the full model's, so no single score can be said to matter on this many events. Translation shows the largest drop.
{fig('figs/p5_recall_ablation.png', 'Left: recall@K with 95\\% bootstrap bands. Right: recall@10 with one score removed at a time.', 1.0)}
{fig('figs/p6_shap_global.png', 'Mean absolute SHAP value of each score on unseen events.', 0.6)}

\section{{Model predictions, with an explanation for each}}
The screenshots are from the Streamlit app (\texttt{{app/shadow\_squad.py}}), which reads the Parquet lake: choose a league, an elite club, a season and a departing starter. It shows the ranked list, what the club really did, the radar (starter's actual numbers, candidate raw, candidate translated into the club), the SHAP waterfall for the chosen candidate, and market values. The cases were picked by rule: the three unseen events with the best rank for the real signing, and the unseen event where a real signing was ranked worst.

\section{{Case studies}}
\subsection*{{Three hits}}
{''.join(case_text(c) for c in hits)}
\subsection*{{One miss}}
{''.join(case_text(c) for c in miss)}

\section{{Limitations and conclusion}}
\begin{{itemize}}
\item Coverage. {pc(1 - p2['in_pool_share'], 0)} of real replacement signings were outside the candidate pool, most of them from leagues the data does not cover. The engine is evaluated only on the rest.
\item Sample size. {p2['in_pool_queries']} in-pool events in seven seasons, {nq} of them unseen. The intervals are wide and the method order in the results table is not reliable.
\item Transfer records. {pc(1 - p2['feeders_with_transfer_record'])} of feeder-seasons belong to players with no record in the Transfermarkt dump and are labelled "no move". Loans are inferred from move patterns, not from a loan flag.
\item The style elasticities, league offsets, archetypes and outlier scores are fitted on all seasons without outcome labels, as in the mid-sem. The ranker, hazard, PageRank and ALS scores use only past data.
\item Local Spark. Everything runs in local mode on one machine; the data is far smaller than what Spark is for.
\end{{itemize}}
Reframing the task as "who replaces whom" gives a test that the mid-sem lacked: a real decision by a real club. On that test the engine has the best top-10 recall of the methods tried and nearly doubles the mid-sem baseline, but the margin is not significant on {nq} events, and market value alone is as good or better deeper in the list. Style translation improves on raw numbers by about {pc((mm['a_raw'] - mm['c_translated']) / mm['a_raw'])} and no more than style-neutral numbers do. The parts that held up are the transfer-based labels, the entity resolution, the survival hazard, and a dashboard that shows why each candidate is ranked where he is.
\end{{document}}
'''
open(f'{ROOT}/report/endsem.tex', 'w').write(doc)
for _ in range(2): r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'endsem.tex'], cwd=f'{ROOT}/report', capture_output=True, text=True)
if r.returncode: print(r.stdout[-1500:]); sys.exit(1)
subprocess.run(['pdfunite', f'{ROOT}/Player_Replacement_Scouting_MidSem_Report.pdf', f'{ROOT}/report/endsem.pdf', f'{ROOT}/Shadow_Squad_EndSem_Report.pdf'], check=True)
print('built Shadow_Squad_EndSem_Report.pdf;', [l for l in r.stdout.split('\n') if 'Output written' in l or 'Overfull' in l][:6])
