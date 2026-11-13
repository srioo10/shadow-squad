# Shadow Squad: End-Semester Build Spec

Player Replacement Scouting, Analysis and Systems of Big Data, IIITDM Kancheepuram.
Hand this file to the coding agent one phase at a time. Do not start a phase before the previous phase's acceptance checks pass.

---

## 0. Ground rules for the agent

1. **Never write a number into the report or a markdown cell by hand.** Every metric comes from a cell that computed it, and is printed or saved to `outputs/metrics.json`. If a run fails, say so; do not estimate.
2. **Do not touch the mid-sem notebook logic.** Copy `player_scouting_final.ipynb` to `midsem_frozen.ipynb` and leave it alone. End-sem work goes in new notebooks (`endsem_01_lake.ipynb` ... `endsem_07_eval.ipynb`) plus a `src/` package for shared code.
3. **PySpark for every data-mining task.** Local mode is fine (`SparkSession.builder.master("local[*]")`). pandas/DuckDB may be used for plotting and small joins only. Each mining section must show its Spark plan (`df.explain()` once) and runtime.
4. **One new algorithm per data-mining task**, never one already used in the mid-sem (list in section 3).
5. Fixed seeds everywhere (`SEED = 42`). Same walk-forward splits as the mid-sem (train on seasons up to t0, test on later seasons).
6. Git commits under the repo owner's name only. No co-author or tool attribution lines.
7. Reuse the existing code for club alias matching, the Understat team-style table (`ctx`), the style elasticities (`elas`), the league offsets (`b_hat`) and the DuckDB outcome query. Port them into `src/`; do not rewrite them from scratch.

---

## 0b. Skills, plugins and token budget

**At the start of the session, list the skills and plugins you have, and map each phase to the ones that fit.** Use them; do not hand-roll what a skill already does. At minimum:

| Where | Use |
|---|---|
| All agent chat, status updates, planning | **caveman** and **ponytail**: short replies, no recaps, no restating code or files |
| Dataset checks, related work, literature for the beyond-syllabus parts | research plugins (**Exa**, **Firecrawl**) |
| Every chart and plot | **dataviz** skill: read it before writing the first plot; one consistent palette across the whole report |
| Streamlit dashboard look | **frontend-design** skill |
| All report prose, captions, findings text | **human-writer** skill: direct, specific, no inflated wording |
| Final report | **pdf** skill (and **pdf-reading** to inspect the rendered pages before delivery) |

**Token rules (agent side only, never the report):**
- Run notebooks headless (`papermill` or `jupyter nbconvert --execute`) and write logs to `outputs/logs/`. Read the log tail, not the whole notebook.
- Never print full dataframes. `head(5)`, shapes and summary rows only.
- Read only the cells or functions you are changing. Do not re-read files you just wrote.
- Do not paste code back into chat after writing it. Report: what changed, what passed, the next step.
- Cache expensive Spark stages to Parquet so re-runs skip them.

**Quality is not where to save:** the PDF, its figures and its prose get full effort. Render it, look at every page, fix overflow, clipped labels and bad page breaks, then deliver.

---

## 1. What already exists (do not rebuild)

From the current notebook:
- FBref 2017-18 to 2023-24 ingest, 16 per-90 metrics, percentiles within season x position.
- Understat team style per club-season: `x_dom`, `x_press`, `x_str`, four tactical families.
- Position-specific style elasticities (WLS + cluster bootstrap) and `adj_team()` which removes style: `stat_neutral = stat * exp(-sum(beta * x))`.
- League offsets from league-changers (`b_hat`).
- Refined outcome rule: best destination in t+1..t+2 at any minutes, five leagues, graded tiers (giant / CL place = 3, top-6 = 2, stronger league = 1).
- Five-league top-N elite reference (`ELITE_TOP_N = 6`).

**Data limit:** FBref removed its advanced (Opta) stats on 20 Jan 2026. Do not plan on 2024-25 or 2025-26 advanced stats from FBref. Check Kaggle for a pre-removal 2024-25 scrape; if none, the season range stays 2017-18 to 2023-24.

---

## Phase 0. Baseline re-run and sanity checks

- Run the current notebook end to end on the refined labels and save every headline number to `outputs/baseline_metrics.json`. These replace the mid-sem report's PL-top-3 numbers as the bar to beat.
- **Check the minutes column:** `col_map['min']` tries `avg_mins_per_match` first. Print the chosen column and the distribution of `min_num`. If it is average minutes per match rather than season total, fix the mapping and record the change.
- Record base rates per tier and per season.

**Accept when:** baseline JSON exists, minutes column is confirmed as season totals, base rates are printed.

---

## Phase 1. Data lake (PySpark + Parquet)

**New source:** Transfermarkt Kaggle dataset `davidcariboo/player-scores` (players, transfers, player_valuations, clubs, appearances). Confirm it is still available; if not, find the closest maintained Transfermarkt dump and note it.

Steps:
1. Fetch with the Kaggle CLI into `data/transfermarkt/`.
2. Write all sources as Parquet, partitioned by `season` and `league`: `lake/fbref/`, `lake/understat_team/`, `lake/tm_transfers/`, `lake/tm_valuations/`, `lake/tm_players/`.
3. **Player entity resolution (FBref <-> Transfermarkt):** blocking on (season, club after alias mapping), then match on accent-insensitive name skeleton + birth year (from age and season) + position group. Use Spark `levenshtein` / Jaro-Winkler UDF on the skeleton within blocks. Output `lake/player_xref/` with a match confidence.
4. Take **fine positions** from Transfermarkt `sub_position` (CB, FB, DM, CM, AM, W, ST).

**Accept when:** match rate per league-season is reported; 30 random matches are printed for a manual check; unmatched feeders are listed, not dropped silently.

---

## Phase 2. Labels v2 and replacement events

1. **Transfer-based outcome:** for every feeder-season, the first permanent or loan-with-option move in the next 24 months, with fee and destination tier. Compare with the old club-row-inferred label in a confusion table (this is a data-quality result for the report).
2. **Survival format:** one row per feeder per season at risk, with `event = moved_up` and right-censoring for seasons whose future is not observed. This lets 2023-24 players be scored properly.
3. **Replacement events** (the evaluation backbone):
   - Elite club C in season t has a starter S (>= 1,350 min) at fine position P.
   - In t+1, S plays < 450 min for C (left, injured long-term or dropped).
   - C signs player N in the window between t and t+1 who plays position P at C.
   - Store `(C, t, S, P, N, fee, N_was_in_candidate_pool)`.
   - Candidate pool at time t: outfield players at non-elite clubs in the five leagues, age <= 26 (parameter), >= 900 min.
   - Report honestly what share of real signings were inside the pool (many come from Portugal, the Netherlands and elsewhere, outside the data).

**Accept when:** events table built; 20 events printed for a manual check against known transfers; in-pool coverage reported.

---

## Phase 3. Data-mining tasks in PySpark (one new algorithm each)

Mid-sem already used: K-Means, Ward, GMM, DBSCAN, Apriori, FP-Growth, DHP, DIC, Eclat, Partition, A-Close, Pincer-Search, decision tree, naive Bayes, logistic regression, random forest, gradient boosting, Isolation Forest, LOF, robust Mahalanobis, cosine/Pearson/Euclidean/Manhattan/Mahalanobis similarity, ridge regression.

| Task | New algorithm (Spark) | Used for |
|---|---|---|
| Similarity search | `BucketedRandomProjectionLSH` (Euclidean LSH) on style-neutral vectors; `MinHashLSH` on binarised trait sets as a second view | candidate generation for each replacement event |
| Sequential pattern mining | `pyspark.ml.fpm.PrefixSpan` over career sequences (club tier, league, role per season) | career paths that precede a move up |
| Clustering | `BisectingKMeans` per fine position; swap for BIRCH / CURE / CLARANS if those were the post-mid lecture algorithms | role archetypes at fine-position level |
| Graph / link analysis | GraphFrames `pageRank` on the club-to-club transfer graph (edge weight = number of moves, or fee); `PowerIterationClustering` for transfer communities | "talent pipeline" score of the selling club |
| Recommendation | `ALS` on a club x archetype implicit matrix (how often club C signed archetype A) | club taste prior: which archetypes this club actually buys |
| Classification | `LinearSVC` and/or `MultilayerPerceptronClassifier` (match the post-mid syllabus) | probability that a feeder is bought by an elite club |
| Outlier detection | use the post-mid outlier method (e.g. distance-based or k-NN outlier score implemented in Spark) | hidden-gem flag inside each archetype |
| Stream mining (only if taught) | Spark Structured Streaming replaying Understat matches week by week, with Count-Min Sketch or DGIM for rolling form | live team-style update |

Each section: what the algorithm does in two sentences, why it fits this task, Spark code, runtime, result, and one plot.

**Accept when:** every task above has a Spark implementation, its own metric, and a saved figure.

---

## Phase 4. The Shadow Squad engine

For a replacement event (club C, season t, departing starter S, position P) produce a ranked list of candidates.

### 4.1 Style translation (core novelty)
Translate a candidate's numbers into the buying club's context:

```
stat_in_C = stat_neutral(candidate) * exp(sum over supported terms of beta[pos, metric, term] * x_term(C, t)) * league_factor(league of C)
```

**Validate it on real movers:** for every player who changed club between consecutive seasons (any club, any league), predict his t+1 per-90s at the new club three ways: (a) unchanged raw stats, (b) style-neutral stats, (c) translated stats. Report MAE per metric and position. If (c) does not beat (a), say so plainly and keep translation only as a display feature.

### 4.2 Candidate scores
For each candidate X:
- `sim_translated`: cosine similarity between X translated into C and S's actual profile at C.
- `traj_sim`: similarity of X's last 2-3 seasons trajectory (age-aligned) to S's trajectory at the same ages. LSH on concatenated age-aligned vectors; DTW as an out-of-course comparison on the top-200 only.
- `pipeline_rank`: PageRank of X's current club.
- `club_taste`: ALS score of C for X's archetype.
- `hazard`: survival-model probability that X moves up in the next 24 months.
- `value`: Transfermarkt market value at t (for the cheapest-replacement view).

### 4.3 Fusion
Learning-to-rank: one query per replacement event, label 1 for the actually signed player, 0 for the rest of the LSH candidate set. Train with Spark `GBTClassifier` pointwise, and LightGBM `LGBMRanker` (LambdaMART, out of course) as the comparison. Walk-forward: train on events up to t0, test on later events.

**Accept when:** the engine returns a ranked list for any (C, t, S) and the translation validation table exists.

---

## Phase 4B. Beyond the syllabus: four questions the course would never ask

The syllabus part answers "can data mining find replacements?". This part answers questions a club, an agent or a player would pay to have answered. Each one has to end in **one sentence a non-technical person would repeat**, backed by a number with a confidence interval. If the number is boring or null, report it as it is. A clean null result is still a finding.

Priority: A and B are must-have, C is strongly wanted, D is a stretch.

### A. Scouting as an investment portfolio (the "Moneyball" test)
Treat each season's shortlist as a portfolio bought at Transfermarkt market value at time t.
- Return = market value 24 months later (or the realised transfer fee if he was sold), relative to the value at t.
- Compare portfolios: Shadow Squad top-25, best-young-performer top-25, highest-market-value top-25, random 25 (Monte Carlo, 10,000 draws for a null distribution).
- Report mean return, hit rate, and a risk-adjusted return (Sharpe-style: mean / std over seasons and players).
- **Hedonic price model (out of course):** regress log market value on translated profile, age, league, minutes and club tier. The residual is the market's mispricing. Which leagues, ages and stat types does the market systematically under-price? (Expected story: a league discount or an undervalued-metric discount. Do not assume it; measure it.)
- Headline example shape: *"A club buying our top-25 each summer would have gained X% in squad value over two years, against Y% for buying the best young performers."*
- Caveat to state in the report: Transfermarkt values are crowd-estimated and react to moves, so value growth is partly circular. Use realised fees as a robustness check.

### B. Sliding doors: does moving up actually help a young player? (causal inference, out of course)
- Treatment: a feeder moves to an elite club at t+1. Control: a statistical twin who stayed.
- Twins come from the Phase 3 LSH index (same position, age, translated profile, minutes, club tier at t). Then propensity-score matching with a balance table (standardised mean differences < 0.1).
- Outcomes at t+2: minutes share, and per-90 output **translated to a neutral team style**. Style translation is what makes performance at Arsenal and at Brentford comparable here. That is the link between the engine and the causal question.
- Heterogeneous effects with a causal forest (`econml` `CausalForestDML`) or DoubleML: does the effect flip by age (e.g. <= 21 vs 23+), by position, by how dominant the buying club is?
- Headline example shape: *"Moving to an elite club at 20 costs a player Z% of his minutes and does not raise his output; at 23 the cost disappears."*
- Sensitivity: Rosenbaum bounds or an E-value, i.e. how strong an unmeasured confounder would need to be to erase the effect.

### C. Red-team the scout: can a player game the model? (adversarial ML, out of course)
Data-driven scouting is used by real clubs, and a player or agent can pad stats that are cheap to produce (take-on attempts, shots from distance, touches).
- Tag each metric as **cheap** (volume the player controls) or **costly** (efficiency or outcome).
- For players just outside the top-K, find the minimal change in cheap metrics that pushes them into the top-K (counterfactual search: DiCE, or greedy L1 search inside realistic bounds such as within the position's 95th percentile).
- **Gameability score** = share of near-miss players who can enter the top-K by changing cheap metrics alone, and the median size of the change needed.
- Then harden the model: monotonic constraints, efficiency-only features for cheap metrics, or adversarial training on padded profiles. Report gameability before and after, and what it costs in recall@K.
- Headline example shape: *"Adding 0.6 take-on attempts per 90 would lift N% of near-miss players into the shortlist; the hardened model cuts this to M% for a k-point drop in recall@10."*

### D. Stretch: budget squad rebuild (optimisation, out of course)
- Integer linear program (`PuLP` or `OR-Tools`): given a budget and the departing starters of club C, pick replacements that maximise the summed predicted translated value, subject to position, age and squad-size constraints.
- Show the cost against quality Pareto front, and expose it in the dashboard ("rebuild Tottenham's midfield for EUR 80m").

**Accept when:** A, B and C each have one results table, one figure, CIs, a limitations paragraph and the one-sentence headline, all computed.

---

## Phase 5. Evaluation

**Main metric:** replacement recall@K (K = 5, 10, 25, 50) and the median rank of the real signing, on unseen seasons, restricted to in-pool events.

**Baselines (all must be in the same table):**
1. Random candidate from the pool.
2. Best young performer at that position (composite index), which was the mid-sem result to beat.
3. Raw cosine similarity to S (the mid-sem method).
4. Highest market value at that position.
5. Shadow Squad without translation, without trajectory, without PageRank, without ALS (ablation, one at a time).

**Uncertainty:** bootstrap over events for CIs on recall@K. Permutation test against baseline 2.

**Conformal prediction (out of course):** split-conformal on the hazard / ranking score to give a list size with a coverage guarantee ("a list of k players contains a real mover with 90% probability"), checked on the test seasons.

**Survival model (out of course):** discrete-time hazard in Spark (logistic regression on person-period rows) and Cox PH via `lifelines` as a check. Report the concordance index.

**Accept when:** one master results table, CIs, the ablation, and a calibration/coverage plot exist.

---

## Phase 6. Explanations and dashboard

- **SHAP** (TreeExplainer on the ranker) for each top candidate: waterfall plot.
- **Translated radar:** candidate's raw profile vs translated into C vs S's actual profile, on one radar.
- **Case studies:** three replacement events where the real signing ranked in the top 10, and one where the engine missed badly, with the reason.
- **Streamlit app** (`app/shadow_squad.py`): choose league -> elite club -> season -> departing starter; shows the ranked list, radar, SHAP waterfall and market value. The report screenshots of model predictions come from here.

**Accept when:** the app runs locally from the Parquet lake and the screenshots are saved to `outputs/screens/`.

---

## Phase 7. Report

Order (required by the instructor): **mid-sem version first, unchanged, then the end-sem version.**

End-sem part:
0. **One-page "at a glance" first:** the four headline sentences (engine recall@K vs baseline, portfolio return, sliding-doors effect, gameability) as big numbers, the way the mid-sem summary page was built.
1. What changed since mid-sem and why (labels, data, question reframed from "who moves up" to "who replaces whom").
2. Data lake and PySpark pipeline (diagram, Parquet layout, entity-resolution match rates).
3. Post-mid-sem syllabus topics and the new algorithm per task (table).
4. Style translation and its validation on real movers.
5. The Shadow Squad engine: results table with baselines and CIs, ablation, recall@K plot.
6. Model prediction screenshots with an explanation for each (SHAP + radar).
7. Case studies (hits and a miss).
8. Beyond the syllabus: A portfolio test, B sliding doors, C red-teaming the scout, (D rebuild optimiser).
9. Techniques beyond the course, summarised (survival, conformal, learning-to-rank, hedonic pricing, causal forest, counterfactual attacks, ILP).
10. Limitations and conclusion.

Deliver as PDF, built with the pdf skill, visually consistent with the mid-sem report (same colours, section bands and takeaway boxes). Group voice ("we"). No mention of AI tools or agents anywhere in the report, notebook or slides.

---

## Syllabus slots to fill in

Replace these with the actual post-mid-sem lecture topics before Phase 3:
- Clustering algorithms taught after mid-sem: ______
- Classification algorithms taught after mid-sem: ______
- Outlier method taught after mid-sem: ______
- Stream mining taught? ______
- Link analysis / PageRank taught? ______
- Recommender systems taught? ______
