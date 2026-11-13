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

