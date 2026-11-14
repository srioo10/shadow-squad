# Shadow Squad

Player replacement scouting for Europe's top five leagues: given a club and a departing starter (or a position), rank the players who could replace him, and check the list against what clubs really did.

## Try the dashboard (no data download needed)

The small tables the app reads are in the repository.

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/shadow_squad.py
```

Pick a season ("Now" for the live shortlists, or a past season to see what the list would have said), a club, and click a player on the pitch.

## Reproduce everything

Needs Python 3.12, Java 17 or newer (Spark 4), and a Kaggle API token.

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/playwright install chromium      # only for the dashboard screenshots
./fetch_data.sh                            # seven Kaggle datasets into data/ (about 1 GB)
.venv/bin/python src/run_baseline.py       # runs midsem_frozen.ipynb unchanged; writes outputs/baseline_metrics.json and lake/midsem/
./run_nb.sh endsem_01_lake                 # Parquet lake, entity resolution, walk-forward style model
./run_nb.sh endsem_02_labels               # transfer labels, survival rows, replacement and signing events
./run_nb.sh endsem_03_mining               # Spark mining: LSH, PrefixSpan, bisecting k-means, PageRank, ALS, SVM/MLP, streaming
./run_nb.sh endsem_03a_postmid             # closed / maximal / longest frequent itemsets, distributed DBSCAN, pattern-match score
./run_nb.sh endsem_04_engine               # style translation check, candidate scores, learning-to-rank
./run_nb.sh endsem_04b_beyond              # hedonic pricing, causal forest, red-team, genetic algorithm vs ILP
./run_nb.sh endsem_05_eval                 # baselines, fee tiers, ablation, survival, conformal
./run_nb.sh endsem_05b_live                # live engine, summer-2026 test, drift check, current shortlists
./run_nb.sh endsem_06_explain              # SHAP, case studies, dashboard screenshots
.venv/bin/python src/build_report.py       # report/endsem.pdf (and the combined PDF if the mid-sem report PDF is in the folder)
```

The whole run takes about 40 minutes on a laptop. Reruns give identical numbers.

## Layout

| Path | What it is |
|---|---|
| `endsem_*.py` / `.ipynb` | each notebook and the script it is generated from (jupytext) |
| `src/` | shared code: engine, live engine, itemset algorithms, distributed DBSCAN, genetic algorithm, report builder |
| `app/shadow_squad.py` | the Streamlit dashboard |
| `lake/` | only the tables the dashboard needs; the rest is rebuilt by the notebooks |
| `outputs/metrics.json` | every number in the report, written by the notebook that computed it |
| `outputs/figs/`, `outputs/screens/` | figures and dashboard screenshots used in the report |
| `midsem_frozen.ipynb` | the mid-semester notebook, unchanged |

Every model that ranks a season is fitted on earlier seasons only. `data/`, most of `lake/`.
