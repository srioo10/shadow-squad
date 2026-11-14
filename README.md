# Shadow Squad

Player replacement scouting for Europe's top five leagues: given an elite club, a season and a departing starter, rank the players who could replace him.

## Run order

```
uv venv -p 3.12 .venv && uv pip install -p .venv pyspark pandas duckdb scikit-learn lightgbm lifelines shap streamlit matplotlib seaborn plotly kaggle pyarrow scipy statsmodels nbconvert nbclient ipykernel mlxtend rapidfuzz graphframes-py dtaidistance jupytext playwright six econml reportlab pypdf
.venv/bin/playwright install chromium
# Kaggle datasets into data/fbref, data/standings, data/understat, data/transfermarkt, data/emrey (emrey3lmaz/top-5-league-football-player-stats-2017-2025) (see midsem_frozen.ipynb cell 1; Transfermarkt = davidcariboo/player-scores)
.venv/bin/python src/run_baseline.py     # Phase 0: runs midsem_frozen.ipynb unchanged, writes outputs/baseline_metrics.json and lake/midsem/
./run_nb.sh endsem_01_lake               # Parquet lake, FBref <-> Transfermarkt entity resolution
./run_nb.sh endsem_02_labels             # transfer-based labels, survival rows, replacement events
./run_nb.sh endsem_03_mining             # eight Spark mining tasks
./run_nb.sh endsem_04_engine             # style translation, candidate scores, learning-to-rank
./run_nb.sh endsem_04b_beyond            # portfolio + hedonic pricing, sliding doors (causal forest), red-team
./run_nb.sh endsem_05_eval               # baselines, ablation, bootstrap, survival, conformal
./run_nb.sh endsem_06_explain            # SHAP, case studies, dashboard screenshots
.venv/bin/python src/build_report.py     # Shadow_Squad_EndSem_Report.pdf (mid-sem report first, then the end-sem part)
.venv/bin/streamlit run app/shadow_squad.py
```

Each notebook is generated from its `.py` source (jupytext percent format). Every reported number is written to `outputs/metrics.json` by the notebook that computed it; the report reads only that file. Needs Java 17 or newer for Spark 4. Notebook logs go to `outputs/logs/`. Every model that ranks season t is fitted on seasons up to t; reruns give identical numbers.
