#!/bin/bash
# build and execute one end-sem notebook from its .py source: ./run_nb.sh endsem_01_lake
set -e; cd "$(dirname "$0")"
.venv/bin/jupytext --to ipynb --set-kernel python3 -o "$1.ipynb" "$1.py" >/dev/null
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --allow-errors "$1.ipynb" 2>&1 | grep -v WARNING | tail -2
.venv/bin/python - "$1.ipynb" <<'P'
import json, sys
nb = json.load(open(sys.argv[1])); bad = 0
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        if o['output_type'] == 'error': 
            bad += 1
            print(f'CELL {i} ERROR: {o["ename"]}: {o["evalue"][-(900 if bad == 1 else 200):]}')
print('errors:', bad); sys.exit(bad > 0)
P
