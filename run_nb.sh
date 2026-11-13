#!/bin/bash
# build and execute one end-sem notebook from its .py source: ./run_nb.sh endsem_01_lake
set -e; cd "$(dirname "$0")"
.venv/bin/jupytext --to ipynb --set-kernel python3 -o "$1.ipynb" "$1.py" >/dev/null
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --allow-errors "$1.ipynb" > "outputs/logs/$1.log" 2>&1 || true
.venv/bin/python - "$1.ipynb" <<'P'
import json, sys
nb = json.load(open(sys.argv[1])); bad = 0
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        if o['output_type'] == 'error': 
            bad += 1
            print(f'CELL {i} ERROR: {o["ename"]}: {o["evalue"][-(900 if bad == 1 else 200):]}')
import re
with open(f'outputs/logs/{sys.argv[1][:-6]}.out', 'w') as f:      # text outputs only, so the result can be read without opening the notebook
    for i, c in enumerate(nb['cells']):
        for o in c.get('outputs', []):
            if o.get('name') == 'stderr' or o['output_type'] == 'error': continue
            t = ''.join(o.get('text') or o.get('data', {}).get('text/plain') or '')
            t = '\n'.join(l for l in t.split('\n') if 'Warning' not in l and 'require_minimum' not in l and not re.search(r'#\d{2,}', l) and not l.startswith(('== ', '*(', 'AdaptiveSpark', '+- ', ':  ', '   +', '      ')))
            if t.strip() and '<Figure' not in t: f.write(f'--[{i}] {t}\n')
print('errors:', bad); sys.exit(bad > 0)
P
