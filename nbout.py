"""print the text outputs of an executed notebook, skipping noise: python nbout.py nb.ipynb [max_chars_per_output]"""
import json, sys
nb = json.load(open(sys.argv[1])); cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        if o.get('name') == 'stderr': continue
        t = ''.join(o.get('text') or o.get('data', {}).get('text/plain') or '')
        t = '\n'.join(l for l in t.split('\n') if 'Warning' not in l and not __import__('re').search(r'#\d{2,}', l) and 'require_minimum' not in l and not l.startswith(('+- ', ':  ', '   ')) or l.startswith('    '))
        if t.strip() and '<Figure' not in t: print(f'--[{i}]', t[:cap])
