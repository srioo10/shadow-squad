"""Shared helpers for the end-sem notebooks (Spark session, name keys, metric log)."""
import json, os, re, time, unicodedata
from contextlib import contextmanager
import warnings; warnings.filterwarnings('ignore')

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM_LEAGUE = {'GB1': 'Premier League', 'ES1': 'La Liga', 'L1': 'Bundesliga', 'IT1': 'Serie A', 'FR1': 'Ligue 1'}
LEAGUES = list(TM_LEAGUE.values())
FINE_POS = {'Centre-Back': 'CB', 'Left-Back': 'FB', 'Right-Back': 'FB', 'Defensive Midfield': 'DM', 'Central Midfield': 'CM', 'Attacking Midfield': 'AM',
            'Left Winger': 'W', 'Right Winger': 'W', 'Left Midfield': 'W', 'Right Midfield': 'W', 'Centre-Forward': 'ST', 'Second Striker': 'ST'}
MID = json.load(open(f'{ROOT}/lake/midsem/config.json')) if os.path.exists(f'{ROOT}/lake/midsem/config.json') else {}


def spark(app='shadow_squad', packages=None):
    import sys; os.environ['PYSPARK_PYTHON'] = os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable   # workers must use the venv python, not the system one
    os.environ['PYTHONPATH'] = f"{ROOT}/src:" + os.environ.get('PYTHONPATH', '')   # UDFs import this module on the workers
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.master('local[*]').appName(app).config('spark.driver.memory', '10g').config('spark.sql.shuffle.partitions', 16)
         .config('spark.ui.showConsoleProgress', 'false').config('spark.sql.session.timeZone', 'UTC').config('spark.sql.sources.partitionOverwriteMode', 'dynamic'))
    if packages: b = b.config('spark.jars.packages', packages)
    s = b.getOrCreate(); s.sparkContext.setLogLevel('ERROR'); s.sparkContext.setCheckpointDir(f'{ROOT}/lake/_chk')
    return s


def skel(s):  # club skeleton, ported unchanged from the mid-sem notebook (section 4A)
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if ch.isascii() and ch.isalpha()).lower()
    return re.sub('[aeiou]', '', s)


_TR = str.maketrans({'ø': 'o', 'đ': 'd', 'ł': 'l', 'ß': 'ss', 'æ': 'ae', 'ı': 'i', 'þ': 'th', 'ð': 'd', 'œ': 'oe'})
def pkey(s):  # player name key: accent-insensitive, lower case, tokens sorted (so "Son Heung-min" == "Heung-min Son")
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s).lower().translate(_TR)) if not unicodedata.combining(ch))
    s = ''.join(ch if (ch.isascii() and ch.isalpha()) else ' ' for ch in s)
    return ' '.join(sorted(s.split()))


@contextmanager
def timed(label, store=None):
    t = time.time(); yield
    dt = time.time() - t; print(f'[{label}] {dt:.1f} s')
    if store is not None: store[label] = round(dt, 2)


def put_metrics(section, values, path=f'{ROOT}/outputs/metrics.json'):
    """Every reported number goes through here; the report reads only this file."""
    m = json.load(open(path)) if os.path.exists(path) else {}
    m.setdefault(section, {}).update(json.loads(json.dumps(values, default=lambda o: o.item() if hasattr(o, 'item') else str(o))))
    json.dump(m, open(path, 'w'), indent=1); return values
