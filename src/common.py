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


# ---- one plot theme for every figure: the mid-sem notebook's palette (validated with the dataviz palette checker) ----
BLUE, ORANGE, GREEN, AMBER, PINK, VIOLET, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#898781'
PAL = [BLUE, ORANGE, GREEN, AMBER, PINK, VIOLET]; INK, INK2, GRID, PAPER = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
try:
    import matplotlib as _mpl
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from cycler import cycler as _cycler
    SEQ = _LSC.from_list('seq_blue', ['#eef5fe', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b']); DIV = _LSC.from_list('div_orange_blue', [ORANGE, '#f0efec', BLUE])
    _mpl.rcParams.update({'axes.prop_cycle': _cycler(color=PAL), 'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER, 'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8,
                          'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True, 'text.color': INK, 'axes.labelcolor': INK2,
                          'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                          'legend.frameon': False, 'legend.fontsize': 8, 'lines.linewidth': 2, 'lines.markersize': 5, 'font.family': 'DejaVu Sans', 'savefig.dpi': 200, 'figure.dpi': 110})
except ImportError:
    pass
# a method keeps its colour in every figure
MCOL = {'Shadow Squad': ORANGE, 'Market value': BLUE, 'Best young performer': GREEN, 'Random': GREY, 'Hazard': VIOLET, 'Raw cosine': PINK, 'Under-priced (hedonic)': AMBER, 'Value fit': AMBER}
def fmt_p(p):
    """p-values the way the report prints them: 0.97, 0.04, <0.001."""
    return '<0.001' if p < 0.001 else (f'{p:.3f}' if p < 0.01 else f'{p:.2f}')
