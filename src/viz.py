"""Figures shared by the explanation notebook and the Streamlit app."""
import numpy as np, matplotlib.pyplot as plt, shap
import engine as E
from common import BLUE, ORANGE, GREY, INK, PAL

NICE = {'log_value': 'market value (log10 EUR)', 'value_gap': 'value gap to the starter', 'sim_translated': 'translated similarity', 'traj_sim': 'trajectory similarity', 'pipeline_rank': 'club pipeline PageRank (log)',
        'club_taste': 'club taste for his archetype (ALS)', 'hazard': 'move-up hazard', 'age_num': 'age', 'composite_perf_index': 'composite index', 'pattern_match': 'pattern-match score (closed itemsets)'}


def radar(lake, row, S, t):
    """Candidate raw vs candidate translated into the buying club vs the departed starter's actual numbers, in SD units of the position."""
    fp = [row['fine_pos']]; xi, si = lake.idx[(row['player_id'], t)], lake.idx[(S, t)]
    raw = lake.zraw(lake.RAWV[[xi]], fp, t)[0]; tr = lake.zraw(np.array([[row['tr_' + m] for m in E.MET]], float), fp, t)[0]; st = lake.zraw(lake.RAWV[[si]], fp, t)[0]
    ang = np.linspace(0, 2 * np.pi, len(E.MET), endpoint=False); cl = lambda v: np.append(np.clip(v, -2.5, 2.5), np.clip(v[0], -2.5, 2.5)); a = np.append(ang, ang[0])
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={'polar': True})
    ax.plot(a, cl(st), color=INK, lw=2, label=f"{lake.F['player'].iat[si]} at the club (actual)"); ax.fill(a, cl(st), color=INK, alpha=.06)
    ax.plot(a, cl(raw), color=GREY, lw=1.3, ls='--', label=f"{row['player']} raw, at {row['squad']}"); ax.plot(a, cl(tr), color=ORANGE, lw=2, label=f"{row['player']} translated into the club")
    ax.set_xticks(ang, E.MET, fontsize=7); ax.set_ylim(-2.6, 2.6); ax.set_yticks([-2, -1, 0, 1, 2], ['-2', '', '0', '', '+2'], fontsize=6); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=7, frameon=False)
    fig.tight_layout(); return fig


def explain(model, X):
    """SHAP explanation of the bagged ranker: the mean of its members' TreeExplainer values."""
    exs = [shap.TreeExplainer(m)(X) for m in model.ms]; ex = exs[0]; ex.values = np.mean([e.values for e in exs], axis=0); ex.base_values = np.mean([e.base_values for e in exs], axis=0); return ex


def waterfall(model, X, i):
    """SHAP waterfall of one candidate's ranking score (TreeExplainer on the LambdaMART model)."""
    ex = explain(model, X[E.FEATS]); ex.feature_names = [NICE[f] for f in E.FEATS]
    plt.figure(); shap.plots.waterfall(ex[i], max_display=9, show=False); fig = plt.gcf(); fig.set_size_inches(6.4, 3.8); fig.tight_layout(); return fig
