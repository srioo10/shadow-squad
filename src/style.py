"""Raw FBref rows for 2017-18 to 2024-25, Understat team style, and the walk-forward style model.
The regressions are the mid-sem ones (sections 4A, 4C, 4D of the frozen notebook) moved here unchanged, so they can be refitted on past seasons only."""
import numpy as np, pandas as pd
from common import ROOT, MID, skel

AF = MID['active_features']; MET = [f[:-4] for f in AF]; RAW = [f.replace('_p90', '_p90raw') for f in AF]; POS3 = ['DF', 'MF', 'FW']; TERMS = ['x_dom', 'x_press', 'x_str']
LG = {'EPL': 'Premier League', 'La_Liga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie_A': 'Serie A', 'Ligue_1': 'Ligue 1'}
EMREY = {'gls': 'Performance_Gls', 'npxg': 'Expected_npxG', 'sh': 'Standard_Sh', 'ast': 'Performance_Ast', 'kp': 'KP_', 'sca': 'SCA_SCA90', 'ppa': 'PPA_', 'prgp': 'Progression_PrgP', 'prgc': 'Progression_PrgC',
         'take_on': 'Take-Ons_Att', 'att_pen': 'Carries_CPA', 'tklw': 'Tackles_TklW', 'int': 'Int_', 'clr': 'Clr_', 'blocks': 'Blocks_Sh', 'aer_won': 'Aerial Duels_Won%'}
ALIAS = {**MID['ALIAS'], 'Ipswich': 'Ipswich Town'}


def _p90(d, cols):
    out = d[['player', 'squad', 'comp', 'yr', 'born', 'age_num', 'pos', 'min_num']].copy(); n90 = out['min_num'] / 90.0
    for m in MET:
        v = pd.to_numeric(d[cols[m]].astype(str).str.replace(',', '.'), errors='coerce').fillna(0); out[m + '_p90raw'] = np.round(v if m in ('sca', 'aer_won') else v / n90, 3)
    return out


def load_raw():
    """One row per player-club-season, same per-90 definitions as the mid-sem notebook. 2017-2023 from the mid-sem FBref file, 2024-25 from the emrey3lmaz scrape (taken before FBref removed these columns)."""
    a = pd.read_parquet(f'{ROOT}/lake/midsem/df_raw.parquet'); cm = MID['col_map']; a['yr'] = a['start_year'].astype(int); a['born'] = pd.to_numeric(a['born'], errors='coerce')
    A = _p90(a, {m: cm[m] for m in MET}).assign(source='akshankrithick')
    e = pd.read_csv(f'{ROOT}/data/emrey/Top5_League_Players_2017to2024_dataset.csv', sep=';', low_memory=False)
    e = e.rename(columns={'team': 'squad', 'pos_': 'pos', 'born_': 'born', 'Playing Time_Min': 'min_num'}); e['comp'] = e['league'].str.split('-', n=1).str[1]; e['yr'] = 2000 + e['season'] // 100
    e['age_num'] = pd.to_numeric(e['age_'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'); e['min_num'] = pd.to_numeric(e['min_num'], errors='coerce').fillna(0); e['born'] = pd.to_numeric(e['born'], errors='coerce')
    e = e[e['min_num'] > 0]; E = _p90(e, EMREY).assign(source='emrey3lmaz')
    chk = A[A['yr'] == 2023].merge(E[E['yr'] == 2023], on=['player', 'squad'], suffixes=('', '_e')); chk = chk[chk['min_num'] >= 900]
    agree = {m: float(np.corrcoef(chk[m + '_p90raw'], chk[m + '_p90raw_e'])[0, 1]) for m in MET}
    return pd.concat([A, E[E['yr'] == 2024]], ignore_index=True), agree, len(chk)


def team_style(fb_squads):
    """Understat club-season style: x_dom (deep-completion share), x_press (PPDA), x_str (npxG difference), each centred inside league-season. fb_squads: DataFrame comp, squad."""
    us = pd.concat([pd.read_csv(f'{ROOT}/data/understat/understats/{k}/season.csv').assign(comp=v) for k, v in LG.items()]); us = us[us['year'].between(2017, 2024)]
    look = {(c, skel(s)): s for c, s in zip(fb_squads['comp'], fb_squads['squad'])}; titles = us[['comp', 'title']].drop_duplicates()
    nm = {(c, t): look.get((c, skel(ALIAS.get(t, t)))) for c, t in zip(titles['comp'], titles['title'])}
    ctx = us.groupby(['comp', 'title', 'year']).agg(matches=('xG', 'size'), npxG=('npxG', 'sum'), npxGA=('npxGA', 'sum'), deep=('deep', 'sum'), deep_allowed=('deep_allowed', 'sum'), ppda_att=('ppda.att', 'sum'), ppda_def=('ppda.def', 'sum')).reset_index()
    ctx['squad'] = [nm[(c, t)] for c, t in zip(ctx['comp'], ctx['title'])]; ctx['yr'] = ctx['year']; g = ['comp', 'yr']
    sh = ctx['deep'] / (ctx['deep'] + ctx['deep_allowed']); ctx['x_dom'] = np.log(sh / (1 - sh)); ctx['x_dom'] -= ctx.groupby(g)['x_dom'].transform('mean')
    ctx['x_press'] = -np.log(ctx['ppda_att'] / ctx['ppda_def']); ctx['x_press'] -= ctx.groupby(g)['x_press'].transform('mean')
    ctx['x_str'] = ((ctx['npxG'] - ctx['npxGA']) / ctx['matches']).groupby([ctx['comp'], ctx['yr']]).transform(lambda s: (s - s.mean()) / s.std())
    return ctx[['comp', 'squad', 'title', 'yr', 'matches', 'x_dom', 'x_press', 'x_str']], [k for k, v in nm.items() if v is None]


def _wls(X, Y, w):
    sw = np.sqrt(w)[:, None]; return np.linalg.lstsq(X * sw, Y * sw, rcond=None)[0]


def fit_elas(P, B=200, seed=0):
    """Position-specific elasticity of each metric to team style: WLS on log per-90, 95% cluster bootstrap over club-seasons (mid-sem 4C)."""
    rng = np.random.default_rng(seed); rows = []; P = P.assign(team_id=P.groupby(['comp', 'squad', 'yr']).ngroup())
    for g in POS3:
        Pg = P[P['pos_group'] == g]; X = np.c_[np.ones(len(Pg)), Pg[TERMS].values]; Y = np.column_stack([np.log(Pg[r].values + MID['EPS_C'][f]) for r, f in zip(RAW, AF)]); w = Pg['min_num'].values / 90.0
        beta = _wls(X, Y, w); ids = Pg['team_id'].values; groups = {i: np.flatnonzero(ids == i) for i in np.unique(ids)}; keys = np.array(list(groups)); boots = np.empty((B, X.shape[1], Y.shape[1]))
        for r in range(B):
            pick = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]); boots[r] = _wls(X[pick], Y[pick], w[pick])
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        for ti, term in enumerate(TERMS, start=1):
            for mi, m in enumerate(MET): rows.append({'pos': g, 'metric': m, 'term': term, 'beta': beta[ti, mi], 'significant': bool((lo[ti, mi] > 0) or (hi[ti, mi] < 0)), 'n': len(Pg)})
    return pd.DataFrame(rows)


def expo(frame, elas):
    """Sum of supported beta x style for every row and metric (n x 16)."""
    out = np.zeros((len(frame), len(MET))); el = elas[elas['significant'] & elas['term'].isin(['x_dom', 'x_press'])]
    for r in el.itertuples(): out[:, MET.index(r.metric)] += np.where(frame['pos_group'].values == r.pos, r.beta * frame[r.term].values, 0.0)
    return out


def fit_offsets(D, T, seed=0, B=300):
    """League output level relative to the Premier League, from players who changed league between consecutive seasons (mid-sem 4D). D rows, T their team-neutral values (n x 16)."""
    rng = np.random.default_rng(seed); LE = [c for c in LG.values() if c != 'Premier League']; A = D[['player', 'yr', 'comp', 'pos_group']].copy(); A['i'] = np.arange(len(D)); Bm = A.copy(); Bm['yr'] -= 1
    mv = A.merge(Bm, on=['player', 'yr'], suffixes=('_a', '_b')); mv = mv[(mv['comp_a'] != mv['comp_b']) & (mv['pos_group_a'] == mv['pos_group_b'])]
    if len(mv) < 30: return pd.DataFrame(0.0, index=LE, columns=MET), pd.DataFrame(False, index=LE, columns=MET), len(mv)
    eps = np.array([MID['EPS_C'][f] for f in AF]); X = np.column_stack([np.ones(len(mv))] + [(mv['comp_b'] == L).astype(float) - (mv['comp_a'] == L).astype(float) for L in LE])
    Y = np.log(T[mv['i_b'].values] + eps) - np.log(T[mv['i_a'].values] + eps); b = np.linalg.lstsq(X, Y, rcond=None)[0]
    bb = np.array([np.linalg.lstsq(X[ix], Y[ix], rcond=None)[0] for ix in rng.integers(0, len(mv), (B, len(mv)))]); lo, hi = np.percentile(bb, 2.5, axis=0), np.percentile(bb, 97.5, axis=0)
    return pd.DataFrame(b[1:], index=LE, columns=MET), pd.DataFrame(((lo > 0) | (hi < 0))[1:], index=LE, columns=MET), len(mv)


def walk_forward_features(D, years):
    """For each season y, fit the style model on seasons <= y and make that season's style-neutral, Premier-League-equivalent per-90s. Nothing from a later season touches a row."""
    parts, EL, OF = [], [], []
    for y in years:
        past = D[D['yr'] <= y].reset_index(drop=True); el = fit_elas(past, seed=y); T = past[RAW].values * np.clip(np.exp(-expo(past, el)), 0.5, 2.0)
        off, sig, n_mv = fit_offsets(past, T, seed=y); cur = (past['yr'] == y).values; fac = np.ones((cur.sum(), len(MET))); comp = past.loc[cur, 'comp'].values
        for L in off.index: fac = np.where((comp == L)[:, None] & sig.loc[L].values[None, :], np.exp(-off.loc[L].values)[None, :], fac)
        out = past[cur].copy(); out[AF] = np.round(T[cur] * fac, 3); parts.append(out); EL.append(el.assign(cutoff=y))
        OF.append(off.where(sig, 0.0).reset_index(names='league').assign(cutoff=y, movers=n_mv))
    F = pd.concat(parts, ignore_index=True)
    for f in AF: F[f + '_pct'] = F.groupby(['yr', 'pos_group'])[f].rank(pct=True) * 100.0
    F['composite_perf_index'] = F[[f + '_pct' for f in AF]].mean(axis=1).round(1)
    return F, pd.concat(EL, ignore_index=True), pd.concat(OF, ignore_index=True)
