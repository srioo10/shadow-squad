"""Live engine: only features that exist for every season including the current one (Understat player-season stats + Transfermarkt).
Used for the summer-2026 test and the current shortlists. No FBref column is touched here."""
import numpy as np, pandas as pd
from rapidfuzz.distance import JaroWinkler
from common import ROOT, MID, SEED, TM_LEAGUE, FINE_POS, pkey
import engine as E

LM = ['goals', 'npg', 'xG', 'npxG', 'assists', 'xA', 'shots', 'key_passes', 'xGChain', 'xGBuildup']; ZL = ['z_' + m for m in LM]
BASE = ['sim_profile', 'traj_sim', 'seller_sales', 'log_value', 'value_gap', 'age', 'composite', 'min_share']
EXTRA = ['seller_bottom3', 'hist_club', 'hist_league', 'same_league', 'min_trend']
UL = {'EPL': 'Premier League', 'La_Liga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie_A': 'Serie A', 'Ligue_1': 'Ligue 1'}
L = lambda n: f'{ROOT}/lake/{n}'
UA = {'EPL': 'Premier League', 'Premier League': 'Premier League', 'La liga': 'La Liga', 'La Liga': 'La Liga', 'La_liga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie A': 'Serie A', 'Serie_A': 'Serie A', 'Ligue 1': 'Ligue 1', 'Ligue_1': 'Ligue 1'}
def rp(n, **k):
    """Read a lake table; partition columns come back as categoricals, so turn them into plain ints and strings."""
    d = pd.read_parquet(L(n), **k)
    for c in d.columns:
        if str(d[c].dtype) == 'category': d[c] = d[c].astype(int) if c == 'season' else d[c].astype(str)
    return d


def load_understat():
    """Understat player-season totals: 2014-15 to 2024-25 (samyogguragai dump) and 2025-26 (hiuhiuthen dump), one row per player, league and season."""
    a = pd.read_csv(f'{ROOT}/data/understat_alt/players.csv'); a['league'] = a['league_name'].map(UA); a = a[a['league'].notna()]
    a = a.rename(columns={'player_id': 'uid', 'player_name': 'name', 'time': 'minutes', 'team_title': 'team', 'season_year': 'yr'})[['league', 'yr', 'uid', 'name', 'team', 'minutes'] + LM]
    c = pd.read_csv(f'{ROOT}/data/understat/understats/La_Liga/player.csv', sep=None, engine='python').assign(league='La Liga').rename(columns={'id': 'uid', 'player_name': 'name', 'time': 'minutes', 'team_title': 'team', 'year': 'yr'})
    c = c[c['yr'] <= 2023][a.columns]
    # La Liga 2024-25 is in neither Understat dump: take the same quantities from the FBref 2024-25 file (Opta xG instead of Understat xG; xGChain and xGBuildup do not exist there and stay empty)
    e = pd.read_csv(f'{ROOT}/data/emrey/Top5_League_Players_2017to2024_dataset.csv', sep=';', low_memory=False); e = e[(e['season'] == 2425) & (e['league'] == 'ESP-La Liga')]; num = lambda col: pd.to_numeric(e[col].astype(str).str.replace(',', '.'), errors='coerce')
    d = pd.DataFrame({'league': 'La Liga', 'yr': 2024, 'uid': -1, 'name': e['player'], 'team': e['team'], 'minutes': num('Playing Time_Min'), 'goals': num('Performance_Gls'), 'npg': num('Performance_G-PK'), 'xG': num('Expected_xG'), 'npxG': num('Expected_npxG'),
                      'assists': num('Performance_Ast'), 'xA': num('Expected_xAG'), 'shots': num('Standard_Sh'), 'key_passes': num('KP_'), 'xGChain': np.nan, 'xGBuildup': np.nan})
    d = d.groupby(['league', 'yr', 'uid', 'name'], as_index=False).agg({'team': lambda s: ','.join(s), 'minutes': 'sum', **{m: (lambda s: s.sum(min_count=1)) for m in LM}})
    a = pd.concat([a, c, d[a.columns]], ignore_index=True)
    b = pd.read_csv(f'{ROOT}/data/understat_2025/understat_player_league_2025_26.csv').rename(columns={'understat_id': 'uid', 'player_name': 'name', 'understat_team': 'team', 'performance_minutes': 'minutes', 'xg': 'xG', 'np_goals': 'npg', 'npxg': 'npxG', 'xa': 'xA', 'xg_chain': 'xGChain', 'xg_buildup': 'xGBuildup'})
    b['yr'] = 2025; b = b.groupby(['league', 'yr', 'uid', 'name'], as_index=False).agg({'team': lambda s: ','.join(s), 'minutes': 'sum', **{m: 'sum' for m in LM}})
    return pd.concat([a, b[a.columns]], ignore_index=True)


def match_tm(U):
    """Understat player-seasons to Transfermarkt ids. Block on league and season; exact name key first, then Jaro-Winkler with league minutes within 15%, then minutes alone when the pair is unique."""
    cm = rp('tm_club_minutes'); pl = pd.read_parquet(L('tm_players'))[['player_id', 'name']]; pl['k'] = pl['name'].map(pkey)
    tm = cm.sort_values('minutes', ascending=False).groupby(['player_id', 'season', 'league'], as_index=False).agg(tm_minutes=('minutes', 'sum'), club_id=('club_id', 'first')).rename(columns={'season': 'yr'}).merge(pl[['player_id', 'k']], on='player_id')
    U = U.copy(); U['k'] = U['name'].map(pkey); U['row'] = np.arange(len(U)); out = []
    for (lg, yr), g in U.groupby(['league', 'yr']):
        t = tm[(tm['league'] == lg) & (tm['yr'] == yr)]; done_u, done_t = set(), set()
        ex = g.merge(t, on='k'); ex = ex[~ex['k'].duplicated(keep=False)]
        for r in ex.itertuples(): out.append((r.row, r.player_id, r.club_id, 'name')); done_u.add(r.row); done_t.add(r.player_id)
        gu = g[~g['row'].isin(done_u)]; tu = t[~t['player_id'].isin(done_t)]; cand = []
        for r in gu.itertuples():
            near = tu[(tu['tm_minutes'] - r.minutes).abs() <= 0.15 * max(r.minutes, 1) + 90]
            for q in near.itertuples():
                s = JaroWinkler.similarity(r.k, q.k); sub = set(r.k.split()) <= set(q.k.split()) or set(q.k.split()) <= set(r.k.split())
                if s >= 0.86 or sub: cand.append((max(s, 0.9 if sub else 0), r.row, q.player_id, q.club_id))
        for s, row, pid, cid in sorted(cand, reverse=True):
            if row not in done_u and pid not in done_t: out.append((row, pid, cid, 'fuzzy')); done_u.add(row); done_t.add(pid)
        gu = g[~g['row'].isin(done_u) & (g['minutes'] >= 450)]; tu = t[~t['player_id'].isin(done_t)]
        for r in gu.itertuples():
            near = tu[(tu['tm_minutes'] - r.minutes).abs() <= 0.02 * r.minutes + 10]
            if len(near) == 1 and near['player_id'].iat[0] not in done_t: out.append((r.row, near['player_id'].iat[0], near['club_id'].iat[0], 'minutes')); done_u.add(r.row); done_t.add(near['player_id'].iat[0])
    X = pd.DataFrame(out, columns=['row', 'player_id', 'club_id', 'method']); return U.merge(X, on='row', how='left').drop(columns=['row', 'k'])


def mark_loans(g):
    """Same rule as notebook 02: a move that undoes the open previous move within 18 months closes a loan; a loan made permanent counts from its start."""
    g = g.sort_values('d').reset_index(drop=True); out = np.zeros(len(g), bool); ret = np.zeros(len(g), bool); open_i = None; i = 0; d, f, c = g['d'].to_numpy(), g['f'].to_numpy(), g['c'].to_numpy()
    days = lambda x, y: (d[x] - d[y]).astype('timedelta64[D]').astype(int)
    while i < len(g):
        if open_i is not None and f[i] == c[open_i] and c[i] == f[open_i] and days(i, open_i) <= 550:
            if days(i, open_i) <= 7: open_i = i
            elif i + 1 < len(g) and f[i + 1] == f[open_i] and c[i + 1] == c[open_i] and days(i + 1, i) <= 62: ret[i] = ret[i + 1] = True; open_i = None; i += 1
            else: out[open_i] = ret[i] = True; open_i = None
        else: open_i = i
        i += 1
    return g.assign(loan_out=out, loan_return=ret)


def transfers(until='2026-10-31'):
    t = pd.read_parquet(L('tm_transfers'), columns=['player_id', 'transfer_date', 'from_club_id', 'to_club_id', 'from_club_name', 'to_club_name', 'transfer_fee']).rename(columns={'transfer_date': 'd', 'from_club_id': 'f', 'to_club_id': 'c', 'from_club_name': 'fn', 'to_club_name': 'tn', 'transfer_fee': 'fee'})
    t['d'] = pd.to_datetime(t['d']); t = t[(t['f'] != t['c']) & (t['d'] <= until)].drop_duplicates(['player_id', 'd', 'f', 'c']); t = pd.concat([mark_loans(g) for _, g in t.groupby('player_id')], ignore_index=True)
    t['from_youth'] = t['fn'].astype(str).str.contains(r'(?:U\d\d| B| II| C|Yth\.?|Youth|Primavera|Castilla|Atlètic|Reserves|Res\.)$', regex=True); return t


class Live:
    def __init__(self):
        U = match_tm(load_understat()); self.match = U.groupby('yr').apply(lambda g: pd.Series({'rows_900': int((g['minutes'] >= 900).sum()), 'matched_900': float(g.loc[g['minutes'] >= 900, 'player_id'].notna().mean())}), include_groups=False)
        self.methods = U[U['minutes'] >= 900]['method'].value_counts(dropna=False).to_dict()
        pl = pd.read_parquet(L('tm_players')); pl['dob'] = pd.to_datetime(pl['date_of_birth'], errors='coerce'); self.pl = pl.set_index('player_id')
        D = U[U['player_id'].notna() & (U['minutes'] >= 900)].copy(); D['player_id'] = D['player_id'].astype('int64'); D['club_id'] = D['club_id'].astype('int64'); D = D.sort_values('minutes', ascending=False).drop_duplicates(['player_id', 'yr'])
        D['fine_pos'] = D['player_id'].map(self.pl['sub_position']).map(FINE_POS); D = D[D['fine_pos'].notna()]; D['age'] = ((pd.to_datetime((D['yr'] + 1).astype(str) + '-06-01') - D['player_id'].map(self.pl['dob'])).dt.days / 365.25).round(1)
        for m in LM: D[m + '_p90'] = D[m] / (D['minutes'] / 90.0)
        D['min_share'] = D['minutes'] / np.where(D['league'] == 'Bundesliga', 34 * 90.0, 38 * 90.0); D = D.sort_values(['fine_pos', 'yr']).reset_index(drop=True)
        for m in LM:                                    # standardise with seasons up to the row's own
            lg_ = np.log(D[m + '_p90'].clip(lower=0) + 0.02); z = np.full(len(D), np.nan)
            for p, idx in D.groupby('fine_pos').indices.items():
                yrs = D['yr'].to_numpy()[idx]; v = lg_.to_numpy()[idx]
                for y in np.unique(yrs): past = v[(yrs <= y) & ~np.isnan(v)]; z[idx[yrs == y]] = (v[yrs == y] - past.mean()) / (past.std() + 1e-9)
            D['z_' + m] = np.nan_to_num(z)      # a metric a source does not carry counts as average for the position
        D['composite'] = D.groupby(['yr', 'fine_pos'])[[m + '_p90' for m in LM]].rank(pct=True).mean(axis=1) * 100
        V = pd.read_parquet(L('tm_valuations'), columns=['player_id', 'date', 'market_value_in_eur']).rename(columns={'market_value_in_eur': 'value'}); V['date'] = pd.to_datetime(V['date']).astype('datetime64[ns]'); V['player_id'] = V['player_id'].astype('int64'); self.V = V.dropna().sort_values('date')
        D['cut'] = pd.to_datetime((D['yr'] + 1).astype(str) + '-05-31').astype('datetime64[ns]'); D = pd.merge_asof(D.sort_values('cut'), self.V, left_on='cut', right_on='date', by='player_id', direction='backward')
        D['stale_days'] = (D['cut'] - D['date']).dt.days; D = D.drop(columns=['date'])
        tab = rp('tm_tables'); tab['n'] = tab.groupby(['season', 'league'])['club_id'].transform('size'); cx = pd.read_parquet(L('club_xref')); giants = set(cx[cx['squad'].isin({c for v in MID['GIANTS'].values() for c in v})]['club_id'])
        tab['elite'] = tab['club_id'].isin(giants) | (tab['tm_pos'] <= MID['ELITE_TOP_N']); tab['bottom3'] = tab['tm_pos'] > tab['n'] - 3; self.tab = tab; self.giants = giants
        D = D.merge(tab.rename(columns={'season': 'yr'})[['club_id', 'yr', 'tm_pos', 'elite', 'bottom3']], on=['club_id', 'yr'], how='left'); D['elite'] = D['elite'].fillna(False).astype(bool); D['bottom3'] = D['bottom3'].fillna(False).astype(bool)
        D = D.sort_values(['player_id', 'yr']); prev = D[['player_id', 'yr', 'min_share', 'value']].assign(yr=lambda d: d['yr'] + 1).rename(columns={'min_share': 'ms_prev', 'value': 'v_prev'}); D = D.merge(prev, on=['player_id', 'yr'], how='left')
        D['min_trend'] = (D['min_share'] - D['ms_prev']).fillna(0.0); D['value_trend'] = (np.log10(D['value']) - np.log10(D['v_prev'])).fillna(0.0)
        self.D = D = D.reset_index(drop=True); D['fi'] = np.arange(len(D)); self.Z = D[ZL].to_numpy(); self.names = pd.read_csv(f'{ROOT}/data/transfermarkt/clubs.csv').set_index('club_id')['name']
        w = D['minutes'].to_numpy(float); key = D[['club_id', 'yr', 'fine_pos']].copy(); key['ci'] = key.groupby(['club_id', 'yr', 'fine_pos']).ngroup(); gi = key['ci'].to_numpy(); ng = gi.max() + 1; ws = np.bincount(gi, w, ng)[:, None]
        self.CENZ = np.stack([np.bincount(gi, w * self.Z[:, j], ng) for j in range(len(ZL))], 1) / ws; v = D['value'].to_numpy(float); self.CENV = np.bincount(gi, np.nan_to_num(v), ng) / np.maximum(np.bincount(gi, ~np.isnan(v), ng), 1); self.CENV[self.CENV == 0] = np.nan
        self.CEN = key.drop_duplicates('ci'); self.T = T = transfers(); five = set(tab['club_id']) | set(pd.read_csv(f'{ROOT}/data/transfermarkt/clubs.csv').query("domestic_competition_id in ['GB1','ES1','L1','IT1','FR1']")['club_id'])
        s = T[(T['d'].dt.month.between(6, 10)) & ~((T['d'].dt.month == 10) & (T['d'].dt.day > 10)) & ~T['loan_return'] & ~T['loan_out'] & ~T['from_youth'] & T['c'].isin(five)].copy(); s['t'] = s['d'].dt.year - 1
        ok = tab[['club_id', 'season']].assign(in5=True).rename(columns={'club_id': 'c', 'season': 'ts'}); s['ts'] = np.minimum(s['t'] + 1, int(tab['season'].max())); s = s.merge(ok, on=['c', 'ts'], how='left'); s = s[s['in5'].fillna(False)]
        s = s.merge(D[['player_id', 'yr', 'fine_pos', 'club_id', 'league']].rename(columns={'yr': 't', 'club_id': 'from_id', 'league': 'from_league'}), on=['player_id', 't']); s = s[s['from_id'] != s['c']].sort_values(['d', 'fee']).drop_duplicates(['c', 't', 'player_id'])
        s['qid'] = s['c'].astype(str) + '_' + s['t'].astype(str) + '_' + s['fine_pos']; self.S = s.rename(columns={'c': 'C', 'fine_pos': 'P', 'player_id': 'N'})
        sal = T[~T['loan_out'] & ~T['loan_return'] & T['c'].isin(five)].assign(season=lambda d: np.where(d['d'].dt.month >= 6, d['d'].dt.year, d['d'].dt.year - 1)); self.sales = sal.groupby(['f', 'season']).size().rename('n').reset_index()
        self.cl = rp('tm_tables')[['club_id', 'season', 'league']].sort_values('season').drop_duplicates('club_id', keep='last').set_index('club_id')['league']

    def _cidx(self, P, k): return P[['C', 'P']].assign(yr=P['t'].to_numpy() - k).merge(self.CEN, left_on=['C', 'yr', 'P'], right_on=['club_id', 'yr', 'fine_pos'], how='left')['ci'].to_numpy()
    def _xidx(self, P, k): return P[['player_id']].assign(yr=P['t'].to_numpy() - k).merge(self.D[['player_id', 'yr', 'fi']], on=['player_id', 'yr'], how='left')['fi'].to_numpy()

    def pairs(self, Q, ref=None):
        """Q: qid, C, t, P (+ S for a starter reference). One row per candidate: every 900-minute player of that position at another club in season t."""
        D = self.D; P = Q.merge(D, left_on=['t', 'P'], right_on=['yr', 'fine_pos']); P = P[P['club_id'] != P['C']].reset_index(drop=True); xi = P['fi'].to_numpy(); nan = lambda v, m: np.where(m, v, np.nan); sims = []
        if ref == 'starter':
            si = P[['S', 't']].merge(D[['player_id', 'yr', 'fi']], left_on=['S', 't'], right_on=['player_id', 'yr'], how='left')['fi'].to_numpy().astype(int); P['sim_profile'] = E.rowcos(self.Z[xi], self.Z[si]); refv = D['value'].to_numpy()[si]
            for k in range(3):
                a = self._xidx(P, k); b = P[['S']].assign(yr=P['t'].to_numpy() - k).merge(D[['player_id', 'yr', 'fi']], left_on=['S', 'yr'], right_on=['player_id', 'yr'], how='left')['fi'].to_numpy(); m = ~np.isnan(a) & ~np.isnan(b)
                sims.append(nan(E.rowcos(self.Z[np.where(m, a, 0).astype(int)], self.Z[np.where(m, b, 0).astype(int)]), m))
        else:
            c0 = self._cidx(P, 0); ok = ~np.isnan(c0); ci = np.where(ok, c0, 0).astype(int); P['sim_profile'] = nan(E.rowcos(self.Z[xi], self.CENZ[ci]), ok); refv = nan(self.CENV[ci], ok)
            for k in range(3):
                a, b = self._xidx(P, k), (c0 if k == 0 else self._cidx(P, k)); m = ~np.isnan(a) & ~np.isnan(b); sims.append(nan(E.rowcos(self.Z[np.where(m, a, 0).astype(int)], self.CENZ[np.where(m, b, 0).astype(int)]), m))
        P['traj_sim'] = pd.DataFrame(sims).mean(axis=0).to_numpy(); P['log_value'] = np.log10(P['value']); P['ref_value'] = refv; P['value_gap'] = np.abs(P['log_value'] - np.log10(refv))
        sl = self.sales; P['seller_sales'] = 0.0; P['hist_club'] = 0.0; P['hist_league'] = 0.0
        for t, idx in P.groupby('t').indices.items():      # everything below uses transfers dated before June of t+1 only
            past = sl[(sl['season'] <= t) & (sl['season'] >= t - 2)].groupby('f')['n'].sum(); P.loc[idx, 'seller_sales'] = np.log1p(P['club_id'].iloc[idx].map(past).fillna(0).to_numpy())
            h = self.S[self.S['t'] <= t - 1]; hc = h.groupby(['C', 'from_id']).size(); hl = h.groupby(['C', 'from_league']).size(); tot = h.groupby('C').size()
            sub = P.iloc[idx]; P.loc[idx, 'hist_club'] = [hc.get(k, 0) for k in zip(sub['C'], sub['club_id'])]; P.loc[idx, 'hist_league'] = [hl.get(k, 0) / max(tot.get(k[0], 0), 1) for k in zip(sub['C'], sub['league'])]
        P['seller_bottom3'] = P['bottom3'].astype(float); P['same_league'] = (P['league'] == P['C'].map(self.cl)).astype(float)
        return P

    def events(self, t_min=2014, t_max=2025):
        S = self.S[self.S['t'].between(t_min, t_max)]; Q = S[['qid', 'C', 't', 'P']].drop_duplicates('qid').sort_values('qid'); P = self.pairs(Q)
        P = P.merge(S[['qid', 'N']].drop_duplicates().assign(y=1).rename(columns={'N': 'player_id'}), on=['qid', 'player_id'], how='left'); P['y'] = P['y'].fillna(0).astype(int); P['q_in_pool'] = True; return P


def true_rank(df, col, asc=False):
    s = df[col].fillna(-np.inf if not asc else np.inf); r = s.groupby(df['qid']).rank(ascending=asc, method='average'); return r[df['y'] == 1].groupby(df['qid']).min()


def psi(train, cur, bins=10):
    """Population stability index of one feature: current season against the training seasons."""
    q = np.unique(np.nanquantile(train, np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    a = np.histogram(train[~np.isnan(train)], q)[0] / max((~np.isnan(train)).sum(), 1); b = np.histogram(cur[~np.isnan(cur)], q)[0] / max((~np.isnan(cur)).sum(), 1); a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


def member_ranks(model, X, feats):
    """Rank of every row under each of the bagged models: the spread is the rank interval shown to the user."""
    return np.stack([pd.Series(m.predict(X[feats])).rank(ascending=False, method='min').to_numpy() for m in model.ms], 1)
