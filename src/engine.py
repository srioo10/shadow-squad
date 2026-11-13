"""Shadow Squad engine: style translation, pair features and the learning-to-rank fusion. Reads only the Parquet lake.
Everything a query of season t touches was fitted on seasons <= t: style elasticities, league offsets, standardisation, archetypes, PageRank, ALS taste, hazard."""
import os, pickle
import numpy as np, pandas as pd
from common import ROOT, MID, SEED

AF = MID['active_features']; MET = [f[:-4] for f in AF]; RAW = [f.replace('_p90', '_p90raw') for f in AF]; ZC = ['z_' + m for m in MET]
EPS = np.array([MID['EPS_C'][f] for f in AF]); Z0 = np.zeros(len(MET))
FEATS = ['sim_translated', 'traj_sim', 'pipeline_rank', 'club_taste', 'hazard', 'log_value', 'value_gap', 'age_num', 'composite_perf_index']
ABLATIONS = {'no translation': ('sim_translated', 'sim_neutral'), 'no trajectory': ('traj_sim', None), 'no PageRank': ('pipeline_rank', None), 'no ALS taste': ('club_taste', None)}
L = lambda name: f'{ROOT}/lake/{name}'


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def rowcos(A, B): return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


class Lake:
    def __init__(self):
        F = pd.read_parquet(L('features')).merge(pd.read_parquet(L('archetypes.parquet'))[['player_id', 'yr', 'archetype']], on=['player_id', 'yr'], how='left')
        F = F.sort_values(['player_id', 'yr']); F['dur'] = F.groupby('player_id').cumcount() + 1
        if os.path.exists(L('value_at_t.parquet')): F = F.merge(pd.read_parquet(L('value_at_t.parquet')), on=['player_id', 'yr'], how='left')
        else: F['value'] = np.nan
        self.F = F = F.reset_index(drop=True); self.idx = {k: i for i, k in enumerate(zip(F['player_id'], F['yr']))}; F['fi'] = np.arange(len(F))
        self.Z = F[ZC].to_numpy(); self.NEU = F[AF].to_numpy(); self.RAWV = F[RAW].to_numpy(); self.age = F['age_num'].to_numpy()
        lr = np.log(self.RAWV + EPS); g = F['fine_pos'].to_numpy(); yr = F['yr'].to_numpy(); self.mu, self.sd = {}, {}      # raw-scale standardisation per fine position and cut-off (seasons <= cut-off)
        for c in range(2017, 2025):
            for p in np.unique(g): m = (g == p) & (yr <= c); self.mu[(c, p)], self.sd[(c, p)] = lr[m].mean(0), lr[m].std(0)
        el = pd.read_parquet(L('elas_wf.parquet')); el = el[el['significant'] & el['term'].isin(['x_dom', 'x_press'])]; self.B = {}
        for (c, p, t), gdf in el.groupby(['cutoff', 'pos', 'term']):
            a = np.zeros(len(MET)); a[[MET.index(m) for m in gdf['metric']]] = gdf['beta'].to_numpy(); self.B[(c, p, t)] = a
        of = pd.read_parquet(L('offsets_wf.parquet')); self.OFF = {(r['cutoff'], r['league']): np.array([r[m] for m in MET], float) for _, r in of.iterrows()}
        self.pool = pd.read_parquet(L('pool'))[['player_id', 'yr']].merge(F, on=['player_id', 'yr'])
        self.pool_all = pd.read_parquet(L('pool_all'))[['player_id', 'yr']].merge(F, on=['player_id', 'yr']) if os.path.exists(L('pool_all')) else self.pool
        pr = pd.read_parquet(L('club_pagerank.parquet')); self.PR = pr[['club_id', 't', 'pr_pipe']]; self.pr_min = float(pr['pr_pipe'].min())
        self.TA = pd.read_parquet(L('club_taste.parquet'))
        self.HZ = pd.read_parquet(L('hazard.parquet')) if os.path.exists(L('hazard.parquet')) else pd.DataFrame(columns=['player_id', 'yr', 'hazard'])
        self.hz = self.HZ.set_index(['player_id', 'yr'])['hazard']
        # club x season x position centroids (minutes-weighted): what the club already fields at that position
        w = F['min_num'].to_numpy(float); key = F[['club_id', 'yr', 'fine_pos']].copy(); key['g'] = key.groupby(['club_id', 'yr', 'fine_pos']).ngroup(); ng = int(key['g'].max()) + 1; gi = key['g'].to_numpy()
        ws = np.bincount(gi, w, ng)[:, None]; agg = lambda V: np.stack([np.bincount(gi, w * V[:, j], ng) for j in range(V.shape[1])], 1) / ws
        self.CENR, self.CENZ = agg(self.RAWV), agg(np.nan_to_num(self.Z)); v = F['value'].to_numpy(float); self.CENV = np.bincount(gi, np.nan_to_num(v), ng) / np.maximum(np.bincount(gi, ~np.isnan(v), ng), 1); self.CENV[self.CENV == 0] = np.nan
        self.CEN = key.drop_duplicates('g').rename(columns={'g': 'ci'})
        self.STY = F.groupby(['club_id', 'yr']).agg(x_dom=('x_dom', 'first'), x_press=('x_press', 'first'), c_league=('comp', 'first')).reset_index()

    def translate(self, neutral, pos_group, x_dom, x_press, league, cutoff):
        """stat_in_C = stat_neutral * exp(sum of supported beta[pos, metric, term] * x_term(C, t)) * league_factor(league of C), with the style model of that season's cut-off. Same clip as the mid-sem adj_team."""
        n = len(neutral); cutoff = np.broadcast_to(cutoff, n); league = np.broadcast_to(league, n)
        expo = (np.stack([self.B.get((c, p, 'x_dom'), Z0) for c, p in zip(cutoff, pos_group)]) * np.reshape(x_dom, (-1, 1)) + np.stack([self.B.get((c, p, 'x_press'), Z0) for c, p in zip(cutoff, pos_group)]) * np.reshape(x_press, (-1, 1)))
        return neutral * np.clip(np.exp(expo), 0.5, 2.0) * np.exp(np.stack([self.OFF.get((c, l), Z0) for c, l in zip(cutoff, league)]))

    def zraw(self, vals, fine_pos, cutoff):
        cutoff = np.broadcast_to(cutoff, len(vals)); return (np.log(vals + EPS) - np.stack([self.mu[(c, p)] for c, p in zip(cutoff, fine_pos)])) / np.stack([self.sd[(c, p)] for c, p in zip(cutoff, fine_pos)])

    def _traj(self, xi, si, t):
        """Mean cosine of season vectors: candidate at t-k against the starter at the same age when that season is in the data, else against the starter at t-k."""
        delta = int(round(self.age[si] - self.age[xi])) if not (np.isnan(self.age[si]) or np.isnan(self.age[xi])) else 0
        px, ps = self.F['player_id'].iat[xi], self.F['player_id'].iat[si]
        for d, aligned in ((delta, True), (0, False)):
            pairs = [(self.idx.get((px, t - k)), self.idx.get((ps, t - d - k))) for k in range(3)]; pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) >= (2 if aligned and d != 0 else 1): return float(np.mean([cos(self.Z[a], self.Z[b]) for a, b in pairs])), aligned and d != 0, len(pairs)
        return np.nan, False, 0

    def _common(self, P, ref_value, tr):
        """Scores that do not depend on how the reference profile was built."""
        P = P.merge(self.PR.rename(columns={'pr_pipe': 'pipeline_rank'}), on=['club_id', 't'], how='left'); P['pipeline_rank'] = np.log(P['pipeline_rank'].fillna(self.pr_min))
        P = P.merge(self.TA.rename(columns={'club_id': 'C', 'taste': 'club_taste'}), on=['C', 't', 'archetype'], how='left'); P['club_taste'] = P['club_taste'].fillna(0.0)
        P = P.merge(self.HZ, on=['player_id', 'yr'], how='left')
        P['log_value'] = np.log10(P['value']); P['S_value'] = ref_value; P['value_gap'] = np.abs(P['log_value'] - np.log10(ref_value))
        return pd.concat([P, pd.DataFrame(tr, columns=['tr_' + m for m in MET], index=P.index)], axis=1)

    def pair_features(self, Q):
        """Replacement queries. Q: qid, C, t, S, P. One row per (query, same-position pool player at t); the reference profile is the departing starter S."""
        P = Q.merge(self.pool, left_on=['t', 'P'], right_on=['yr', 'fine_pos']).reset_index(drop=True); F = self.F
        si = np.array([self.idx[k] for k in zip(P['S'], P['t'])]); xi = P['fi'].to_numpy(); fp = P['P'].to_numpy(); t = P['t'].to_numpy()
        tr = self.translate(self.NEU[xi], P['pos_group'].to_numpy(), F['x_dom'].to_numpy()[si], F['x_press'].to_numpy()[si], F['comp'].to_numpy()[si], t)
        zS = self.zraw(self.RAWV[si], fp, t); P['sim_translated'] = rowcos(self.zraw(tr, fp, t), zS); P['sim_raw'] = rowcos(self.zraw(self.RAWV[xi], fp, t), zS); P['sim_neutral'] = rowcos(self.Z[xi], self.Z[si])
        tj = [self._traj(a, b, tt) for a, b, tt in zip(xi, si, t)]; P['traj_sim'], P['traj_age_aligned'], P['traj_n'] = zip(*tj)
        return self._common(P, F['value'].to_numpy()[si], tr)

    def pair_features_club(self, Q):
        """Club-level queries (all signings). Q: qid, C, t, P, league. Candidates: every 900-minute player of that position at another club. The reference profile is what club C fielded at that position in season t."""
        P = Q.merge(self.pool_all, left_on=['t', 'P'], right_on=['yr', 'fine_pos']); P = P[P['club_id'] != P['C']].reset_index(drop=True)
        P = P.merge(self.STY.rename(columns={'club_id': 'C', 'yr': 't', 'x_dom': 'cx_dom', 'x_press': 'cx_press'}), on=['C', 't'], how='left')
        lg = P['c_league'].fillna(P['league']).to_numpy(); xi = P['fi'].to_numpy(); fp = P['P'].to_numpy(); t = P['t'].to_numpy()
        tr = self.translate(self.NEU[xi], P['pos_group'].to_numpy(), P['cx_dom'].fillna(0).to_numpy(), P['cx_press'].fillna(0).to_numpy(), lg, t)
        def cidx(k): return P[['C', 'P']].assign(yr=t - k).merge(self.CEN, left_on=['C', 'yr', 'P'], right_on=['club_id', 'yr', 'fine_pos'], how='left')['ci'].to_numpy()
        def xidx(k): return P[['player_id']].assign(yr=t - k).merge(self.F[['player_id', 'yr', 'fi']], on=['player_id', 'yr'], how='left')['fi'].to_numpy()
        c0 = cidx(0); ok = ~np.isnan(c0); ci = np.where(ok, c0, 0).astype(int); nan = lambda v, m: np.where(m, v, np.nan)
        zC = self.zraw(self.CENR[ci], fp, t); P['sim_translated'] = nan(rowcos(self.zraw(tr, fp, t), zC), ok); P['sim_raw'] = nan(rowcos(self.zraw(self.RAWV[xi], fp, t), zC), ok); P['sim_neutral'] = nan(rowcos(self.Z[xi], self.CENZ[ci]), ok)
        sims = []
        for k in range(3):
            a, b = xidx(k), (c0 if k == 0 else cidx(k)); m = ~np.isnan(a) & ~np.isnan(b); sims.append(nan(rowcos(self.Z[np.where(m, a, 0).astype(int)], self.CENZ[np.where(m, b, 0).astype(int)]), m))
        P['traj_sim'] = pd.DataFrame(sims).mean(axis=0).to_numpy(); P['traj_age_aligned'] = False
        return self._common(P, nan(self.CENV[ci], ok), tr)


def fit_ranker(train, feats, seed=SEED):
    """LambdaMART: one group per event query, label 1 for the player actually signed."""
    import lightgbm as lgb
    train = train.sort_values('qid'); grp = train.groupby('qid', sort=False).size().to_numpy()
    return lgb.LGBMRanker(objective='lambdarank', n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=10, subsample=1.0, colsample_bytree=1.0, random_state=seed, deterministic=True, force_row_wise=True,
                          n_jobs=1, verbose=-1).fit(train[feats], train['y'], group=grp)


def walk_forward(pairs, feats, test_seasons=(2020, 2021, 2022, 2023, 2024), keep_models=False, train_pairs=None, use_candidates=True):
    """Train on events with t <= Y-1, score every candidate of the events at t = Y. `train_pairs` lets a model learn from another event set (all signings) and rank replacement queries.
    With use_candidates, training rows are the LSH candidate set and non-candidates get the lowest score."""
    src = pairs if train_pairs is None else train_pairs; out, models = [], {}
    for Y in test_seasons:
        tr = src[(src['t'] <= Y - 1) & src['q_in_pool']]
        if use_candidates and train_pairs is None: tr = tr[tr['candidate']]
        tr = tr[tr.groupby('qid')['y'].transform('max') == 1]; te = pairs[pairs['t'] == Y].copy(); m = fit_ranker(tr, feats); sc = m.predict(te[feats])
        te['score'] = np.where(te['candidate'], sc, -1e9) if (use_candidates and 'candidate' in te) else sc; out.append(te[['qid', 'player_id', 't', 'y', 'score']]); models[Y] = m
    res = pd.concat(out)
    return (res, models) if keep_models else res


def load_models(path=None): return pickle.load(open(path or L('models.pkl'), 'rb'))


def rank(lake, models, C, t, S, P, top=25):
    """Ranked replacement list for any (club, season, departing starter). Uses the model trained only on events before t when there is one."""
    Q = pd.DataFrame([{'qid': f'{C}_{t}_{S}', 'C': C, 't': t, 'S': S, 'P': P}]); X = lake.pair_features(Q); years = [k for k in models if k != 'all']
    key = t if t in models else ('all' if t > max(years) else min(years))
    X['score'] = models[key].predict(X[FEATS]); X['model_used'] = str(key); X['out_of_time'] = bool(key == t)
    return X.sort_values('score', ascending=False).head(top).reset_index(drop=True)
