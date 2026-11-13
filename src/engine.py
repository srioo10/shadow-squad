"""Shadow Squad engine: style translation, pair features and the learning-to-rank fusion. Reads only the Parquet lake."""
import os, pickle
import numpy as np, pandas as pd
from common import ROOT, MID, SEED

AF = MID['active_features']; MET = [f[:-4] for f in AF]; RAW = [f.replace('_p90', '_p90raw') for f in AF]; ZC = ['z_' + m for m in MET]
EPS = np.array([MID['EPS_C'][f] for f in AF])
FEATS = ['sim_translated', 'traj_sim', 'pipeline_rank', 'club_taste', 'hazard', 'log_value', 'value_gap', 'age_num', 'composite_perf_index']
ABLATIONS = {'no translation': ('sim_translated', 'sim_neutral'), 'no trajectory': ('traj_sim', None), 'no PageRank': ('pipeline_rank', None), 'no ALS taste': ('club_taste', None)}
L = lambda name: f'{ROOT}/lake/{name}'


class Lake:
    def __init__(self):
        F = pd.read_parquet(L('features')).merge(pd.read_parquet(L('archetypes.parquet'))[['player_id', 'yr', 'archetype']], on=['player_id', 'yr'], how='left')
        F = F.sort_values(['player_id', 'yr']); F['dur'] = F.groupby('player_id').cumcount() + 1
        self.F = F.reset_index(drop=True); self.idx = {k: i for i, k in enumerate(zip(F['player_id'], F['yr']))}
        self.Z = F[ZC].to_numpy(); self.NEU = F[AF].to_numpy(); self.RAWV = F[RAW].to_numpy(); self.age = F['age_num'].to_numpy()
        lr = np.log(self.RAWV + EPS); g = F['fine_pos'].to_numpy()                      # raw-scale standardisation per fine position, used to compare a translated candidate with a starter's actual numbers
        self.mu = {p: lr[g == p].mean(0) for p in np.unique(g)}; self.sd = {p: lr[g == p].std(0) for p in np.unique(g)}
        el = pd.read_parquet(L('midsem/elas.parquet')); el = el[el['significant'] & el['term'].isin(['x_dom', 'x_press'])]
        self.B = {(p, t): np.array([el[(el['pos'] == p) & (el['term'] == t) & (el['metric'] == m)]['beta'].sum() for m in MET]) for p in ('DF', 'MF', 'FW') for t in ('x_dom', 'x_press')}
        off = pd.read_parquet(L('midsem/league_offsets.parquet')).set_index('league'); sig = pd.read_parquet(L('midsem/league_offsets_sig.parquet')).set_index('league')
        self.OFF = {lg: (off.loc[lg, MET].to_numpy(float) * sig.loc[lg, MET].to_numpy(bool)) for lg in off.index}; self.OFF['Premier League'] = np.zeros(len(MET))
        self.pool = pd.read_parquet(L('pool'))[['player_id', 'yr']].merge(self.F, on=['player_id', 'yr'])
        pr = pd.read_parquet(L('club_pagerank.parquet')); self.PR = pr.set_index(['club_id', 't'])['pr_pipe']; self.pr_min = float(pr['pr_pipe'].min())
        self.TA = pd.read_parquet(L('club_taste.parquet')).set_index(['club_id', 't', 'archetype'])['taste']
        self.HZ = pd.read_parquet(L('hazard.parquet')).set_index(['player_id', 'yr'])['hazard'] if os.path.exists(L('hazard.parquet')) else pd.Series(dtype=float)
        self.VAL = pd.read_parquet(L('value_at_t.parquet')).set_index(['player_id', 'yr'])['value'] if os.path.exists(L('value_at_t.parquet')) else pd.Series(dtype=float)

    def translate(self, neutral, pos_group, x_dom, x_press, league):
        """stat_in_C = stat_neutral * exp(sum of supported beta[pos, metric, term] * x_term(C, t)) * league_factor(league of C). Inverse of the mid-sem adj_team + league weights, same clip."""
        expo = np.stack([self.B[(p, 'x_dom')] for p in pos_group]) * np.reshape(x_dom, (-1, 1)) + np.stack([self.B[(p, 'x_press')] for p in pos_group]) * np.reshape(x_press, (-1, 1))
        return neutral * np.clip(np.exp(expo), 0.5, 2.0) * np.exp(np.stack([self.OFF[l] for l in np.broadcast_to(league, len(neutral))]))

    def zraw(self, vals, fine_pos):
        return (np.log(vals + EPS) - np.stack([self.mu[p] for p in fine_pos])) / np.stack([self.sd[p] for p in fine_pos])

    def _traj(self, xi, si, t):
        """Mean cosine of season vectors: candidate at t-k against the starter at the same age when that season is in the data, else against the starter at t-k."""
        delta = int(round(self.age[si] - self.age[xi])) if not (np.isnan(self.age[si]) or np.isnan(self.age[xi])) else 0
        F = self.F; px, ps = F['player_id'].iat[xi], F['player_id'].iat[si]
        for d, aligned in ((delta, True), (0, False)):
            pairs = [(self.idx.get((px, t - k)), self.idx.get((ps, t - d - k))) for k in range(3)]; pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) >= (2 if aligned and d != 0 else 1): return float(np.mean([cos(self.Z[a], self.Z[b]) for a, b in pairs])), aligned and d != 0, len(pairs)
        return np.nan, False, 0

    def pair_features(self, Q):
        """Q: one row per query with qid, C, t, S, P. Returns one row per (query, same-position pool player at t) with every candidate score."""
        P = Q.merge(self.pool, left_on=['t', 'P'], right_on=['yr', 'fine_pos']); F = self.F
        si = np.array([self.idx[k] for k in zip(P['S'], P['t'])]); xi = np.array([self.idx[k] for k in zip(P['player_id'], P['yr'])]); fp = P['P'].to_numpy()
        tr = self.translate(self.NEU[xi], P['pos_group'].to_numpy(), F['x_dom'].to_numpy()[si], F['x_press'].to_numpy()[si], F['comp'].to_numpy()[si])
        zS = self.zraw(self.RAWV[si], fp); P['sim_translated'] = rowcos(self.zraw(tr, fp), zS); P['sim_raw'] = rowcos(self.zraw(self.RAWV[xi], fp), zS); P['sim_neutral'] = rowcos(self.Z[xi], self.Z[si])
        tj = [self._traj(a, b, t) for a, b, t in zip(xi, si, P['t'])]; P['traj_sim'], P['traj_age_aligned'], P['traj_n'] = zip(*tj)
        P['pipeline_rank'] = np.log([self.PR.get(k, self.pr_min) for k in zip(P['club_id'], P['t'])])
        P['club_taste'] = [self.TA.get(k, 0.0) for k in zip(P['C'], P['t'], P['archetype'])]
        P['hazard'] = [self.HZ.get(k, np.nan) for k in zip(P['player_id'], P['yr'])]
        val = np.array([self.VAL.get(k, np.nan) for k in zip(P['player_id'], P['yr'])]); vs = np.array([self.VAL.get(k, np.nan) for k in zip(P['S'], P['t'])])
        P['value'] = val; P['log_value'] = np.log10(val); P['value_gap'] = np.abs(np.log10(val) - np.log10(vs)); P['S_value'] = vs
        for i, m in enumerate(MET): P['tr_' + m] = tr[:, i]
        return P


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def rowcos(A, B): return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def fit_ranker(train, feats, seed=SEED):
    """LambdaMART: one group per replacement event, label 1 for the player actually signed."""
    import lightgbm as lgb
    train = train.sort_values('qid'); grp = train.groupby('qid', sort=False).size().to_numpy()
    return lgb.LGBMRanker(objective='lambdarank', n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=10, subsample=1.0, colsample_bytree=1.0, random_state=seed, deterministic=True, force_row_wise=True,
                          n_jobs=1, verbose=-1).fit(train[feats], train['y'], group=grp)


def walk_forward(pairs, feats, test_seasons=(2020, 2021, 2022, 2023), keep_models=False):
    """Train on in-pool events with t <= Y-1 (LSH candidate rows only), score every same-position pool player of the events at t = Y. Non-candidates get the lowest score."""
    out, models = [], {}
    for Y in test_seasons:
        tr = pairs[(pairs['t'] <= Y - 1) & pairs['candidate'] & pairs['q_in_pool']]; tr = tr[tr.groupby('qid')['y'].transform('max') == 1]
        te = pairs[pairs['t'] == Y].copy(); m = fit_ranker(tr, feats); te['score'] = np.where(te['candidate'], m.predict(te[feats]), -1e9); out.append(te[['qid', 'player_id', 't', 'y', 'score']]); models[Y] = m
    res = pd.concat(out)
    return (res, models) if keep_models else res


def load_models(path=None):
    return pickle.load(open(path or L('models.pkl'), 'rb'))


def rank(lake, models, C, t, S, P, top=25):
    """Ranked replacement list for any (club, season, departing starter). Uses the model trained only on events before t when there is one."""
    Q = pd.DataFrame([{'qid': f'{C}_{t}_{S}', 'C': C, 't': t, 'S': S, 'P': P}]); X = lake.pair_features(Q)
    key = t if t in models else ('all' if t > max(k for k in models if k != 'all') else min(k for k in models if k != 'all'))
    X['score'] = models[key].predict(X[FEATS]); X['model_used'] = str(key); X['out_of_time'] = (key == t) or (key == 'all' and t > 2023)
    return X.sort_values('score', ascending=False).head(top).reset_index(drop=True)
