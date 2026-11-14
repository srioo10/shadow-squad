"""Grid-partitioned distributed DBSCAN in Spark. Points are cut into slabs along the first principal axis; each slab also receives a halo of width eps from its neighbours,
so every point's eps-neighbourhood is complete inside its home slab. Core points are found per slab, clusters are built per slab, and clusters that share a core point across slabs are merged."""
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph


def dist_dbscan(sp, X, eps, min_pts, n_slabs=4):
    """X: (n, d) array. Returns labels (-1 = noise), core mask, and counters. Same definition as sklearn: a core point has at least min_pts points (itself included) within eps."""
    n = len(X); edges = np.quantile(X[:, 0], np.linspace(0, 1, n_slabs + 1)); edges[0], edges[-1] = -np.inf, np.inf
    home = np.clip(np.searchsorted(edges, X[:, 0], side='right') - 1, 0, n_slabs - 1); rows = []
    for s in range(n_slabs):
        m = (X[:, 0] >= edges[s] - eps) & (X[:, 0] < edges[s + 1] + eps); idx = np.flatnonzero(m); rows.append(pd.DataFrame({'slab': s, 'pid': idx, 'inner': home[idx] == s}))
    R = pd.concat(rows); cols = [f'x{j}' for j in range(X.shape[1])]; R[cols] = X[R['pid'].to_numpy()]; sdf = sp.createDataFrame(R).repartition(n_slabs, 'slab').cache()
    def cores(g):                                   # pass 1: core status of the points this slab owns (their neighbourhood is complete here)
        G = radius_neighbors_graph(g[cols].to_numpy(), eps, include_self=True); cnt = np.asarray(G.sum(1)).ravel()
        return g.loc[g['inner'].to_numpy(), ['pid']].assign(core=(cnt >= min_pts)[g['inner'].to_numpy()])
    core = sdf.groupBy('slab').applyInPandas(cores, 'pid long, core boolean').toPandas().set_index('pid')['core'].reindex(range(n)).to_numpy(bool)
    bc = sp.sparkContext.broadcast(core)
    def local(g):                                   # pass 2: connect core points within eps; a non-core point joins the cluster of a core neighbour
        P = g['pid'].to_numpy(); c = bc.value[P]; G = radius_neighbors_graph(g[cols].to_numpy(), eps, include_self=False).tocsr(); lab = np.full(len(g), -1)
        if c.any():
            k, comp = connected_components(G[c][:, c], directed=False); lab[c] = comp
            for i in np.flatnonzero(~c):
                nb = G.indices[G.indptr[i]:G.indptr[i + 1]]; nb = nb[c[nb]]
                if len(nb): lab[i] = lab[nb[np.argmin(P[nb])]]          # deterministic choice for border points
        return pd.DataFrame({'slab': g['slab'].to_numpy(), 'pid': P, 'inner': g['inner'].to_numpy(), 'lab': lab})
    L = sdf.groupBy('slab').applyInPandas(local, 'slab long, pid long, inner boolean, lab long').toPandas(); sdf.unpersist()
    L = L[L['lab'] >= 0].assign(key=lambda d: d['slab'] * 10 ** 6 + d['lab']); parent = {k: k for k in L['key'].unique()}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    merges = 0
    for pid, g in L[core[L['pid'].to_numpy()]].groupby('pid'):          # a core point seen in two slabs glues their clusters together
        ks = g['key'].to_numpy()
        for k in ks[1:]:
            a, b = find(ks[0]), find(k)
            if a != b: parent[b] = a; merges += 1
    H = L[L['inner']].copy(); H['root'] = [find(k) for k in H['key']]; out = np.full(n, -1); ids = {r: i for i, r in enumerate(sorted(H['root'].unique()))}
    out[H['pid'].to_numpy()] = H['root'].map(ids).to_numpy()
    return out, core, {'slabs': n_slabs, 'rows_with_halo': len(R), 'halo_share': float(len(R) / n - 1), 'merges': merges}
