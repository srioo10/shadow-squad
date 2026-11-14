"""Post-mid-sem frequent-itemset algorithms. Itemsets and transactions are bitmasks (bit i = item i), tidsets are Python ints used as bitsets.
Taught algorithms run their support counting as Spark jobs (mapPartitions + reduceByKey); the new algorithms (CHARM, GenMax) are vertical and work on tidsets built by Spark."""
from operator import add, and_

bits = lambda m: [i for i in range(m.bit_length()) if m >> i & 1]
pc = lambda x: bin(x).count('1')


def spark_count(rdd, cands):
    """Support of every candidate itemset in one pass over the distributed transactions."""
    bc = rdd.context.broadcast(cands)
    def part(it):
        c = bc.value; acc = [0] * len(c)
        for t in it:
            for i, m in enumerate(c):
                if t & m == m: acc[i] += 1
        return [(i, v) for i, v in enumerate(acc) if v]
    out = dict(rdd.mapPartitions(part).reduceByKey(add).collect()); return [out.get(i, 0) for i in range(len(cands))]


def spark_closure(rdd, gens):
    """Closure of each generator: the intersection of all transactions that contain it."""
    bc = rdd.context.broadcast(gens)
    def part(it):
        g = bc.value; acc = [-1] * len(g)
        for t in it:
            for i, m in enumerate(g):
                if t & m == m: acc[i] &= t
        return [(i, v) for i, v in enumerate(acc) if v != -1]
    out = dict(rdd.mapPartitions(part).reduceByKey(and_).collect()); return [out.get(i, 0) for i in range(len(gens))]


def join(level):
    """Apriori candidate generation: join k-itemsets that share k-1 items, keep those whose k-subsets are all in the level."""
    L = set(level); out = set()
    for a in level:
        for b in level:
            c = a | b
            if a < b and pc(c) == pc(a) + 1 and all((c & ~(1 << i)) in L for i in bits(c)): out.add(c)
    return sorted(out)


def aclose(rdd, n_items, minsup, n_tx):
    """A-Close (Pasquier et al.): find the minimal generators level by level, then close them. Returns {closed itemset: support} and work counters."""
    stats = {'passes': 0, 'candidates': 0}; gens = {0: n_tx}; prev = {0: n_tx}; cand = [1 << i for i in range(n_items)]
    while cand:
        sup = spark_count(rdd, cand); stats['passes'] += 1; stats['candidates'] += len(cand); level = {}
        for c, s in zip(cand, sup):      # a generator is frequent and has no immediate subset with the same support
            if s >= minsup and all(prev.get(c & ~(1 << i)) != s for i in bits(c)): level[c] = s
        gens.update(level); prev = level; cand = join(sorted(level))
    g = sorted(gens); clo = spark_closure(rdd, g); stats['passes'] += 1; stats['generators'] = len(g)
    return {c: gens[x] for x, c in zip(g, clo) if c}, stats


def tidsets(rdd, n_items):
    """Vertical layout built by Spark: item -> bitset of the transactions that contain it."""
    pairs = rdd.zipWithIndex().flatMap(lambda ti: [(i, 1 << ti[1]) for i in bits(ti[0])]).reduceByKey(lambda a, b: a | b).collect()
    return {1 << i: t for i, t in pairs}


def charm(tid, minsup):
    """CHARM (Zaki & Hsiao): search itemset-tidset pairs, merging items whose tidsets coincide or nest, so only closed sets are produced."""
    closed = {}; stats = {'intersections': 0}
    def add_closed(X, t):
        closed[t] = closed.get(t, 0) | X          # itemsets with the same tidset have the same closure
    def extend(nodes):
        nodes = sorted(nodes, key=lambda n: pc(n[1]))
        for i in range(len(nodes)):
            if nodes[i] is None: continue
            X, tX = nodes[i]; kids = []
            for j in range(i + 1, len(nodes)):
                if nodes[j] is None: continue
                Y, tY = nodes[j]; t = tX & tY; stats['intersections'] += 1
                if pc(t) < minsup: continue
                if tX == tY: X |= Y; nodes[j] = None                      # property 1: same tidset, Y always occurs with X
                elif t == tX: X |= Y                                       # property 2: X occurs only where Y does
                elif t == tY: nodes[j] = None; kids.append((Y, t))          # property 3
                else: kids.append((Y, t))                                   # property 4
            if kids: extend([(X | Y, t) for Y, t in kids])
            add_closed(X, tX)
    extend([(x, t) for x, t in tid.items() if pc(t) >= minsup])
    return {X: pc(t) for t, X in closed.items()}, stats


def pincer(rdd, n_items, minsup):
    """Pincer-Search (Lin & Kedem): bottom-up Apriori passes, and in the same pass a top-down count of the maximal frequent candidate set (MFCS). An MFCS element found frequent is maximal at once, and its subsets need no counting."""
    stats = {'passes': 0, 'candidates': 0, 'saved_by_mfs': 0}; mfcs = {(1 << n_items) - 1}; mfs = set(); cand = [1 << i for i in range(n_items)]; freq_all = set()
    while cand or mfcs:
        known = [c for c in cand if any(c & m == c for m in mfs)]; todo = [c for c in cand if c not in set(known)]; top = sorted(mfcs)
        sup = spark_count(rdd, todo + top) if (todo or top) else []; stats['passes'] += 1; stats['candidates'] += len(todo) + len(top); stats['saved_by_mfs'] += len(known)
        level = set(known) | {c for c, s in zip(todo, sup) if s >= minsup}; infreq = [c for c, s in zip(todo, sup) if s < minsup]
        for m, s in zip(top, sup[len(todo):]):
            if s >= minsup: mfs.add(m); mfcs.discard(m)
        for s_ in infreq:                              # MFCS-gen: no element of the MFCS may contain an infrequent itemset
            for m in [m for m in mfcs if m & s_ == s_]:
                mfcs.discard(m)
                for i in bits(s_):
                    n = m & ~(1 << i)
                    if n and not any(n & o == n for o in mfcs | mfs): mfcs.add(n)
        freq_all |= level; cand = join(sorted(level))
        if not cand: break
        mfcs = {m for m in mfcs if pc(m) > pc(cand[0]) - 1}
        if all(any(c & m == c for m in mfs) for c in cand) and not mfcs: break
    allf = freq_all | mfs; res = {x for x in allf if not any(x != y and x & y == x for y in allf)}
    return res, stats


def genmax(tid, minsup):
    """GenMax-style depth-first search on tidsets (Gouda & Zaki): extend an itemset with its frequent tail, and cut a branch when itemset + tail already sits inside a known maximal set."""
    M = []; stats = {'intersections': 0, 'pruned_branches': 0}
    def rec(I, tI, tail):
        whole = I
        for y, _ in tail: whole |= y
        if any(whole & m == whole for m in M): stats['pruned_branches'] += 1; return
        if not tail:
            M.append(I); return
        for k, (y, ty) in enumerate(tail):
            t = tI & ty; new = []
            for z, tz in tail[k + 1:]:
                stats['intersections'] += 1
                if pc(t & tz) >= minsup: new.append((z, tz))
            rec(I | y, t, new)
    items = sorted([(x, t) for x, t in tid.items() if pc(t) >= minsup], key=lambda n: pc(n[1])); full = 0
    for _, t in items: full |= t
    rec(0, full, items)
    return {m for m in M if m and not any(m != o and m & o == m for o in M)}, stats


def brute_closed_max(tx, n_items, minsup):
    """Reference by definition, for small inputs only: all frequent itemsets by Apriori in memory, then the closed and maximal ones."""
    sup = lambda m: sum(1 for t in tx if t & m == m); freq = {}; cand = [1 << i for i in range(n_items)]
    while cand:
        level = {c: s for c in cand if (s := sup(c)) >= minsup}; freq.update(level); cand = join(sorted(level))
    closed = {x: s for x, s in freq.items() if not any(freq.get(x | (1 << i)) == s for i in range(n_items) if not x >> i & 1)}
    return closed, {x for x in freq if not any((x | (1 << i)) in freq for i in range(n_items) if not x >> i & 1)}


def lfi(tid, minsup):
    """Longest frequent itemsets: depth-first search on tidsets that keeps the best length found so far and drops any branch that cannot reach it (itemset size + tail size < best)."""
    best = {'len': 0, 'sets': set()}; stats = {'intersections': 0, 'pruned_by_length': 0}
    def rec(I, tI, tail):
        k = pc(I)
        if k + len(tail) < best['len']: stats['pruned_by_length'] += 1; return
        if k > best['len']: best['len'], best['sets'] = k, {I}
        elif k == best['len'] and k: best['sets'].add(I)
        for j, (y, ty) in enumerate(tail):
            t = tI & ty; new = []
            for z, tz in tail[j + 1:]:
                stats['intersections'] += 1
                if pc(t & tz) >= minsup: new.append((z, tz))
            rec(I | y, t, new)
    items = sorted([(x, t) for x, t in tid.items() if pc(t) >= minsup], key=lambda n: -pc(n[1])); full = 0
    for _, t in items: full |= t
    rec(0, full, items)
    return best['sets'], best['len'], stats
