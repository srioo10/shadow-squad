"""Canonical genetic algorithm, as in the lectures: fixed-length binary chromosomes, fitness-proportional (roulette-wheel) selection, single-point crossover, bit-flip mutation. Elitism keeps the best individuals.
The fitness of a whole population is evaluated by one call, so it can be a Spark job."""
import numpy as np


def run(fitness, n_bits, pop=60, gens=40, pc=0.8, pm=None, elite=2, seed=42, init=None):
    """fitness(population as (n, n_bits) 0/1 array) -> array of fitness values (higher is better). Returns best chromosome, best fitness, history of (best, mean) per generation."""
    rng = np.random.default_rng(seed); pm = pm or 1.0 / n_bits; P = rng.integers(0, 2, (pop, n_bits)) if init is None else init.copy(); hist = []; best, best_f = None, -np.inf
    for g in range(gens):
        f = np.asarray(fitness(P), float); i = int(np.argmax(f)); hist.append((float(f.max()), float(f.mean())))
        if f[i] > best_f: best, best_f = P[i].copy(), float(f[i])
        w = f - f.min() + 1e-9; w = w / w.sum()                                     # selection probability proportional to (shifted, positive) fitness
        mates = P[rng.choice(pop, pop, p=w)]; kids = mates.copy()
        for a in range(0, pop - 1, 2):                                              # single-point crossover
            if rng.random() < pc:
                x = rng.integers(1, n_bits); kids[a, x:], kids[a + 1, x:] = mates[a + 1, x:].copy(), mates[a, x:].copy()
        kids ^= (rng.random(kids.shape) < pm)                                       # mutation
        kids[:elite] = P[np.argsort(-f)[:elite]]; P = kids
    f = np.asarray(fitness(P), float); i = int(np.argmax(f)); hist.append((float(f.max()), float(f.mean())))
    if f[i] > best_f: best, best_f = P[i].copy(), float(f[i])
    return best, best_f, hist


def decode(ch, n_slots, bits):
    """Each slot's gene is `bits` bits: the index of the chosen option."""
    w = 1 << np.arange(bits); return (ch.reshape(n_slots, bits) * w).sum(1)


def squad_fitness(q, price, pid, budget, bits):
    """Budget squad rebuild. q, price, pid: (slots, options) arrays. Sum of quality, minus penalties for overspending and for picking one player twice."""
    J, K = q.shape
    def f(P):
        out = np.empty(len(P))
        for n, ch in enumerate(np.asarray(P)):
            idx = np.minimum(decode(ch, J, bits), K - 1); ar = np.arange(J); cost = price[ar, idx].sum(); ids = pid[ar, idx]
            out[n] = q[ar, idx].sum() - 3.0 * max(0.0, cost - budget) / max(budget, 1.0) * J - 1.0 * (J - len(set(ids.tolist())))
        return out
    return f


def squad_ilp(q, price, pid, budget):
    """Exact optimum of the same problem by integer programming (PuLP + CBC)."""
    import pulp
    J, K = q.shape; pr = pulp.LpProblem('rebuild', pulp.LpMaximize); x = {(j, k): pulp.LpVariable(f'x_{j}_{k}', cat='Binary') for j in range(J) for k in range(K)}
    pr += pulp.lpSum(q[j, k] * x[j, k] for j in range(J) for k in range(K))
    for j in range(J): pr += pulp.lpSum(x[j, k] for k in range(K)) == 1
    pr += pulp.lpSum(price[j, k] * x[j, k] for j in range(J) for k in range(K)) <= budget
    for p in set(pid.ravel().tolist()): pr += pulp.lpSum(x[j, k] for j in range(J) for k in range(K) if pid[j, k] == p) <= 1
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); feasible = pulp.LpStatus[st] == 'Optimal'
    pick = np.array([next((k for k in range(K) if x[j, k].value() and x[j, k].value() > 0.5), 0) for j in range(J)]); return pick, (float(pulp.value(pr.objective)) if feasible else float('nan')), feasible
