"""
mlpap_core.py
=============
Fast core for the MLPAP metaheuristic battery.

The expensive part of every metaheuristic is the deterministic decoder plus the
penalized evaluation (O(n*m) per candidate, run millions of times). Here that
hot path is JIT-compiled with numba and the population is evaluated in a single
compiled call, which is the source of the speed-up over the pure-Python version.

Decoder semantics are identical to the original implementation:
  - clients processed in non-increasing order of rho_c = w_c * q_c
  - each client assigned to the OPEN, REACHABLE (d<=Dmax), capacity-feasible hub
    of minimum weighted walking distance (= nearest, since w_c is constant per c)
  - unassigned clients are penalized; the same penalized objective is used.
"""
import json
import numpy as np
from numba import njit, prange

PENALTY = 1.0e4  # pi


# ---------------------------------------------------------------------------
# Instance preparation
# ---------------------------------------------------------------------------
class Instance:
    """numba-ready arrays for one MLPAP instance."""
    __slots__ = ("id", "n", "m", "q", "w", "d", "f", "L", "mu", "o",
                 "order", "cand_ptr", "cand_hub", "Dmax", "Pmin", "Pmax", "pi")


def prepare(record, pi=PENALTY):
    """Build an Instance from a JSON record (coordinates -> distances)."""
    clients = np.asarray(record["clients"], dtype=np.float64)
    hubs = np.asarray(record["hubs"], dtype=np.float64)
    d = np.linalg.norm(clients[:, None, :] - hubs[None, :, :], axis=2)

    q = np.asarray(record["q"], dtype=np.float64)
    w = np.asarray(record["w"], dtype=np.float64)
    Dmax = float(record["Dmax"])
    n, m = d.shape

    # client processing order: non-increasing rho_c = w_c q_c
    order = np.argsort(-(w * q)).astype(np.int64)

    # per-client reachable hubs, sorted by ascending distance (CSR layout)
    cand_hub = []
    cand_ptr = [0]
    for c in range(n):
        hs = np.where(d[c] <= Dmax)[0]
        hs = hs[np.argsort(d[c, hs], kind="stable")]
        cand_hub.extend(hs.tolist())
        cand_ptr.append(len(cand_hub))

    inst = Instance()
    inst.id = record.get("id", "?")
    inst.n, inst.m = n, m
    inst.q, inst.w, inst.d = q, w, np.ascontiguousarray(d)
    inst.f = np.asarray(record["f"], dtype=np.float64)
    inst.L = np.asarray(record["L"], dtype=np.float64)
    inst.mu = np.asarray(record["mu"], dtype=np.float64)
    inst.o = np.asarray(record["o"], dtype=np.float64)
    inst.order = order
    inst.cand_ptr = np.asarray(cand_ptr, dtype=np.int64)
    inst.cand_hub = np.asarray(cand_hub, dtype=np.int64)
    inst.Dmax, inst.Pmin, inst.Pmax = Dmax, int(record["Pmin"]), int(record["Pmax"])
    inst.pi = float(pi)
    return inst


def load_tier(path, pi=PENALTY):
    with open(path) as fh:
        doc = json.load(fh)
    return [prepare(r, pi) for r in doc["instances"]]


# ---------------------------------------------------------------------------
# JIT-compiled decoder + evaluation
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, parallel=True)
def _eval_pop(Y, q, w, d, f, L, mu, o, order, cand_ptr, cand_hub,
              Pmin, Pmax, pi):
    """Evaluate a whole population Y (Npop x m) in one compiled call.
    The candidates are independent, so the outer loop runs in parallel across
    cores (numba prange); each candidate uses thread-local work buffers.
    Returns f_tilde, f_raw, v_total (each length Npop)."""
    Npop, m = Y.shape
    n = q.shape[0]
    ft = np.empty(Npop)
    fr = np.empty(Npop)
    vt = np.empty(Npop)

    for i in prange(Npop):
        y = Y[i]
        residual = np.empty(m)
        loads = np.empty(m)
        for j in range(m):
            residual[j] = L[j] * y[j]
            loads[j] = 0.0
        assign_cost = 0.0
        v_assign = 0.0
        for idx in range(n):
            c = order[idx]
            qc = q[c]
            placed = False
            for p in range(cand_ptr[c], cand_ptr[c + 1]):
                h = cand_hub[p]
                if y[h] == 1 and residual[h] >= qc:
                    residual[h] -= qc
                    loads[h] += qc
                    assign_cost += w[c] * d[c, h]
                    placed = True
                    break
            if not placed:
                v_assign += 1.0
        facility = 0.0
        v_cap = 0.0
        v_min = 0.0
        nopen = 0
        for j in range(m):
            facility += f[j] * y[j] + o[j] * loads[j]
            cap = L[j] * y[j]
            if loads[j] > cap:
                v_cap += loads[j] - cap
            minu = mu[j] * y[j]
            if minu > loads[j]:
                v_min += minu - loads[j]
            nopen += y[j]
        v_budget = 0.0
        if Pmin > nopen:
            v_budget += Pmin - nopen
        if nopen > Pmax:
            v_budget += nopen - Pmax
        v = v_assign + v_cap + v_min + v_budget  # v_link, v_walk are 0 by construction
        f_raw = facility + assign_cost
        fr[i] = f_raw
        vt[i] = v
        ft[i] = f_raw + pi * v
    return ft, fr, vt


@njit(cache=True, fastmath=True)
def _eval_full(y, q, w, d, f, L, mu, o, order, cand_ptr, cand_hub, Pmin, Pmax, pi):
    """Detailed evaluation of one y: returns components + assignment vector."""
    m = y.shape[0]
    n = q.shape[0]
    residual = L * y.astype(np.float64)
    loads = np.zeros(m)
    assign = -np.ones(n, dtype=np.int64)
    assign_cost = 0.0
    v_assign = 0.0
    for idx in range(n):
        c = order[idx]
        qc = q[c]
        placed = False
        for p in range(cand_ptr[c], cand_ptr[c + 1]):
            h = cand_hub[p]
            if y[h] == 1 and residual[h] >= qc:
                residual[h] -= qc
                loads[h] += qc
                assign[c] = h
                assign_cost += w[c] * d[c, h]
                placed = True
                break
        if not placed:
            v_assign += 1.0
    facility = 0.0
    v_cap = 0.0
    v_min = 0.0
    nopen = 0
    for j in range(m):
        facility += f[j] * y[j] + o[j] * loads[j]
        cap = L[j] * y[j]
        if loads[j] > cap:
            v_cap += loads[j] - cap
        minu = mu[j] * y[j]
        if minu > loads[j]:
            v_min += minu - loads[j]
        nopen += y[j]
    v_budget = 0.0
    if Pmin > nopen:
        v_budget += Pmin - nopen
    if nopen > Pmax:
        v_budget += nopen - Pmax
    v_total = v_assign + v_cap + v_min + v_budget
    f_raw = facility + assign_cost
    return (f_raw + pi * v_total, f_raw,
            v_assign, v_cap, v_min, 0.0, 0.0, v_budget, assign, loads)


# ---------------------------------------------------------------------------
# Python-facing evaluators
# ---------------------------------------------------------------------------
def eval_pop(inst, Y):
    """Y: (Npop, m) int array of hub-opening vectors."""
    return _eval_pop(np.ascontiguousarray(Y.astype(np.int64)),
                     inst.q, inst.w, inst.d, inst.f, inst.L, inst.mu, inst.o,
                     inst.order, inst.cand_ptr, inst.cand_hub,
                     inst.Pmin, inst.Pmax, inst.pi)


def eval_full(inst, y):
    out = _eval_full(np.ascontiguousarray(y.astype(np.int64)),
                     inst.q, inst.w, inst.d, inst.f, inst.L, inst.mu, inst.o,
                     inst.order, inst.cand_ptr, inst.cand_hub,
                     inst.Pmin, inst.Pmax, inst.pi)
    keys = ["v_assign", "v_cap", "v_min", "v_link", "v_walk", "v_budget"]
    comp = {k: float(out[2 + i]) for i, k in enumerate(keys)}
    comp["v_total"] = float(sum(comp.values()))
    return dict(f_tilde=float(out[0]), f_raw=float(out[1]), v=comp,
                assign=out[8], loads=out[9],
                feasible=(comp["v_total"] == 0.0))


# ---------------------------------------------------------------------------
# Transfer functions (binarization)
# ---------------------------------------------------------------------------
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -60.0, 60.0)))


def v_shape(delta):
    return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * delta))
