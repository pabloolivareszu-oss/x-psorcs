"""
cmlpap_core.py
==============
Núcleo del CMLPAP — localización CONTINUA de microhubs con asignación peatonal
capacitada. Es la versión continua del MLPAP: en lugar de decidir CUÁLES de los m
sitios candidatos abrir (vector binario), se decide DÓNDE ubicar los p hubs
(coordenadas en el plano).

Corresponde al *capacitated multi-facility Weber problem* con costes fijos de
instalación, problema no convexo clásico de localización continua [R3].

Motivación (medida sobre la campaña del MLPAP binario):
  - El modelo online alcanza AUC 0,443-0,481 en la familia binaria: por debajo del
    azar, y sigue por debajo aunque se haga el objetivo determinista (MLPAP-TW) y
    se calibre el controlador. En FIR, continuo nativo, la capa online gana 7/8.
  - Hipótesis: la binarización por umbral es un mapeo muchos-a-uno, y genera un
    paisaje de alta NEUTRALIDAD (mesetas) donde los estadísticos de población no
    informan sobre la mejora futura. El marco y la métrica —grado de neutralidad—
    están establecidos en la literatura de análisis de paisajes de aptitud [R6].

Este módulo permite el experimento controlado que contrasta esa hipótesis: como
2*Pmax = m en los seis tiers, la dimensión de búsqueda es IDÉNTICA a la de la
familia binaria instancia por instancia. Misma geografía, misma dimensión, mismo
presupuesto; lo único que cambia es la codificación.

Decisiones de modelado heredadas de mlpap_tw_core:
  - Toda solución es factible por construcción: nada de penalización dura de 1e4.
    Lo que no se puede servir se externaliza a un coste explícito y finito.
  - Objetivo determinista: f(x) es una función pura.

Referencias citadas por clave en el resto del paquete:
  [R3] Irawan, Salhi y Soemadi, J. Glob. Optim., 78(2):271-294, 2020.
  [R6] Zou et al., "A survey of fitness landscape analysis for optimization",
       Neurocomputing, 503:129-139, 2022.
"""
from __future__ import annotations

import json

import numpy as np
from numba import njit, prange

# Niveles de restricción: multiplicador del radio peatonal Dmax. Radio más corto
# = más clientes fuera de alcance = decisiones de ubicación más acopladas.
LEVELS = {
    "low": 1.25,
    "medium": 1.00,
    "high": 0.85,
    "extra": 0.70,
    "max": 0.55,
}

# Costes económicos finitos, como múltiplos del coste medio de asignación de la
# instancia, de modo que escalan solos con el tamaño y no hay que recalibrarlos.
OUTSOURCING_RATE = 3.0   # servir una unidad de demanda con un tercero
IDLE_RATE = 0.5          # capacidad instalada por debajo de su utilización mínima

# Holgura de capacidad total sobre la demanda total.
CAPACITY_SLACK = 1.25

# Vecinos considerados al interpolar el coste del suelo desde los sitios candidatos.
IDW_K = 4
IDW_POWER = 2.0


class ContInstance:
    """Arrays numba-ready de una instancia CMLPAP."""
    __slots__ = ("id", "level", "n", "m", "p", "q", "w", "clients", "sites",
                 "site_cost", "Dmax", "Dmax_base", "hub_capacity", "mu_hub",
                 "order", "lower", "upper", "unit_out", "unit_idle")


def prepare(record, level="medium"):
    """Construye una ContInstance desde un registro JSON del MLPAP original.

    Reutiliza clientes, demandas, pesos, sitios candidatos y sus costes fijos. Lo
    que cambia es qué se decide: no qué sitios abrir, sino dónde poner los hubs.
    """
    if level not in LEVELS:
        raise ValueError(f"Nivel desconocido {level!r}. Usa uno de {list(LEVELS)}.")

    clients = np.ascontiguousarray(np.asarray(record["clients"], dtype=np.float64))
    sites = np.ascontiguousarray(np.asarray(record["hubs"], dtype=np.float64))
    q = np.asarray(record["q"], dtype=np.float64)
    w = np.asarray(record["w"], dtype=np.float64)
    site_cost = np.asarray(record["f"], dtype=np.float64)

    n = clients.shape[0]
    m = sites.shape[0]
    p = int(record["Pmax"])          # nº de hubs a ubicar; 2p = m en los seis tiers

    inst = ContInstance()
    inst.id = record.get("id", "?")
    inst.level = level
    inst.n, inst.m, inst.p = n, m, p
    inst.q, inst.w = q, w
    inst.clients, inst.sites = clients, sites
    inst.site_cost = site_cost

    # Los clientes de mayor rho = w*q se colocan primero, como en el MLPAP.
    inst.order = np.argsort(-(w * q)).astype(np.int64)

    # Radio peatonal escalado por el nivel de restricción.
    inst.Dmax_base = float(record["Dmax"])
    inst.Dmax = inst.Dmax_base * LEVELS[level]

    # Capacidad uniforme por hub, con holgura sobre la demanda total.
    total_demand = float(q.sum())
    inst.hub_capacity = total_demand * CAPACITY_SLACK / p
    # Utilización mínima por hub, proporcional a la del modelo original.
    mu_frac = float(np.mean(np.asarray(record["mu"], dtype=np.float64)
                            / np.asarray(record["L"], dtype=np.float64)))
    inst.mu_hub = inst.hub_capacity * mu_frac

    # Caja de búsqueda: envolvente de los clientes, cuadrada para que ambas
    # coordenadas compartan rango (el runner espera lower/upper escalares).
    lo = float(min(clients.min(), sites.min()))
    hi = float(max(clients.max(), sites.max()))
    inst.lower, inst.upper = lo, hi

    unit = float(np.mean(w)) * inst.Dmax_base
    inst.unit_out = OUTSOURCING_RATE * unit
    inst.unit_idle = IDLE_RATE * unit
    return inst


def load_tier(path, level="medium"):
    with open(path) as fh:
        doc = json.load(fh)
    return [prepare(r, level) for r in doc["instances"]]


@njit(cache=True, fastmath=True, inline="always")
def _land_cost(hx, hy, sites, site_cost, k, power):
    """Coste de instalación en (hx,hy): interpolación por distancia inversa sobre
    los k sitios candidatos más cercanos.

    Convierte los costes fijos discretos que traen las instancias en un campo
    continuo — el suelo es más caro en unas zonas que en otras — sin introducir
    saltos que romperían la continuidad del objetivo.
    """
    m = sites.shape[0]
    # Selección parcial de los k más cercanos sin ordenar todo el vector.
    best_d = np.full(k, 1.0e300)
    best_j = np.zeros(k, dtype=np.int64)
    for j in range(m):
        dx = hx - sites[j, 0]
        dy = hy - sites[j, 1]
        dist = dx * dx + dy * dy
        if dist < best_d[k - 1]:
            pos = k - 1
            while pos > 0 and dist < best_d[pos - 1]:
                best_d[pos] = best_d[pos - 1]
                best_j[pos] = best_j[pos - 1]
                pos -= 1
            best_d[pos] = dist
            best_j[pos] = j

    num = 0.0
    den = 0.0
    for t in range(k):
        d2 = best_d[t]
        if d2 <= 1.0e-12:            # justo encima de un sitio candidato
            return site_cost[best_j[t]]
        wgt = 1.0 / (d2 ** (power * 0.5))
        num += wgt * site_cost[best_j[t]]
        den += wgt
    return num / den


@njit(cache=True, fastmath=True, parallel=True)
def _eval_pop_cont(X, clients, sites, site_cost, q, w, order,
                   Dmax, hub_capacity, mu_hub, unit_out, unit_idle,
                   idw_k, idw_power):
    """Evalúa una población X (Npop x 2p) de configuraciones de coordenadas.

    Devuelve (coste_total, coste_base, demanda_externalizada, clientes_servidos).
    Cada candidato asigna cada cliente, en orden de rho decreciente, al hub más
    cercano dentro del radio Dmax con capacidad residual; lo no asignable se
    externaliza a coste finito.
    """
    Npop = X.shape[0]
    p = X.shape[1] // 2
    n = q.shape[0]

    total = np.empty(Npop)
    base = np.empty(Npop)
    outsourced = np.empty(Npop)
    served = np.empty(Npop)

    for i in prange(Npop):
        hx = np.empty(p)
        hy = np.empty(p)
        for j in range(p):
            hx[j] = X[i, 2 * j]
            hy[j] = X[i, 2 * j + 1]

        residual = np.empty(p)
        loads = np.empty(p)
        for j in range(p):
            residual[j] = hub_capacity
            loads[j] = 0.0

        assign_cost = 0.0
        out_demand = 0.0
        n_served = 0.0

        for idx in range(n):
            c = order[idx]
            qc = q[c]
            cx = clients[c, 0]
            cy = clients[c, 1]
            best = -1
            best_d = 1.0e300
            for j in range(p):
                if residual[j] < qc:
                    continue
                dx = cx - hx[j]
                dy = cy - hy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= Dmax and dist < best_d:
                    best_d = dist
                    best = j
            if best >= 0:
                residual[best] -= qc
                loads[best] += qc
                assign_cost += w[c] * best_d
                n_served += 1.0
            else:
                out_demand += qc

        facility = 0.0
        idle_cost = 0.0
        for j in range(p):
            facility += _land_cost(hx[j], hy[j], sites, site_cost, idw_k, idw_power)
            if loads[j] < mu_hub:
                idle_cost += unit_idle * (mu_hub - loads[j])

        f_base = facility + assign_cost
        total[i] = f_base + idle_cost + unit_out * out_demand
        base[i] = f_base
        outsourced[i] = out_demand
        served[i] = n_served
    return total, base, outsourced, served


def eval_pop(inst, X):
    """X: (Npop, 2p) float. Devuelve (total, base, demanda_externalizada, servidos)."""
    return _eval_pop_cont(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)),
        inst.clients, inst.sites, inst.site_cost, inst.q, inst.w, inst.order,
        inst.Dmax, inst.hub_capacity, inst.mu_hub,
        inst.unit_out, inst.unit_idle, IDW_K, IDW_POWER,
    )


def eval_one(inst, x):
    """Evaluación detallada de un candidato, para inspección y diagnóstico."""
    total, base, out, served = eval_pop(inst, np.asarray(x)[None, :])
    return {
        "f": float(total[0]),
        "f_base": float(base[0]),
        "outsourced_demand": float(out[0]),
        "outsourced_frac": float(out[0]) / float(inst.q.sum()),
        "served": int(served[0]),
        "served_frac": float(served[0]) / inst.n,
    }
