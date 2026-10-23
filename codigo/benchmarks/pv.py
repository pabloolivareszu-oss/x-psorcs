"""Benchmark de extracción de parámetros de modelos fotovoltaicos (6 casos PV1..PV6).

Problema de identificación: dado un conjunto de pares (V, I) MEDIDOS sobre una
celda o módulo real, hallar los parámetros del circuito equivalente de diodo que
reproducen esa curva. Es el problema aplicado estándar de la literatura de
metaheurísticas en energía solar, y a diferencia de FIR o antenna no se evalúa
contra una especificación de diseño sino contra datos experimentales.

Modelos (todos con la ecuación implícita del circuito equivalente):

  Diodo simple (SDM), 5 parámetros [Iph, Isd, Rs, Rsh, n]:
      I = Iph - Isd*(exp(q(V + I*Rs)/(n*k*T)) - 1) - (V + I*Rs)/Rsh

  Diodo doble (DDM), 7 parámetros [Iph, Isd1, Isd2, Rs, Rsh, n1, n2]:
      añade un segundo término de difusión/recombinación con su propio ideality.

  Diodo triple (TDM), 9 parámetros: un tercer término análogo.

Función objetivo: RMSE del residuo de la ecuación implícita sobre los puntos
medidos, que es la formulación estándar en la literatura y evita tener que
despejar I (lo que exigiría la función W de Lambert):

      f_k = Iph - Isd*(exp(...) - 1) - (V_k + I_k*Rs)/Rsh - I_k
      RMSE = sqrt( mean_k f_k^2 )

Con los parámetros correctos el residuo tiende a cero, así que el óptimo es
conocido en su valor (0) pero no alcanzable exactamente por el ruido de medida;
los mejores valores publicados rondan 1e-3 A. Se declara f_global=None y el
análisis usa el fitness crudo, como en BBOB/FIR/antenna.

DATOS EXPERIMENTALES. Las dos curvas son las de referencia de este campo y NO
fueron transcritas de memoria: se descargaron de los datos suplementarios en
acceso abierto de

  E. I. Batzelis et al., "Simple and efficient estimation of photovoltaic cells
  and modules parameters using approximation and correction technique",
  PLOS ONE, vol. 14, no. 6, art. e0216201, 2019,
  doi: 10.1371/journal.pone.0216201     (S1 Dataset y S3 Dataset)

que a su vez las toma de la fuente original del benchmark:

  T. Easwarakhanthan, J. Bottin, I. Bouhouch y C. Boutrit, "Nonlinear
  minimization algorithm for determining the solar cell parameters with
  microcomputers", International Journal of Solar Energy, vol. 4, no. 1, 1986.

  - R.T.C. France: celda de silicio de 57 mm, 26 puntos, 33 °C, 1000 W/m².
  - Photowatt-PWP201: módulo de 36 celdas en serie, 26 puntos, 45 °C.

Interfaz uniforme con el resto de familias:
    get_pv_problem(name, dim) -> (objective, lower, upper, info)
"""

from __future__ import annotations

import numpy as np

# Constantes físicas.
Q = 1.60217646e-19      # carga del electrón [C]
K_B = 1.3806503e-23     # constante de Boltzmann [J/K]

# --- Datos experimentales -------------------------------------------------
# R.T.C. France, 33 °C, 1000 W/m². 26 pares (V, I). PLOS ONE S1 Dataset.
_RTC_V = np.array([
    -0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132,
    0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590,
    0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
    0.5833, 0.5900,
])
_RTC_I = np.array([
    0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570,
    0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755,
    0.6320, 0.5730, 0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
    -0.1230, -0.2100,
])

# Photowatt-PWP201, 45 °C. 26 pares (V, I). PLOS ONE S3 Dataset.
_PWP_V = np.array([
    -1.9426, 0.1248, 1.8093, 3.3511, 4.7622, 6.0538, 7.2364, 8.3189,
    9.3097, 10.2163, 11.0449, 11.8018, 12.4929, 13.1231, 13.6983, 14.2221,
    14.6995, 15.1346, 15.5311, 15.8929, 16.2229, 16.5241, 16.7987, 17.0499,
    17.2793, 17.4885,
])
_PWP_I = np.array([
    1.0345, 1.0315, 1.0300, 1.0260, 1.0220, 1.0180, 1.0155, 1.0140,
    1.0100, 1.0035, 0.9880, 0.9630, 0.9255, 0.8725, 0.8075, 0.7265,
    0.6345, 0.5345, 0.4275, 0.3185, 0.2085, 0.1010, -0.0080, -0.1110,
    -0.2090, -0.3030,
])

_DATASETS = {
    "rtc": dict(V=_RTC_V, I=_RTC_I, temp_c=33.0, n_series=1,
                label="R.T.C. France (celda Si, 57 mm)"),
    "pwp": dict(V=_PWP_V, I=_PWP_I, temp_c=45.0, n_series=36,
                label="Photowatt-PWP201 (módulo, 36 celdas en serie)"),
}

# --- Cotas de los parámetros (convención de la literatura) ----------------
# [Iph (A), Isd (uA), Rs (ohm), Rsh (ohm), n]
_BOUNDS_CELL = {
    "Iph": (0.0, 1.0), "Isd": (0.0, 1.0e-6), "Rs": (0.0, 0.5),
    "Rsh": (0.0, 100.0), "n": (1.0, 2.0),
}
# Para el módulo las escalas de Rs y Rsh son mayores (36 celdas en serie).
_BOUNDS_MODULE = {
    "Iph": (0.0, 2.0), "Isd": (0.0, 50.0e-6), "Rs": (0.0, 2.0),
    "Rsh": (0.0, 2000.0), "n": (1.0, 50.0),
}


class PVObjective:
    """RMSE del residuo de la ecuación implícita sobre los puntos medidos.

    dataset  : 'rtc' | 'pwp'
    n_diodes : 1 (SDM, 5 params) | 2 (DDM, 7) | 3 (TDM, 9)

    Orden del vector de decisión:
        SDM: [Iph, Isd, Rs, Rsh, n]
        DDM: [Iph, Isd1, Isd2, Rs, Rsh, n1, n2]
        TDM: [Iph, Isd1, Isd2, Isd3, Rs, Rsh, n1, n2, n3]

    Las variables se optimizan NORMALIZADAS en [0, 1] y se desescalan a sus cotas
    físicas dentro de la evaluación, para que todas las familias compartan una
    caja de búsqueda escalar como espera el runner.
    """

    def __init__(self, dataset: str, n_diodes: int):
        if dataset not in _DATASETS:
            raise ValueError(f"Conjunto desconocido: {dataset}. Usa 'rtc' o 'pwp'.")
        if n_diodes not in (1, 2, 3):
            raise ValueError(f"n_diodes debe ser 1, 2 o 3; recibido {n_diodes}.")
        d = _DATASETS[dataset]
        self.dataset = dataset
        self.V = d["V"]
        self.I = d["I"]
        self.n_series = d["n_series"]
        self.n_diodes = n_diodes
        self.dim = 3 + 2 * n_diodes          # 5, 7 o 9

        # Voltaje térmico: Vt = n_series * k * T / q  (T en kelvin).
        self.Vt = self.n_series * K_B * (d["temp_c"] + 273.15) / Q

        b = _BOUNDS_MODULE if self.n_series > 1 else _BOUNDS_CELL
        lows = [b["Iph"][0]] + [b["Isd"][0]] * n_diodes + [b["Rs"][0], b["Rsh"][0]] \
            + [b["n"][0]] * n_diodes
        highs = [b["Iph"][1]] + [b["Isd"][1]] * n_diodes + [b["Rs"][1], b["Rsh"][1]] \
            + [b["n"][1]] * n_diodes
        self.low = np.asarray(lows, dtype=float)
        self.high = np.asarray(highs, dtype=float)

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """[0,1]^D -> parámetros físicos."""
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        return self.low + z * (self.high - self.low)

    def evaluate(self, z: np.ndarray) -> float:
        p = self.denormalize(z)
        nd = self.n_diodes
        Iph = p[0]
        Isd = p[1:1 + nd]
        Rs = p[1 + nd]
        Rsh = p[2 + nd]
        n = p[3 + nd:3 + 2 * nd]

        if Rsh <= 1e-12:
            return 1.0e6

        Vd = self.V + self.I * Rs                 # tensión sobre el diodo
        diode = 0.0
        for j in range(nd):
            arg = np.clip(Vd / (n[j] * self.Vt), -700.0, 700.0)
            diode = diode + Isd[j] * (np.exp(arg) - 1.0)

        resid = Iph - diode - Vd / Rsh - self.I
        if not np.all(np.isfinite(resid)):
            return 1.0e6
        return float(np.sqrt(np.mean(resid ** 2)))


# PV1-PV3: celda R.T.C. France, 1/2/3 diodos.  PV4-PV6: módulo Photowatt.
_PV_CASES = {
    "PV1": dict(dataset="rtc", n_diodes=1),
    "PV2": dict(dataset="rtc", n_diodes=2),
    "PV3": dict(dataset="rtc", n_diodes=3),
    "PV4": dict(dataset="pwp", n_diodes=1),
    "PV5": dict(dataset="pwp", n_diodes=2),
    "PV6": dict(dataset="pwp", n_diodes=3),
}

# Búsqueda normalizada: todas las variables viven en [0, 1].
LOWER = 0.0
UPPER = 1.0
SUPPORTED_DIMS = (5, 7, 9)


def _canonical(name: str) -> str:
    """Normaliza 'pv3' / '3' / 'P3' -> 'PV3'."""
    s = str(name).upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise KeyError(f"Nombre de caso inválido: {name!r} (esperado 'PV1'..'PV6').")
    return f"PV{int(digits)}"


def case_dim(name: str) -> int:
    """Dimensión (nº de parámetros) de un caso concreto."""
    canonical = _canonical(name)
    if canonical not in _PV_CASES:
        raise KeyError(f"Caso desconocido: {name}. Disponibles: {list(_PV_CASES)}")
    return 3 + 2 * _PV_CASES[canonical]["n_diodes"]


def get_pv_problem(name: str, dim: int | None = None):
    """Devuelve (objective, lower, upper, info) para un caso PV.

    La objective recibe un vector normalizado en [0,1]^D y devuelve el RMSE en
    amperios. `dim` se valida contra la dimensión intrínseca del caso.
    """
    canonical = _canonical(name)
    if canonical not in _PV_CASES:
        raise KeyError(f"Caso desconocido: {name}. Disponibles: {list(_PV_CASES)}")

    spec = _PV_CASES[canonical]
    problem = PVObjective(**spec)
    if dim is not None and dim != problem.dim:
        raise ValueError(
            f"El caso {canonical} usa {spec['n_diodes']} diodo(s), así que su "
            f"dimensión es {problem.dim}; pedida dim={dim}."
        )

    def objective(x: np.ndarray) -> float:
        return problem.evaluate(np.asarray(x, dtype=float))

    info = {
        "family": "pv",
        "name": canonical,
        "dataset": spec["dataset"],
        "dataset_label": _DATASETS[spec["dataset"]]["label"],
        "n_diodes": spec["n_diodes"],
        "n_points": len(problem.V),
        "dim": problem.dim,
        "lower": LOWER,
        "upper": UPPER,
        "f_global": None,   # el residuo ideal es 0, pero no alcanzable por ruido
    }
    return objective, LOWER, UPPER, info


def list_pv_functions() -> list[str]:
    return list(_PV_CASES.keys())
