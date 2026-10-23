"""Benchmark de síntesis de arreglos lineales de antenas (8 casos ANT1..ANT8).

Problema de diseño en el dominio angular, hermano estructural del benchmark FIR:
en FIR se moldea la respuesta en frecuencia de un filtro ajustando sus coeficientes;
aquí se moldea el patrón de radiación de un arreglo lineal ajustando las amplitudes
de excitación de sus elementos. La relación no es analógica sino matemática: el
factor de arreglo es la transformada de Fourier de las excitaciones evaluada en
u = cos(theta).

Definición:
  - Arreglo lineal simétrico de N elementos, espaciado uniforme d = lambda/2,
    centrado en el origen. Por simetría par, las excitaciones quedan determinadas
    por las N/2 amplitudes de una mitad  ->  dimensión D = N/2.
  - Vector de decisión: a = [a_1, ..., a_{N/2}] in [0, 1]^{N/2}.
  - Factor de arreglo del arreglo simétrico, con u = cos(theta):

        AF(u) = 2 * sum_{n=1..N/2} a_n * cos( (2n-1) * (pi/2) * u )

    (el factor (2n-1)/2 * (2*pi*d/lambda) con d = lambda/2 se reduce a
     (2n-1)*(pi/2), que es el desplazamiento del n-ésimo par de elementos.)
  - El patrón se normaliza por su máximo, de modo que el objetivo no depende de
    la escala global de las excitaciones.
  - Objetivo escalar, misma forma que FIR (F = Wp*MSE_p + Ws*MSE_s):

        MSE_p = media en el lóbulo principal de (|AF_norm| - 1)^2
        MSE_s = media en la región de lóbulos laterales de |AF_norm|^2
        MSE_n = media en los nulos impuestos de |AF_norm|^2   (casos con nulos)

    Minimizar MSE_s suprime los lóbulos laterales; MSE_p mantiene el lóbulo
    principal formado y evita la solución degenerada de apagar el arreglo.

Ocho casos que cruzan dos tamaños de arreglo con cuatro especificaciones:
  ANT1/ANT5  supresión de lóbulos laterales, haz ancho
  ANT2/ANT6  supresión de lóbulos laterales, haz estrecho (más exigente)
  ANT3/ANT7  supresión con nulos impuestos en direcciones de interferencia
  ANT4/ANT8  supresión con nulos y haz estrecho
ANT1..ANT4 usan N = 32 (D = 16); ANT5..ANT8 usan N = 64 (D = 32), comparable a
los 31 coeficientes de FIR.

Como en BBOB/FIR, el óptimo F* es DESCONOCIDO a priori (f_global=None); el
análisis se hace sobre el fitness crudo.

Referencia del problema:
  A. Sharma, "Antenna array pattern synthesis using metaheuristic algorithms:
  A review", IETE Technical Review, vol. 40, no. 1, pp. 90-115, 2022,
  doi: 10.1080/02564602.2022.2051616.

Interfaz uniforme con el resto de familias:
    get_antenna_problem(name, dim) -> (objective, lower, upper, info)
"""

from __future__ import annotations

import numpy as np

N_GRID = 2048              # muestras del patrón sobre u = cos(theta) in [-1, 1]
LOWER = 0.0                # amplitudes de excitación no negativas
UPPER = 1.0

# Zona de transición a cada lado del lóbulo principal, como fracción de su
# semiancho: no se penaliza allí, igual que la banda de transición de un filtro.
# Proporcional y no absoluta, porque el ancho de haz varía un factor 2 entre los
# arreglos de 32 y de 64 elementos.
_TRANSITION_FRAC = 0.3


class ArrayFactorObjective:
    """Función objetivo de un caso de arreglo. Precalcula rejilla y máscaras.

    n_elements : nº total de elementos del arreglo (par).
    main_lobe  : semiancho del lóbulo principal en u = cos(theta), centrado en
                 broadside (u = 0). Más pequeño = haz más estrecho = más difícil.
    nulls      : direcciones u donde se exige nulo, o None.
    """

    def __init__(self, n_elements: int, main_lobe: float,
                 nulls: tuple[float, ...] | None = None,
                 null_width: float = 0.02,
                 n_grid: int = N_GRID, Wp: float = 1.0, Ws: float = 100.0,
                 Wn: float = 100.0):
        if n_elements % 2 != 0:
            raise ValueError(f"El arreglo debe tener un nº par de elementos: {n_elements}")
        self.n_elements = n_elements
        self.dim = n_elements // 2
        self.main_lobe = main_lobe
        self.nulls = tuple(nulls) if nulls else ()
        self.Wp, self.Ws, self.Wn = Wp, Ws, Wn

        # Rejilla en el espacio u = cos(theta), que es donde el factor de arreglo
        # es una suma de cosenos (el equivalente del eje de frecuencia en FIR).
        u = np.linspace(-1.0, 1.0, n_grid)
        self.u = u

        # Matriz de la base: fila = punto de la rejilla, columna = par de elementos.
        # AF(u) = 2 * B @ a, con B[i, n] = cos((2n+1) * (pi/2) * u_i).
        n_idx = np.arange(self.dim)
        self.basis = 2.0 * np.cos(np.outer(u, (2 * n_idx + 1) * (np.pi / 2.0)))

        # Máscaras: lóbulo principal, lóbulos laterales y nulos.
        self.main_idx = np.where(np.abs(u) <= main_lobe)[0]
        side_mask = np.abs(u) >= main_lobe * (1.0 + _TRANSITION_FRAC)
        for u0 in self.nulls:
            side_mask &= np.abs(u - u0) > null_width
        self.side_idx = np.where(side_mask)[0]

        null_mask = np.zeros_like(u, dtype=bool)
        for u0 in self.nulls:
            null_mask |= np.abs(u - u0) <= null_width
        self.null_idx = np.where(null_mask)[0]

        self.inv_Lp = 1.0 / max(1, len(self.main_idx))
        self.inv_Ls = 1.0 / max(1, len(self.side_idx))
        self.inv_Ln = 1.0 / max(1, len(self.null_idx))

    def pattern(self, amplitudes: np.ndarray) -> np.ndarray:
        """Patrón normalizado |AF(u)| / max|AF(u)| sobre la rejilla."""
        af = self.basis @ np.asarray(amplitudes, dtype=float)
        peak = np.max(np.abs(af))
        if peak <= 1e-12:      # arreglo apagado: patrón nulo, penalizado por MSE_p
            return np.zeros_like(af)
        return np.abs(af) / peak

    def evaluate(self, amplitudes: np.ndarray) -> float:
        """Evalúa un único individuo (vector 1D de amplitudes)."""
        mag = self.pattern(amplitudes)

        MSEp = self.inv_Lp * np.sum((mag[self.main_idx] - 1.0) ** 2) if len(self.main_idx) else 0.0
        MSEs = self.inv_Ls * np.sum(mag[self.side_idx] ** 2) if len(self.side_idx) else 0.0
        MSEn = self.inv_Ln * np.sum(mag[self.null_idx] ** 2) if len(self.null_idx) else 0.0

        return float(self.Wp * MSEp + self.Ws * MSEs + self.Wn * MSEn)

    def peak_sidelobe_db(self, amplitudes: np.ndarray) -> float:
        """Nivel de lóbulo lateral máximo en dB. Métrica de ingeniería, para
        informar el resultado en las unidades en que se reporta en la literatura;
        no interviene en la optimización."""
        mag = self.pattern(amplitudes)
        if not len(self.side_idx):
            return float("-inf")
        peak = np.max(mag[self.side_idx])
        return float(20.0 * np.log10(max(peak, 1e-12)))


# ANT1..ANT4: N = 32 (D = 16).   ANT5..ANT8: N = 64 (D = 32).
#
# El semiancho del lóbulo principal se fija respecto del ancho de haz NATURAL del
# arreglo uniforme, cuyo primer nulo está en u = 2/N (0,0625 para N=32; 0,03125
# para N=64). Un main_lobe igual a ese valor pide el haz que el arreglo da de
# forma natural; uno menor exige estrecharlo, lo que obliga a sacrificar lóbulos
# laterales y hace el caso más difícil. Una máscara MÁS ANCHA que el primer nulo
# se tragaría el primer lóbulo lateral y el caso dejaría de medir lo que dice.
# Ws = 100 en los casos naturales, 120 en los estrechos (convención de FIR).
_ANTENNA_CASES = {
    "ANT1": dict(n_elements=32, main_lobe=0.0625, Wp=1.0, Ws=100.0),
    "ANT2": dict(n_elements=32, main_lobe=0.045, Wp=1.0, Ws=120.0),
    "ANT3": dict(n_elements=32, main_lobe=0.0625, nulls=(-0.45, 0.45), Wp=1.0, Ws=100.0),
    "ANT4": dict(n_elements=32, main_lobe=0.045, nulls=(-0.30, 0.30, 0.60), Wp=1.0, Ws=120.0),
    "ANT5": dict(n_elements=64, main_lobe=0.03125, Wp=1.0, Ws=100.0),
    "ANT6": dict(n_elements=64, main_lobe=0.022, Wp=1.0, Ws=120.0),
    "ANT7": dict(n_elements=64, main_lobe=0.03125, nulls=(-0.45, 0.45), Wp=1.0, Ws=100.0),
    "ANT8": dict(n_elements=64, main_lobe=0.022, nulls=(-0.30, 0.30, 0.60), Wp=1.0, Ws=120.0),
}

# Dimensiones soportadas: 16 (ANT1-4) y 32 (ANT5-8). La dim es intrínseca al caso.
SUPPORTED_DIMS = (16, 32)


def _canonical(name: str) -> str:
    """Normaliza 'ant3' / '3' / 'A3' -> 'ANT3'."""
    s = str(name).upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise KeyError(f"Nombre de caso inválido: {name!r} (esperado 'ANT1'..'ANT8').")
    return f"ANT{int(digits)}"


def case_dim(name: str) -> int:
    """Dimensión (N/2 amplitudes) de un caso concreto."""
    canonical = _canonical(name)
    if canonical not in _ANTENNA_CASES:
        raise KeyError(f"Caso desconocido: {name}. Disponibles: {list(_ANTENNA_CASES)}")
    return _ANTENNA_CASES[canonical]["n_elements"] // 2


def get_antenna_problem(name: str, dim: int | None = None):
    """Devuelve (objective, lower, upper, info) para un caso de arreglo.

    `dim` se valida contra la dimensión intrínseca del caso (N/2). Si se pasa None
    se usa directamente la del caso.
    """
    canonical = _canonical(name)
    if canonical not in _ANTENNA_CASES:
        raise KeyError(f"Caso desconocido: {name}. Disponibles: {list(_ANTENNA_CASES)}")

    spec = _ANTENNA_CASES[canonical]
    problem = ArrayFactorObjective(**spec)
    if dim is not None and dim != problem.dim:
        raise ValueError(
            f"El caso {canonical} tiene {spec['n_elements']} elementos, así que su "
            f"dimensión es N/2={problem.dim}; pedida dim={dim}."
        )

    def objective(x: np.ndarray) -> float:
        return problem.evaluate(np.asarray(x, dtype=float))

    info = {
        "family": "antenna",
        "name": canonical,
        "n_elements": spec["n_elements"],
        "main_lobe": spec["main_lobe"],
        "nulls": list(spec.get("nulls", ())),
        "dim": problem.dim,
        "lower": LOWER,
        "upper": UPPER,
        "Wp": spec["Wp"],
        "Ws": spec["Ws"],
        "f_global": None,   # óptimo desconocido a priori (como BBOB/FIR)
    }
    return objective, LOWER, UPPER, info


def list_antenna_functions() -> list[str]:
    return list(_ANTENNA_CASES.keys())
