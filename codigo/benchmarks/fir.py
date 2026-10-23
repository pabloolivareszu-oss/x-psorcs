"""Benchmark de diseño de filtros FIR (8 casos FIR1..FIR8).

A diferencia de las suites CEC/BBOB, este es un problema de diseño en el dominio
de la frecuencia: optimizar los coeficientes h de un filtro FIR de orden 30 para
ajustar su respuesta espectral a una especificación (paso-bajo, paso-alto,
paso-banda o rechaza-banda).

Definición (PDF "FIR", Rodrigo Olivares, 2026):
  - Orden del filtro fijo a 30  ->  dimensión D = N_order + 1 = 31.
  - Vector de decisión: h = [h0, ..., h30] in R^31, con caja -1 <= hj <= 1.
  - Respuesta evaluada con N_fft = 2048 sobre el semiespectro no redundante.
  - Objetivo escalar:  F(h) = Wp * MSE_p + Ws * MSE_s
        MSE_p = media en banda de paso de (|H| - 1)^2     (desviación de ganancia 1)
        MSE_s = media en banda de rechazo de |H|^2         (fuga espectral)
  - Pesos: Wp = 1 siempre; Ws = 100 (casos Easy/Wide) o 120 (Hard/Narrow).

Como en BBOB, el óptimo F* es DESCONOCIDO a priori (f_global=None); el análisis
se hace sobre el fitness crudo (best_fitness), no sobre error al óptimo.

Interfaz uniforme con el resto de familias:
    get_fir_problem(name, dim) -> (objective, lower, upper, info)
con name in {"FIR1".."FIR8"} (o "1".."8") y dim ignorada salvo validación (D=31).
"""

from __future__ import annotations

import numpy as np

N_ORDER = 30
DIM = N_ORDER + 1          # 31
N_FFT = 2048
LOWER = -1.0
UPPER = 1.0

# Dimensión soportada: solo D=31 (orden 30 fijo, por definición del benchmark).
SUPPORTED_DIMS = (DIM,)


class FIRFilterObjective:
    """Función objetivo de un caso FIR. Precalcula los índices de banda una vez.

    case_type in {'LPF','HPF','BPF','BSF'}. Los edges f1..f4 se interpretan según
    el tipo (ver _build_band_indices). Devuelve un float al evaluar un vector de
    coeficientes de longitud DIM.
    """

    def __init__(self, case_type: str, n_order: int = N_ORDER, n_fft: int = N_FFT,
                 f1=None, f2=None, f3=None, f4=None, Wp: float = 1.0, Ws: float = 100.0):
        self.case_type = case_type
        self.n_order = n_order
        self.dim = n_order + 1
        self.n_fft = n_fft
        self.Wp = Wp
        self.Ws = Ws

        # Rejilla de frecuencias normalizadas sobre el semiespectro [0, 1].
        # Equivale a w_norm = (0:(N_fft/2)) / (N_fft/2).
        w_norm = np.linspace(0.0, 1.0, n_fft // 2 + 1)

        if case_type == "LPF":      # paso-bajo:  passband <= f1, stopband >= f2
            self.passband_idx = np.where(w_norm <= f1)[0]
            self.stopband_idx = np.where(w_norm >= f2)[0]
        elif case_type == "HPF":    # paso-alto:  stopband <= f3, passband >= f4
            self.stopband_idx = np.where(w_norm <= f3)[0]
            self.passband_idx = np.where(w_norm >= f4)[0]
        elif case_type == "BPF":    # paso-banda
            self.stopband_idx = np.where((w_norm <= f1) | (w_norm >= f4))[0]
            self.passband_idx = np.where((w_norm >= f2) & (w_norm <= f3))[0]
        elif case_type == "BSF":    # rechaza-banda
            self.passband_idx = np.where((w_norm <= f1) | (w_norm >= f4))[0]
            self.stopband_idx = np.where((w_norm >= f2) & (w_norm <= f3))[0]
        else:
            raise ValueError(f"Tipo de filtro no soportado: {case_type}")

        # Inversos de longitud (para el MSE), protegidos contra división por cero.
        self.inv_Lp = 1.0 / max(1, len(self.passband_idx))
        self.inv_Ls = 1.0 / max(1, len(self.stopband_idx))

    def evaluate(self, coeffs: np.ndarray) -> float:
        """Evalúa un único individuo (array 1D de coeficientes)."""
        H_fft = np.fft.fft(coeffs, n=self.n_fft)
        H_half = H_fft[: self.n_fft // 2 + 1]

        if len(self.stopband_idx) > 0:
            H_stop = H_half[self.stopband_idx]
            MSEs = self.inv_Ls * np.sum(np.real(H_stop) ** 2 + np.imag(H_stop) ** 2)
        else:
            MSEs = 0.0

        if len(self.passband_idx) > 0:
            H_pass_mag = np.abs(H_half[self.passband_idx])
            MSEp = self.inv_Lp * np.sum((H_pass_mag - 1.0) ** 2)
        else:
            MSEp = 0.0

        return float(self.Wp * MSEp + self.Ws * MSEs)


# --- Definición de los 8 casos FIR (PDF, sección 1) --------------------------
#   FIR1 LPF Easy  | FIR2 LPF Hard  | FIR3 HPF Easy  | FIR4 HPF Hard
#   FIR5 BPF Wide  | FIR6 BPF Narrow| FIR7 BSF Wide  | FIR8 BSF Narrow
# Ws = 100 para Easy/Wide; Ws = 120 para Hard/Narrow.
_FIR_CASES = {
    "FIR1": dict(case_type="LPF", f1=0.18, f2=0.25, Wp=1.0, Ws=100.0),
    "FIR2": dict(case_type="LPF", f1=0.22, f2=0.26, Wp=1.0, Ws=120.0),
    "FIR3": dict(case_type="HPF", f3=0.75, f4=0.82, Wp=1.0, Ws=100.0),
    "FIR4": dict(case_type="HPF", f3=0.74, f4=0.78, Wp=1.0, Ws=120.0),
    "FIR5": dict(case_type="BPF", f1=0.25, f2=0.35, f3=0.65, f4=0.75, Wp=1.0, Ws=100.0),
    "FIR6": dict(case_type="BPF", f1=0.32, f2=0.40, f3=0.60, f4=0.68, Wp=1.0, Ws=120.0),
    "FIR7": dict(case_type="BSF", f1=0.25, f2=0.35, f3=0.65, f4=0.75, Wp=1.0, Ws=100.0),
    "FIR8": dict(case_type="BSF", f1=0.32, f2=0.40, f3=0.60, f4=0.68, Wp=1.0, Ws=120.0),
}


def _canonical(name: str) -> str:
    """Normaliza 'fir3' / '3' / 'F3' -> 'FIR3'."""
    s = str(name).upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise KeyError(f"Nombre FIR inválido: {name!r} (esperado 'FIR1'..'FIR8').")
    return f"FIR{int(digits)}"


def get_fir_problem(name: str, dim: int = DIM):
    """Devuelve (objective, lower, upper, info) para un caso FIR.

    `dim` se valida contra D=31 (la dimensión es intrínseca al benchmark, no libre).
    """
    canonical = _canonical(name)
    if canonical not in _FIR_CASES:
        raise KeyError(f"Caso FIR desconocido: {name}. Disponibles: {list(_FIR_CASES)}")
    if dim != DIM:
        raise ValueError(
            f"FIR tiene dimensión fija D={DIM} (orden {N_ORDER}); pedida dim={dim}."
        )

    spec = _FIR_CASES[canonical]
    problem = FIRFilterObjective(**spec)

    def objective(x: np.ndarray) -> float:
        return problem.evaluate(np.asarray(x, dtype=float))

    info = {
        "family": "FIR",
        "name": canonical,
        "case_type": spec["case_type"],
        "dim": DIM,
        "lower": LOWER,
        "upper": UPPER,
        "Wp": spec["Wp"],
        "Ws": spec["Ws"],
        "f_global": None,   # óptimo desconocido a priori (como BBOB)
    }
    return objective, LOWER, UPPER, info


def list_fir_functions() -> list[str]:
    return list(_FIR_CASES.keys())
