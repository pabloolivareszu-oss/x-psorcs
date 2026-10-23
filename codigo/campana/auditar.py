"""Auditoría de la campaña v3: consolida resultados/auditoria.json.

Lee lo que la campaña dejó registrado —run_status.json, cada summary.json, run.log
y errors.log— y comprueba, sin fiarse de ningún resumen intermedio:

  - celdas por familia: completas, fallidas y corridas por celda frente a las
    esperadas (30 en las familias originales, 31 en rail);
  - presupuesto en CADA corrida: 3.000 evaluaciones (20.000 en rail), con las
    excepciones listadas por algoritmo;
  - salud de SIA-online: corridas en las que el modelo nunca se entrenó o el
    controlador nunca adaptó (el síntoma de meseta de microhubs);
  - tiempos: CPU por familia y algoritmo, y duración real de cada fase;
  - ETA: lo que el log predijo al 10 %, 25 % y 50 % de cada fase frente a cuándo
    terminó de verdad;
  - memoria: el registro periódico de run.py (disponible, campaña, swap) y cada
    actuación de la guardia de memoria, con el estado del sistema en ese momento;
  - reanudaciones: celdas que retomaron corridas ya completas tras una interrupción.

Lo invoca `campana/lanzar.sh auditoria`.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import RESULTS  # noqa: E402

ESPERADO = {"rail": {"corridas": 31, "evaluaciones": 20000}}
ESPERADO_DEFECTO = {"corridas": 30, "evaluaciones": 3000}

_CELDA = re.compile(
    r"^(\d\d:\d\d:\d\d) \[(\d+)/(\d+)\s+([\d.]+)%\] \(celda (\d+)/(\d+)\) (OK|FALLO)\s+"
    r"(\S+)\s.*?([\d.]+)s\s+ETA (\S+)")
_INICIO = re.compile(r"^INICIO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_FIN = re.compile(r"^FIN (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_FAMILIAS = re.compile(r"^Familias: \[(.*)\]")
_MEMORIA = re.compile(r"^(\d\d:\d\d:\d\d) \[memoria\] disponible ([\d.]+) GB, campaña ([\d.]+) GB, "
                      r"swap ([\d.]+) GB")


def _esperado(familia: str) -> dict:
    return ESPERADO.get(familia, ESPERADO_DEFECTO)


def _eta_seg(txt: str) -> int | None:
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", txt)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def auditar_corridas() -> dict:
    """Recorre todos los summary.json: presupuesto y salud de SIA-online."""
    por_familia: dict = collections.defaultdict(lambda: {
        "corridas": 0, "evaluaciones_fuera_de_presupuesto": collections.Counter(),
        "ejemplos_fuera_de_presupuesto": {}, "sia_online_sin_modelo": 0,
        "sia_online_sin_adaptaciones": 0, "sia_online_corridas": 0})
    for p in RESULTS.glob("*/*/*/*/dim_*/run_*/summary.json"):
        d = json.loads(p.read_text())
        fam = d.get("family", p.parts[-5]).lower()
        rec = por_familia[fam]
        rec["corridas"] += 1
        esperado = _esperado(fam)["evaluaciones"]
        fes = d.get("function_evaluations")
        if fes != esperado:
            algo = d.get("algorithm")
            rec["evaluaciones_fuera_de_presupuesto"][algo] += 1
            rec["ejemplos_fuera_de_presupuesto"].setdefault(algo, fes)
        if d.get("algorithm") == "SIA-online":
            rec["sia_online_corridas"] += 1
            rec["sia_online_sin_modelo"] += int(d.get("n_model_updates", 0) == 0)
            rec["sia_online_sin_adaptaciones"] += int(d.get("n_adaptations", 0) == 0)
    out = {}
    for fam, rec in sorted(por_familia.items()):
        rec["evaluaciones_fuera_de_presupuesto"] = dict(rec["evaluaciones_fuera_de_presupuesto"])
        rec["evaluaciones_esperadas"] = _esperado(fam)["evaluaciones"]
        out[fam] = rec
    return out


def auditar_celdas() -> dict:
    """run_status.json: completas, fallidas, corridas por celda, CPU."""
    st_path = RESULTS / "run_status.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    fam: dict = collections.defaultdict(lambda: {
        "celdas": 0, "ok": 0, "fallidas": [], "corridas_incompletas": [],
        "cpu_horas": 0.0, "cpu_horas_por_algoritmo": collections.Counter()})
    for key, v in st.items():
        algo, family, func, dim = key.split("|")
        r = fam[family]
        r["celdas"] += 1
        if v.get("ok"):
            r["ok"] += 1
        else:
            r["fallidas"].append({"celda": key, "error": v.get("error")})
        if v.get("n_done") != _esperado(family)["corridas"]:
            r["corridas_incompletas"].append({"celda": key, "corridas": v.get("n_done")})
        r["cpu_horas"] += (v.get("seconds") or 0) / 3600
        if v.get("n_reused"):
            r.setdefault("celdas_reanudadas", []).append(
                {"celda": key, "corridas_reutilizadas": v["n_reused"]})
        r["cpu_horas_por_algoritmo"][algo] += (v.get("seconds") or 0) / 3600
    out = {}
    for f, r in sorted(fam.items()):
        r["cpu_horas"] = round(r["cpu_horas"], 2)
        r["cpu_horas_por_algoritmo"] = {a: round(h, 3) for a, h in
                                        r["cpu_horas_por_algoritmo"].most_common()}
        r["corridas_esperadas"] = _esperado(f)["corridas"]
        out[f] = r
    return out


def auditar_log() -> list[dict]:
    """run.log: duración de cada fase y precisión del ETA."""
    log = RESULTS / "run.log"
    if not log.exists():
        return []
    fases, actual = [], None
    for linea in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if m := _INICIO.match(linea):
            actual = {"inicio": m.group(1), "familias": None, "fin": None, "celdas": []}
            fases.append(actual)
        elif actual is None:
            continue
        elif m := _FAMILIAS.match(linea):
            actual["familias"] = [s.strip(" '") for s in m.group(1).split(",")]
        elif m := _CELDA.match(linea):
            actual["celdas"].append((m.group(1), float(m.group(4)), _eta_seg(m.group(10)),
                                     m.group(7) == "OK"))
        elif m := _FIN.match(linea):
            actual["fin"] = m.group(1)

    out = []
    for f in fases:
        ini = datetime.strptime(f["inicio"], "%Y-%m-%d %H:%M:%S")
        res = {"familias": f["familias"], "inicio": f["inicio"], "fin": f["fin"],
               "celdas_registradas": len(f["celdas"]),
               "celdas_fallidas": sum(1 for c in f["celdas"] if not c[3])}
        if f["fin"]:
            fin = datetime.strptime(f["fin"], "%Y-%m-%d %H:%M:%S")
            res["duracion_horas"] = round((fin - ini).total_seconds() / 3600, 3)
            # Hora de reloj de cada línea: HH:MM:SS, avanzando de día si cruza medianoche.
            precision, dia, previa = [], ini.date(), None
            for hhmmss, pct, eta, _ok in f["celdas"]:
                t = datetime.combine(dia, datetime.strptime(hhmmss, "%H:%M:%S").time())
                if previa and t < previa:
                    dia += timedelta(days=1)
                    t += timedelta(days=1)
                previa = t
                for hito in (10, 25, 50):
                    if pct >= hito and eta is not None and not any(
                            p["hito_pct"] == hito for p in precision):
                        predicho = t + timedelta(seconds=eta)
                        precision.append({
                            "hito_pct": hito, "hora": t.strftime("%Y-%m-%d %H:%M:%S"),
                            "eta_predicha": predicho.strftime("%Y-%m-%d %H:%M:%S"),
                            "error_minutos": round((predicho - fin).total_seconds() / 60, 1)})
            res["precision_eta"] = precision
        out.append(res)
    return out


def auditar_memoria() -> dict:
    """Registro periódico de memoria (run.log) y actuaciones de la guardia (errors.log)."""
    muestras = []
    log = RESULTS / "run.log"
    if log.exists():
        for linea in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if m := _MEMORIA.match(linea):
                muestras.append({"hora": m.group(1), "disponible_gb": float(m.group(2)),
                                 "campana_gb": float(m.group(3)), "swap_gb": float(m.group(4))})
    actuaciones = []
    err = RESULTS / "errors.log"
    if err.exists():
        for linea in err.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "guardia de memoria  ->  " in linea:
                try:
                    actuaciones.append(json.loads(linea.split("  ->  ", 1)[1]))
                except json.JSONDecodeError:
                    actuaciones.append({"linea": linea})
    return {
        "muestras": len(muestras),
        "disponible_minimo_gb": min((m["disponible_gb"] for m in muestras), default=None),
        "campana_maxima_gb": max((m["campana_gb"] for m in muestras), default=None),
        "swap_maximo_gb": max((m["swap_gb"] for m in muestras), default=None),
        "actuaciones_guardia": actuaciones,
        "serie": muestras,
    }


def main() -> None:
    cfg_path = RESULTS / "config.json"
    errores = RESULTS / "errors.log"
    informe = {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raiz_resultados": str(RESULTS),
        "config": json.loads(cfg_path.read_text()) if cfg_path.exists() else None,
        "celdas": auditar_celdas(),
        "corridas": auditar_corridas(),
        "fases": auditar_log(),
        "memoria": auditar_memoria(),
        "errores_log": (errores.read_text().splitlines() if errores.exists() else []),
    }
    out = RESULTS / "auditoria.json"
    out.write_text(json.dumps(informe, indent=2, ensure_ascii=False, default=str))

    print(f"auditoria.json en {out}\n")
    for fam, c in informe["celdas"].items():
        r = informe["corridas"].get(fam, {})
        fuera = r.get("evaluaciones_fuera_de_presupuesto", {})
        print(f"  {fam:10} celdas {c['ok']}/{c['celdas']} ok  fallidas {len(c['fallidas'])}  "
              f"incompletas {len(c['corridas_incompletas'])}  CPU {c['cpu_horas']:.1f} h  "
              f"corridas {r.get('corridas', 0)}  fuera de presupuesto {fuera or 'ninguna'}")
        if r.get("sia_online_corridas"):
            print(f"             SIA-online sin modelo {r['sia_online_sin_modelo']}/"
                  f"{r['sia_online_corridas']}, sin adaptaciones "
                  f"{r['sia_online_sin_adaptaciones']}/{r['sia_online_corridas']}")
    mem = informe["memoria"]
    print(f"  memoria: {mem['muestras']} muestras, disponible mínimo {mem['disponible_minimo_gb']} GB, "
          f"campaña máx {mem['campana_maxima_gb']} GB, swap máx {mem['swap_maximo_gb']} GB, "
          f"actuaciones de la guardia {len(mem['actuaciones_guardia'])}")
    for f in informe["fases"]:
        print(f"  fase {f['familias']}: {f.get('duracion_horas', '?')} h, "
              f"ETA {[(p['hito_pct'], p['error_minutos']) for p in f.get('precision_eta', [])]}")


if __name__ == "__main__":
    main()
