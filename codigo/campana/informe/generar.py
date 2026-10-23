"""Genera el informe de la campaña v3 a partir de sus resultados.

Ninguna cifra del informe se copia a mano: este script las lee de
resultados/ (análisis por familia, resúmenes de cada corrida, reporte del
caso ferroviario, auditoría y manifiesto), las escribe en datos.json y las inyecta
en plantilla.html para producir informe.html, que es lo que se publica.

    python campana/informe/generar.py   (desde RecursosPaper/codigo)
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import statistics as st
from pathlib import Path

AQUI = Path(__file__).resolve().parent
R = AQUI.parents[2] / "resultados"
FAMILIAS = [("classical", "classical"), ("cec2014", "CEC 2014"), ("cec2017", "CEC 2017"),
            ("cec2022", "CEC 2022"), ("bbob", "BBOB"), ("fir", "FIR")]


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def familias() -> tuple[list, list, list]:
    fams, rangos, acum, n_total = [], [], collections.Counter(), 0
    for fid, nombre in FAMILIAS:
        w = _json(R / "analysis" / fid / "wilcoxon_A_vs_Aprime.json")
        fr = _json(R / "analysis" / fid / "friedman.json")
        if not w or not fr:
            continue
        fams.append({"id": fid, "nombre": nombre, "n": w["n_problems"], "wA": w["wins_a"],
                     "wB": w["wins_b"], "emp": w["ties"],
                     "mejora": round(-100 * w["median_rel_diff"], 2),
                     "p": w["p_value"], "sig": bool(w["significant_0.05"])})
        mr = fr["mean_ranks"]
        orden = sorted(mr, key=mr.get)
        rangos.append({"nombre": nombre, "on": orden.index("SIA-online") + 1,
                       "onMr": round(mr["SIA-online"], 2), "base": orden.index("SIA-base") + 1,
                       "baseMr": round(mr["SIA-base"], 2),
                       "lideres": [(a, round(mr[a], 2)) for a in orden[:3]]})
        n = fr["n_problems"]
        n_total += n
        for a, r in mr.items():
            acum[a] += r * n
    glob_rank = sorted(((a, v / n_total) for a, v in acum.items()), key=lambda x: x[1])
    nombres = [a for a, _ in glob_rank]
    if glob_rank:
        rangos.append({"nombre": f"Global ({n_total} celdas)", "global": True,
                       "on": nombres.index("SIA-online") + 1,
                       "onMr": round(dict(glob_rank)["SIA-online"], 2),
                       "base": nombres.index("SIA-base") + 1,
                       "baseMr": round(dict(glob_rank)["SIA-base"], 2)})
    return fams, rangos, [(a, round(r, 2)) for a, r in glob_rank]


def salud() -> list:
    out = []
    for fid, nombre in FAMILIAS + [("rail", "rail")]:
        ps = glob.glob(str(R / "SIA_online/SIA-online" / fid / "*/*/run_*/summary.json"))
        if not ps:
            continue
        inten, div, mins, ad, up = [], [], [], [], []
        for p in ps:
            d = json.loads(Path(p).read_text())
            op = d["final_operator_probabilities"]
            inten.append(op["intensification"]); div.append(op["diversification"])
            mins.append(min(op.values())); ad.append(d["n_adaptations"]); up.append(d["n_model_updates"])
        out.append({"nombre": nombre, "corridas": len(ps),
                    "cero": sum(1 for x in inten if x < 1e-9), "min": round(min(mins), 3),
                    "div": round(st.mean(div), 3), "adapt": round(st.mean(ad), 1),
                    "upd": round(st.mean(up), 1),
                    "sin_modelo": sum(1 for u in up if u == 0)})
    return out


def rail() -> dict | None:
    base = R / "analysis" / "rail"
    if not (base / "rail_summary.csv").exists():
        return None
    resumen = list(csv.DictReader(open(base / "rail_summary.csv")))
    pruebas = list(csv.DictReader(open(base / "rail_tests.csv")))
    rep = _json(base / "rail_report.json") or {}
    instancias, rangos = [], collections.defaultdict(list)
    for inst in sorted({r["instance"] for r in resumen}):
        filas = sorted((r for r in resumen if r["instance"] == inst), key=lambda r: float(r["rpd_mean"]))
        for k, r in enumerate(filas, 1):
            rangos[r["algorithm"]].append(k)
        meta = rep.get("instances", {}).get(inst, {})
        ab = next((t for t in pruebas if t["instance"] == inst and t["a"] == "SIA-base"), None)
        comp = [t for t in pruebas if t["instance"] == inst and t["a"] == "SIA-online"]
        instancias.append({
            "nombre": inst, "bks": meta.get("bks"), "bks_verificado": meta.get("bks_verified"),
            "t10": meta.get("t10_v2elitr_rpd"),
            "metodos": [{"alg": r["algorithm"], "media": round(float(r["rpd_mean"]), 3),
                         "mediana": round(float(r["rpd_median"]), 3),
                         "desv": round(float(r["rpd_std"]), 3),
                         "mejor": round(float(r["rpd_best"]), 3),
                         "peor": round(float(r["rpd_worst"]), 3)} for r in filas],
            "ab": ab and {"mejor": int(ab["better_ref"]), "igual": int(ab["equal"]),
                          "peor": int(ab["worse_ref"]), "p": float(ab["p"]),
                          "p_finner": float(ab["p_finner"]), "A": float(ab["A_ref"])},
            "frente_competidores": {
                "mejor": sum(1 for t in comp if float(t["p_finner"]) < .05 and float(t["A_ref"]) > .5),
                "peor": sum(1 for t in comp if float(t["p_finner"]) < .05 and float(t["A_ref"]) < .5),
                "sin_diferencia": sum(1 for t in comp if float(t["p_finner"]) >= .05),
                "total": len(comp)},
        })
    ranking = sorted(((a, st.mean(v)) for a, v in rangos.items()), key=lambda x: x[1])
    return {"instancias": instancias, "avisos": rep.get("warnings", []),
            "ranking": [(a, round(v, 2)) for a, v in ranking]}


def main() -> None:
    fams, rangos, ranking_global = familias()
    datos = {"familias": fams, "rangos": rangos, "ranking_global": ranking_global,
             "salud": salud(), "rail": rail(), "auditoria": _json(R / "auditoria.json"),
             "config": _json(R / "config.json"),
             "hipotesis": _json(R / "analysis" / "hipotesis.json")}
    rep = _json(R / "informe" / "reproduccion_v2_fase1.json")
    if rep:
        n = f"{rep['comunes']:,}".replace(",", ".")
        datos["reproduccion_v2"] = (
            f"las {n} corridas de las familias originales coinciden bit a bit con las de la "
            f"campaña v2 ({rep['distintas']} distintas), ejecutada un día antes y con un "
            f"reinicio de la máquina de por medio")
    if datos["auditoria"]:          # la serie de memoria es larga y el informe no la usa
        datos["auditoria"]["memoria"].pop("serie", None)
        datos["auditoria"].pop("config", None)
    (R / "informe" / "datos.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False, default=str))
    plantilla = (AQUI / "plantilla.html").read_text()
    marca = "/*__DATOS__*/null"
    assert plantilla.count(marca) == 1, "plantilla sin marca de datos"
    html = plantilla.replace(marca, json.dumps(datos, ensure_ascii=False, default=str))
    (R / "informe" / "informe.html").write_text(html)
    print(f"datos.json e informe.html generados ({len(fams)} familias, "
          f"rail {'incluido' if datos['rail'] else 'pendiente'})")


if __name__ == "__main__":
    main()
