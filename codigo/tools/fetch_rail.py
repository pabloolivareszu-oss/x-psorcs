"""Descarga las instancias rail de la OR-Library y las verifica.

Uso (desde RecursosPaper/codigo/):
    python tools/fetch_rail.py

Para cada instancia: descarga, descomprime a benchmarks/rail_data/<nombre>.txt,
lee el encabezado (m n), lo compara con benchmarks/rail_data/bks.json y valida
la instancia completa con el lector del benchmark (toda fila cubierta, índices
en rango). Si n difiere de bks.json, lo corrige en el archivo y lo avisa.
El BKS NO se toca: se verifica a mano contra la fuente primaria.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.rail import DATA_DIR, INSTANCES, load_rail  # noqa: E402

BASE_URLS = [
    "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/",
    "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/",
]


def download(name: str) -> Path:
    out = DATA_DIR / f"{name}.txt"
    if out.exists():
        print(f"[{name}] ya existe {out.name}, no se descarga.")
        return out
    last = None
    for base in BASE_URLS:
        for suffix, gz in ((".gz", True), ("", False)):
            url = f"{base}{name}{suffix}"
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read()
                if gz:
                    data = gzip.decompress(data)
                out.write_bytes(data)
                print(f"[{name}] descargado de {url} ({len(data)/1e6:.1f} MB)")
                return out
            except Exception as e:  # noqa: BLE001
                last = e
    raise RuntimeError(f"No se pudo descargar {name}: {last}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bks_path = DATA_DIR / "bks.json"
    bks = json.loads(bks_path.read_text(encoding="utf-8"))
    changed = False
    for name in INSTANCES:
        path = download(name)
        with open(path) as fh:
            m, n = map(int, fh.readline().split()[:2])
        inst = load_rail(path)  # valida cobertura e índices
        rec = bks["instances"][name]
        print(f"[{name}] m={m} n={n} nnz={len(inst['col_rows'])} "
              f"costo min/max={inst['cost'].min():.0f}/{inst['cost'].max():.0f}")
        if m != rec["m"]:
            raise SystemExit(f"[{name}] m={m} no coincide con bks.json ({rec['m']}). Detente.")
        if n != rec["n"]:
            print(f"[{name}] AVISO: n={n} difiere de bks.json ({rec['n']}); se corrige.")
            rec["n"] = n
            changed = True
    if changed:
        bks_path.write_text(json.dumps(bks, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nInstancias listas. Recuerda verificar los BKS a mano y marcar 'verified': true.")


if __name__ == "__main__":
    main()
