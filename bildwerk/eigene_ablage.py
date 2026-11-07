"""Eigene, schon quantisierte Ablage eines mflux-Modells (MR-4): einmal aus dem Original im Zwischenspeicher bauen und
speichern, danach lädt der Dienst nur noch daraus (schneller, halb so groß, nichts wird beim Laden umgerechnet).

Ablauf von `umwandeln`: Original prüfen, das Modell mit `bauen` erzeugen (quantisiert, Zusätze wie ein LoRA eingerechnet)
und in den Ordner `lokal` der Modellablage speichern, Begleitdateien dazulegen, Prüfliste aus dem Ordner schreiben, mit
`probe` zur Probe aus der eigenen Ablage laden. Das Original bleibt liegen: Quellen und Duplikate räumt der Nutzer später
gemeinsam ab (MR-4).
"""

from __future__ import annotations

import gc
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .gewichte import Pruefliste, ablage, aus_ordner, pruefen


def fertig(liste: Pruefliste) -> bool:
    """Liegt die eigene Ablage vollständig vor?"""
    return bool(liste.dateien) and pruefen(liste, ablage()).vollstaendig


def umwandeln(kennung: str, quelle: Pruefliste, ziel_liste: Path, lokal: str, bauen: Callable[[], Any],
              probe: Callable[[str], Any], beilagen: list[Path] | None = None) -> int:
    """Die eigene Ablage bauen (siehe oben). 0: fertig; 1: das Original fehlt."""
    wo = ablage()
    stand = pruefen(quelle, wo)
    if not stand.vollstaendig:
        print(f"{stand.meldung}. Zuerst: bildwerk gewichte {kennung} --laden")
        return 1
    ziel = wo.parent / lokal
    if not (ziel.is_dir() and any(ziel.iterdir())):
        print(f"speichere nach {ziel} ...", flush=True)
        m = bauen()
        m.save_model(str(ziel))
        del m
        gc.collect()
    for datei in beilagen or []:
        if not (ziel / datei.name).is_file():
            shutil.copy2(datei, ziel / datei.name)
    liste = aus_ordner(ziel, quelle, lokal)
    liste.schreiben(ziel_liste)
    print(f"{ziel_liste.name}: {len(liste.dateien)} Dateien, {liste.groesse / 1e9:.1f} GB in {ziel}", flush=True)
    print("lade zur Probe aus der eigenen Ablage ...", flush=True)
    m = probe(str(ziel))
    del m
    gc.collect()
    print("Probeladen gelungen. Das Original bleibt; Quellen und Duplikate räumen wir später gemeinsam ab (MR-4).")
    return 0
