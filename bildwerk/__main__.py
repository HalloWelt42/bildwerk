"""Startpunkt.

    bildwerk <modell> [--port N] [--probe]               Dienst des Modells (je Prozess genau ein Modell)
    bildwerk gewichte <modell> [--pruefsummen] [--laden] Gewichte in der Modellablage prüfen, mit --laden fehlende laden
    bildwerk pruefliste <modell>                         Prüfliste aus dem gepinnten Stand von mflux neu schreiben (Netz)
"""

from __future__ import annotations

import argparse
import sys

from .modelle import MODELLE, modul


def dienst(kennung: str, port: int | None, probe: bool) -> None:
    from .bilddienst import erstellen
    from .unterbau import Einstellungen, starten

    m = modul(kennung)
    einstellungen = Einstellungen()
    app = erstellen(kennung, m.Motor(einstellungen), einstellungen, probe=probe)
    starten(app, port or m.PORT, einstellungen)


def gewichte(kennung: str, pruefsummen: bool, laden: bool) -> int:
    from . import gewichte as g

    liste = g.Pruefliste.lesen(modul(kennung).PRUEFLISTE)
    wo = g.ablage()
    print(f"{liste.repo} @ {liste.revision}: {len(liste.dateien)} Dateien, {liste.groesse / 1e9:.1f} GB, Ablage {wo}")
    stand = g.pruefen(liste, wo)
    if laden and not stand.vollstaendig:
        print(f"lade {len(stand.fehlen)} Dateien ...", flush=True)
        g.laden(liste, wo)
    if pruefsummen or laden:
        stand = g.pruefen(liste, wo, pruefsummen=True)
    print(stand.meldung + (" (Prüfsummen stimmen)" if stand.vollstaendig and (pruefsummen or laden) else ""))
    return 0 if stand.vollstaendig else 1


def pruefliste(kennung: str) -> int:
    from .gewichte import erstellen

    m = modul(kennung)
    repo, revision, muster = m.mflux_herkunft()
    liste = erstellen(repo, revision, muster, m.ANGABEN.lizenz)
    liste.schreiben(m.PRUEFLISTE)
    print(f"{m.PRUEFLISTE.name}: {liste.repo} @ {liste.revision}, {len(liste.dateien)} Dateien, {liste.groesse / 1e9:.1f} GB")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="bildwerk", description="Bildwerk: Bilddienste, je Prozess ein Modell")
    befehle = p.add_subparsers(dest="befehl", required=True)
    for kennung in MODELLE:
        d = befehle.add_parser(kennung, help=f"Dienst {kennung}")
        d.add_argument("--port", type=int, help="Vorgabe je Modell")
        d.add_argument("--probe", action="store_true", help="Probemodus: lädt und rechnet nichts, antwortet auf /gesundheit und /v1/models")
    g = befehle.add_parser("gewichte", help="Gewichte eines Modells in der Modellablage (HF_HUB_CACHE) prüfen oder laden")
    g.add_argument("modell", choices=sorted(MODELLE))
    g.add_argument("--pruefsummen", action="store_true", help="auch den Inhalt prüfen (liest alle Dateien)")
    g.add_argument("--laden", action="store_true", help="fehlende Dateien aus dem Netz laden, danach mit Prüfsummen prüfen")
    l = befehle.add_parser("pruefliste", help="Prüfliste aus dem gepinnten Stand von mflux neu schreiben (braucht Netz)")
    l.add_argument("modell", choices=sorted(MODELLE))
    a = p.parse_args()
    if a.befehl == "gewichte":
        sys.exit(gewichte(a.modell, a.pruefsummen, a.laden))
    if a.befehl == "pruefliste":
        sys.exit(pruefliste(a.modell))
    dienst(a.befehl, a.port, a.probe)


if __name__ == "__main__":
    main()
