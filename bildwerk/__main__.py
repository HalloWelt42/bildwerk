"""Startpunkt.

    bildwerk <modell> [--port N] [--probe]               Dienst des Modells (je Prozess genau ein Modell)
    bildwerk gewichte <modell> [--pruefsummen] [--laden] Gewichte in der Modellablage prüfen, mit --laden fehlende laden
    bildwerk pruefliste <modell>                         Prüfliste aus dem gepinnten Stand von mflux neu schreiben (Netz)
    bildwerk umwandeln <modell> [--original-loeschen]    quantisiert in die eigene Ablage speichern (Modelle mit `lokal`);
                                                          das Original erst nach einer Messung, und nur in den Papierkorb

Modelle mit eigener, quantisierter Ablage haben zwei Listen: PRUEFLISTE_QUELLE (was aus Hugging Face geladen wird) und
PRUEFLISTE (die eigene Ablage, aus der der Dienst lädt).
"""

from __future__ import annotations

import argparse
import inspect
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

    m = modul(kennung)
    # mit --laden die Quelle aus Hugging Face, sonst das, woraus der Dienst lädt
    liste = g.Pruefliste.lesen(m.PRUEFLISTE_QUELLE if laden and hasattr(m, "PRUEFLISTE_QUELLE") else m.PRUEFLISTE)
    wo = g.ablage()
    print(f"{liste.repo} @ {liste.revision}: {len(liste.dateien)} Dateien, {liste.groesse / 1e9:.1f} GB, Ablage {wo}")
    stand = g.pruefen(liste, wo)
    if laden and not stand.vollstaendig:
        print(f"lade {len(stand.fehlen)} Dateien ...", flush=True)
        g.laden(liste, wo)
    if pruefsummen or laden:
        stand = g.pruefen(liste, wo, pruefsummen=True)
    print(stand.meldung + (" (Prüfsummen stimmen)" if stand.vollstaendig and (pruefsummen or laden) else ""))
    vollstaendig = stand.vollstaendig
    # Zusätze aus anderen Repos (etwa ein LoRA), die der Dienst neben den Grundgewichten braucht
    for datei in getattr(m, "ZUSATZ_PRUEFLISTEN", []):
        zusatz = g.Pruefliste.lesen(datei)
        print(f"Zusatz {zusatz.repo} @ {zusatz.revision}: {len(zusatz.dateien)} Dateien, {zusatz.groesse / 1e9:.2f} GB")
        z = g.pruefen(zusatz, wo)
        if laden and not z.vollstaendig:
            g.laden(zusatz, wo)
        if pruefsummen or laden:
            z = g.pruefen(zusatz, wo, pruefsummen=True)
        print(z.meldung)
        vollstaendig = vollstaendig and z.vollstaendig
    return 0 if vollstaendig else 1


def pruefliste(kennung: str) -> int:
    from .gewichte import erstellen

    m = modul(kennung)
    repo, revision, muster = m.herkunft()
    liste = erstellen(repo, revision, muster, m.ANGABEN.lizenz)
    ziel = getattr(m, "PRUEFLISTE_QUELLE", m.PRUEFLISTE)
    liste.schreiben(ziel)
    print(f"{ziel.name}: {liste.repo} @ {liste.revision}, {len(liste.dateien)} Dateien, {liste.groesse / 1e9:.1f} GB")
    return 0


def umwandeln(kennung: str, original_loeschen: bool, bits: int | None = None) -> int:
    """Quantisiert in die eigene Ablage speichern, die neue Prüfliste schreiben, Probeladen; erst danach auf Wunsch das
    Original aus dem Zwischenspeicher löschen. Rechnet nichts (kein Bild), braucht aber den Speicher des Modells."""
    m = modul(kennung)
    if not hasattr(m, "umwandeln"):
        print(f"{kennung} lädt direkt aus Hugging Face; Umwandeln gibt es nur für Modelle mit eigener Ablage.")
        return 2
    if bits is None:
        return m.umwandeln(original_loeschen)
    if "bits" not in inspect.signature(m.umwandeln).parameters:
        print(f"{kennung} hat nur eine Ablage; --bits gilt hier nicht.")
        return 2
    return m.umwandeln(original_loeschen, bits)


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
    u = befehle.add_parser("umwandeln", help="quantisiert in die eigene Ablage speichern (Modelle mit eigener Ablage)")
    u.add_argument("modell", choices=sorted(MODELLE))
    u.add_argument("--bits", type=int, choices=[4, 8], help="Bitbreite der eigenen Ablage (Modelle, die mehrere kennen; Vorgabe 8)")
    u.add_argument("--original-loeschen", action="store_true", help="das Original in den Papierkorb legen, wenn eine Messung belegt, dass die eigene Ablage rechnet")
    a = p.parse_args()
    if a.befehl == "gewichte":
        sys.exit(gewichte(a.modell, a.pruefsummen, a.laden))
    if a.befehl == "pruefliste":
        sys.exit(pruefliste(a.modell))
    if a.befehl == "umwandeln":
        sys.exit(umwandeln(a.modell, a.original_loeschen, a.bits))
    dienst(a.befehl, a.port, a.probe)


if __name__ == "__main__":
    main()
