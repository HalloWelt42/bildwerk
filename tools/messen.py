"""Kurzmessung eines Bilddienstes über die Leitstelle (nie direkt am Dienst): Läufe ohne und mit Bildvorlage, je mit
eigenem Startwert; Dauer im Dienst, Zeit bis zur Antwort (beim ersten Lauf mit Laden) und die Speicherspitze, die der
Maschinenraum dieses Rechners für den Dienst gemessen hat. Das Ergebnis geht als JSON nach docs/messungen/<modell>.json
und als Tabelle auf die Ausgabe.

    LEITSTELLE_ADRESSE=http://leitstelle:8530 .venv/bin/python tools/messen.py flux2-klein \\
        --schluessel-datei ~/.config/leitstelle/<anwendung>.schluessel --vorlage stil.png [--ergebnis bild.png]

Der Dienst heißt, wie das Modul ihn nennt (DIENST nach Fähigkeit, sonst bildwerk-<modell>).

Der Schlüssel gehört einer Anwendung der Leitstelle, die bild_erzeugen für dieses Modell darf; er wird nie ausgegeben.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

WURZEL = Path(__file__).resolve().parent.parent
MASCHINENRAUM = os.environ.get("MASCHINENRAUM_ADRESSE", "http://127.0.0.1:8540")


def spitze_gb(dienst: str) -> float | None:
    """Die gemessene Speicherspitze des Dienstes laut Maschinenraum dieses Rechners (nur lesen)."""
    try:
        stand = httpx.get(f"{MASCHINENRAUM}/api/stand", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return None
    eintrag = next((d for d in stand.get("dienste", []) if d.get("kennung") == dienst), None)
    return eintrag.get("spitze_gb") if eintrag else None


def lauf(c: httpx.Client, dienst: str, seed: int, vorlage: str | None, groesse: str, schritte: int, ergebnis: Path | None = None) -> dict:
    koerper = {"model": dienst, "prompt": "A quiet reading room at night, warm orange light, a globe and an open book on a desk",
               "size": groesse, "steps": schritte, "seed": seed, "response_format": "b64_json"}
    if vorlage:
        koerper["images"] = [vorlage]
    beginn = time.monotonic()
    r = c.post("/v1/images/generations", json=koerper)
    gesamt_s = time.monotonic() - beginn
    if r.status_code != 200:
        raise SystemExit(f"Lauf mit Startwert {seed}: {r.status_code} {r.text[:300]}")
    zusatz = r.json().get("x-leitstelle", {})
    if ergebnis is not None and not ergebnis.exists():
        ergebnis.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
    return {"seed": seed, "vorlage": bool(vorlage), "groesse": groesse, "schritte": zusatz.get("schritte", schritte),
            "dauer_s": round(zusatz.get("dauer_ms", 0) / 1000, 2), "bis_antwort_s": round(gesamt_s, 2)}


def main() -> int:
    p = argparse.ArgumentParser(description="Kurzmessung eines Bilddienstes über die Leitstelle")
    p.add_argument("modell", help="Kennung im Bildwerk, etwa flux2-klein")
    p.add_argument("--schluessel-datei", required=True, type=Path)
    p.add_argument("--vorlage", type=Path, help="Bild als Stilvorlage für die Läufe mit Vorlage")
    p.add_argument("--laeufe", type=int, default=3)
    p.add_argument("--groesse", default="1024x1024")
    p.add_argument("--schritte", type=int, help="Vorgabe: die des Modells")
    p.add_argument("--ergebnis", type=Path, help="das erste Bild mit Vorlage hierhin speichern (zum Ansehen der Qualität)")
    a = p.parse_args()
    adresse = os.environ.get("LEITSTELLE_ADRESSE")
    if not adresse:
        raise SystemExit("LEITSTELLE_ADRESSE setzen (Adresse der Leitstelle)")
    sys.path.insert(0, str(WURZEL))
    from bildwerk.modelle import dienstkennung, modul
    dienst = dienstkennung(a.modell)
    angaben = modul(a.modell).ANGABEN
    schritte = a.schritte or angaben.schritte.vorgabe
    if angaben.vorlagen_min and not a.vorlage:
        raise SystemExit(f"{angaben.name} rechnet nur mit Vorlage: --vorlage angeben")
    schluessel = a.schluessel_datei.expanduser().read_text().strip()
    vorlage = base64.b64encode(a.vorlage.read_bytes()).decode("ascii") if a.vorlage else None
    laeufe = []
    with httpx.Client(base_url=adresse, headers={"Authorization": f"Bearer {schluessel}"}, timeout=httpx.Timeout(900, connect=10)) as c:
        # ein reines Bearbeitungsmodell (vorlagen_min) rechnet nur mit Vorlage
        for i in range(0 if angaben.vorlagen_min else a.laeufe):
            laeufe.append(lauf(c, dienst, 100 + i, None, a.groesse, schritte))
            print(laeufe[-1], flush=True)
        if vorlage:
            for i in range(a.laeufe):
                laeufe.append(lauf(c, dienst, 200 + i, vorlage, a.groesse, schritte, a.ergebnis))
                print(laeufe[-1], flush=True)

    def mittel(mit: bool) -> float | None:
        werte = [x["dauer_s"] for x in laeufe if x["vorlage"] == mit]
        return round(statistics.mean(werte), 2) if werte else None

    ergebnis = {"modell": a.modell, "dienst": dienst, "gemessen_am": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "groesse": a.groesse, "schritte": schritte,
                "laden_und_warten_s": round(laeufe[0]["bis_antwort_s"] - laeufe[0]["dauer_s"], 1),
                "dauer_ohne_vorlage_s": mittel(False), "dauer_mit_vorlage_s": mittel(True),
                "speicher_spitze_gb": spitze_gb(dienst), "laeufe": laeufe}
    ziel = WURZEL / "docs" / "messungen" / f"{a.modell}.json"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n| Lauf | Vorlage | Dauer im Dienst | bis Antwort |\n|---|---|---|---|")
    for x in laeufe:
        print(f"| {x['seed']} | {'ja' if x['vorlage'] else 'nein'} | {x['dauer_s']} s | {x['bis_antwort_s']} s |")
    print(f"\nLaden und Warten vor dem ersten Bild: {ergebnis['laden_und_warten_s']} s; Speicherspitze: {ergebnis['speicher_spitze_gb']} GB")
    print(f"geschrieben: {ziel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
