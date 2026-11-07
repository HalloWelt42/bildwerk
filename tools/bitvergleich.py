"""Messreihe 4 gegen 8 Bit eines Bilddienstes (MR-5): dieselben Aufgaben aus dem Portfolio in beiden Bitbreiten über die
Leitstelle rechnen und vergleichen: Dauer im Dienst, Speicherspitze laut Maschinenraum, Größe der eigenen Ablage und wie
weit das Bild in 4 Bit von dem in 8 Bit abweicht (mittlere Abweichung je Farbwert, PSNR). Die Bilder liegen danach
nebeneinander zum Ansehen.

    LEITSTELLE_ADRESSE=http://leitstelle:8530 .venv/bin/python tools/bitvergleich.py bild-entfernen \\
        --modell flux2-klein-entfernen --schluessel-datei ~/.config/leitstelle/<anwendung>.schluessel

Ablauf je Bitbreite: in der Dienstdatei des Maschinenraums BILDWERK_QUANTISIERUNG setzen, den Dienst über die Leitstelle
entladen (der nächste Auftrag startet ihn mit der neuen Einstellung), dann rechnen. Am Ende steht die Einstellung wieder
wie vorher. Ergebnis: docs/messungen/bits/<dienst>.json und Bilder unter docs/messungen/bits/<dienst>/<bits>bit/.
Bilddienste, die nur erzeugen (etwa MOSS-Image), rechnen die Motive des Portfolios statt der Aufgaben.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beispiele  # noqa: E402

WURZEL = Path(__file__).resolve().parent.parent
ZIEL = WURZEL / "docs" / "messungen" / "bits"
DIENSTE = Path(os.environ.get("MASCHINENRAUM_ORDNER", Path.home() / "Maschinenraum")) / "dienste"
MASCHINENRAUM = os.environ.get("MASCHINENRAUM_ADRESSE", "http://127.0.0.1:8540")
#: Verwaltung der Leitstelle im Heimnetz (Entladen); aus der Adresse der Fassade abgeleitet (Port + 1, https)
VERWALTUNG = os.environ.get("LEITSTELLE_VERWALTUNG")
BITS = (8, 4)


def einstellen(dienst: str, bits: int) -> str | None:
    """BILDWERK_QUANTISIERUNG in der Dienstdatei setzen; liefert den Wert davor (None: stand nicht drin)."""
    datei = DIENSTE / f"{dienst}.toml"
    text = datei.read_text(encoding="utf-8")
    vorher = re.search(r'^BILDWERK_QUANTISIERUNG = "([^"]*)"', text, re.M)
    zeile = f'BILDWERK_QUANTISIERUNG = "{bits}bit"'
    text = re.sub(r'^BILDWERK_QUANTISIERUNG = "[^"]*"', zeile, text, flags=re.M) if vorher else text.rstrip("\n") + f"\n{zeile}\n"
    datei.write_text(text, encoding="utf-8")
    return vorher.group(1) if vorher else None


def zuruecksetzen(dienst: str, vorher: str | None) -> None:
    datei = DIENSTE / f"{dienst}.toml"
    text = datei.read_text(encoding="utf-8")
    if vorher is None:
        text = re.sub(r'^BILDWERK_QUANTISIERUNG = "[^"]*"\n?', "", text, flags=re.M)
    else:
        text = re.sub(r'^BILDWERK_QUANTISIERUNG = "[^"]*"', f'BILDWERK_QUANTISIERUNG = "{vorher}"', text, flags=re.M)
    datei.write_text(text, encoding="utf-8")


def stand(dienst: str) -> dict:
    return next(x for x in httpx.get(f"{MASCHINENRAUM}/api/stand", timeout=10).json()["dienste"] if x["kennung"] == dienst)


def entladen(verwaltung: str, dienst: str) -> None:
    """Den Dienst über die Leitstelle entladen und warten, bis der Maschinenraum ihn als aus meldet."""
    lage = httpx.get(f"{verwaltung}/api/lage", verify=False, timeout=15).json()
    modell = next(m["kennung"] for m in lage["modelle"] if m.get("familie") == dienst)
    httpx.post(f"{verwaltung}/api/modelle/{modell}/entladen", verify=False, timeout=60)
    for _ in range(120):
        if stand(dienst)["zustand"] == "aus":
            return
        time.sleep(2)
    raise SystemExit(f"{dienst} wurde nicht entladen")


#: so lange nach dem Umstellen warten, bis der Maschinenraum die Dienstdatei in seiner nächsten Runde gelesen hat
WARTEN_NACH_UMSTELLEN_S = 30


def gerechnet_mit(c: httpx.Client, dienst: str) -> str | None:
    """Die Genauigkeit, die der Dienst bei seinem letzten Start im Steckbrief gemeldet hat (laut Leitstelle)."""
    for m in c.get("/v1/models").json()["data"]:
        if m["id"] == dienst:
            return ((m.get("x-leitstelle") or {}).get("steckbrief") or {}).get("quantisierung")
    return None


def abweichung(a: Path, b: Path) -> dict:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return {"mittel": round(float(np.abs(x - y).mean()), 2), "psnr_db": round(10 * np.log10(255 ** 2 / mse), 1) if mse else None}


def ablage_gb(modell: str, bits: int) -> float | None:
    datei = next((WURZEL / "bildwerk" / "modelle").glob(f"{modell.replace('-', '_')}_{bits}bit.json"), None)
    return round(sum(d["groesse"] for d in json.loads(datei.read_text())["dateien"]) / 1e9, 1) if datei else None


def main() -> int:
    p = argparse.ArgumentParser(description="Messreihe 4 gegen 8 Bit eines Bilddienstes über die Leitstelle")
    p.add_argument("dienst", help="Kennung im Maschinenraum, etwa bild-entfernen")
    p.add_argument("--modell", required=True, help="Kennung im Bildwerk (für die Größe der Ablagen), etwa flux2-klein-entfernen")
    p.add_argument("--schluessel-datei", required=True, type=Path)
    p.add_argument("--aufgabe", help="nur diese Aufgabe")
    a = p.parse_args()
    adresse = os.environ.get("LEITSTELLE_ADRESSE") or sys.exit("LEITSTELLE_ADRESSE fehlt")
    verwaltung = VERWALTUNG or re.sub(r"^http://([^:/]+):(\d+)", lambda m: f"https://{m.group(1)}:{int(m.group(2)) + 1}", adresse)
    cfg = beispiele.lesen()
    schluessel = a.schluessel_datei.expanduser().read_text().strip()
    reihe: dict = {"dienst": a.dienst, "modell": a.modell, "gemessen_am": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bits": {}}
    vorher = einstellen(a.dienst, BITS[0])
    try:
        with httpx.Client(base_url=adresse, headers={"Authorization": f"Bearer {schluessel}"}, timeout=httpx.Timeout(3600, connect=10)) as c:
            bearbeitet = any(x["kennung"] == a.dienst and "bild_bearbeiten" in x.get("faehigkeiten", []) for x in c.get("/v1/dienste").json()["data"])
            for bits in BITS:
                einstellen(a.dienst, bits)
                time.sleep(WARTEN_NACH_UMSTELLEN_S)
                entladen(verwaltung, a.dienst)
                # die Speicherspitze je Lauf: der Maschinenraum hält sonst den Höchstwert über alle Läufe
                httpx.post(f"{MASCHINENRAUM}/api/dienste/{a.dienst}/spitzen_verwerfen", headers={"X-Maschinenraum": "1"}, timeout=10).raise_for_status()
                ausgabe = ZIEL / a.dienst / f"{bits}bit"
                dauern = {}
                if bearbeitet:
                    beispiele.rechnen(c, cfg, a.aufgabe, True, a.dienst, ausgabe)
                    for j in ausgabe.glob("*/*.json"):
                        dauern[j.parent.name] = json.loads(j.read_text())["dauer_s"]
                else:
                    m = dict(cfg["motive"], dienst=a.dienst)
                    for aufgabe, werte in cfg["aufgabe"].items():
                        if a.aufgabe and aufgabe != a.aufgabe:
                            continue
                        beginn = time.monotonic()
                        r = beispiele.senden(c, {"model": a.dienst, "prompt": " ".join(werte["motiv"].split()), "size": m["groesse"],
                                                 "steps": m["schritte"], "seed": m["startwert"], "response_format": "b64_json"})
                        if r.status_code != 200:
                            raise SystemExit(f"{aufgabe}: {r.status_code} {r.text[:200]}")
                        (ausgabe / aufgabe).mkdir(parents=True, exist_ok=True)
                        beispiele.bild_aus(r).save(ausgabe / aufgabe / f"{a.dienst}.webp", quality=88)
                        dauern[aufgabe] = round(r.json().get("x-leitstelle", {}).get("dauer_ms", 0) / 1000, 2) or round(time.monotonic() - beginn, 2)
                        print(f"{aufgabe}: {dauern[aufgabe]} s", flush=True)
                # ohne diese Probe wäre ein Lauf in der falschen Bitbreite nicht zu erkennen (erste Reihe 07.11.2026)
                if (wirklich := gerechnet_mit(c, a.dienst)) != f"{bits}bit":
                    raise SystemExit(f"{a.dienst} lief mit {wirklich}, verlangt waren {bits} Bit: Messung ungültig")
                reihe["bits"][bits] = {"dauer_s": dauern, "spitze_gb": stand(a.dienst).get("spitze_gb"), "ablage_gb": ablage_gb(a.modell, bits)}
    finally:
        zuruecksetzen(a.dienst, vorher)
    acht, vier = (ZIEL / a.dienst / "8bit"), (ZIEL / a.dienst / "4bit")
    reihe["abweichung_4_zu_8"] = {b.parent.name: abweichung(b, vier / b.parent.name / b.name)
                                  for b in sorted(acht.glob("*/*.webp")) if (vier / b.parent.name / b.name).is_file()}
    ZIEL.mkdir(parents=True, exist_ok=True)
    (ZIEL / f"{a.dienst}.json").write_text(json.dumps(reihe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(reihe, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
