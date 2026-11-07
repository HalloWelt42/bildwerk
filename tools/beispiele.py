"""Portfolio der Bildmodelle (ATL-45 des Ateliers): je Aufgabe ein eigenes Motiv und je Modell, das die Aufgabe kann,
sein Ergebnis an genau diesem Motiv. Alles läuft über die Leitstelle (nie direkt an einen Dienst); was gerechnet
wird, steht in docs/beispiele/beispiele.toml.

    LEITSTELLE_ADRESSE=http://leitstelle:8530 .venv/bin/python tools/beispiele.py motive \\
        --schluessel-datei ~/.config/leitstelle/<anwendung>.schluessel
    LEITSTELLE_ADRESSE=... .venv/bin/python tools/beispiele.py rechnen --schluessel-datei ... [--aufgabe entfernen] [--neu]

Ablage: docs/beispiele/motive/<aufgabe>.png (Motiv), docs/beispiele/vorlagen/<aufgabe>.webp (was die Modelle sehen,
markiert wie in der Bildbearbeitung) und docs/beispiele/<aufgabe>/<dienst>.webp mit <dienst>.json (Dauer, Anweisung).
Vorhandenes bleibt, außer mit --neu.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

WURZEL = Path(__file__).resolve().parent.parent
ORDNER = WURZEL / "docs" / "beispiele"
BESCHREIBUNG = ORDNER / "beispiele.toml"
#: Kanten der Motive (Nutzer 07.11.2026: nicht über 1000 und nicht unter 400 Pixel)
KANTE_MIN, KANTE_MAX = 400, 1000
SIGNAL = (255, 0, 255)


def lesen() -> dict:
    b = tomllib.loads(BESCHREIBUNG.read_text(encoding="utf-8"))
    breite, hoehe = (int(x) for x in b["motive"]["groesse"].split("x"))
    if not (KANTE_MIN <= min(breite, hoehe) and max(breite, hoehe) <= KANTE_MAX):
        raise SystemExit(f"Motivgröße {breite}x{hoehe}: Kanten zwischen {KANTE_MIN} und {KANTE_MAX} Pixeln")
    return b


def b64(bild: Image.Image) -> str:
    puffer = io.BytesIO()
    bild.save(puffer, format="PNG")
    return base64.b64encode(puffer.getvalue()).decode("ascii")


def bild_aus(antwort: httpx.Response) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(antwort.json()["data"][0]["b64_json"]))).convert("RGB")


#: Verbindungsabbrüche (etwa ein Neustart der Leitstelle) so oft abwarten und den Auftrag wiederholen
VERSUCHE, WARTEN_S = 4, 15


def warten(c: httpx.Client) -> None:
    """Bis die Leitstelle wieder antwortet."""
    while True:
        time.sleep(WARTEN_S)
        try:
            if c.get("/v1/dienste", timeout=10).status_code == 200:
                return
        except httpx.TransportError:
            pass


def senden(c: httpx.Client, koerper: dict) -> httpx.Response:
    """Einen Bildauftrag an die Leitstelle; ist sie nicht bereit (Abbruch, 503), warten und denselben Auftrag neu senden."""
    for versuch in range(1, VERSUCHE + 1):
        try:
            r = c.post("/v1/images/generations", json=koerper)
            # 503: die Leitstelle startet gerade neu oder der Maschinenraum hat sich noch nicht gemeldet
            if r.status_code != 503 or versuch == VERSUCHE:
                return r
            grund = r.text[:120]
        except httpx.TransportError as f:
            if versuch == VERSUCHE:
                raise
            grund = type(f).__name__
        print(f"Leitstelle nicht bereit ({grund}), Versuch {versuch + 1} von {VERSUCHE} ...", flush=True)
        warten(c)
    raise AssertionError("unerreichbar")


def erzeugen(c: httpx.Client, m: dict, text: str) -> Image.Image:
    r = senden(c, {"model": m["dienst"], "prompt": " ".join(text.split()), "size": m["groesse"],
                                               "steps": m["schritte"], "seed": m["startwert"], "response_format": "b64_json"})
    if r.status_code != 200:
        raise SystemExit(f"Motiv: {r.status_code} {r.text[:300]}")
    return bild_aus(r)


def motive(c: httpx.Client, b: dict, neu: bool) -> None:
    (ORDNER / "motive").mkdir(parents=True, exist_ok=True)
    for aufgabe, a in b["aufgabe"].items():
        for art, text in (("", a["motiv"]), ("-objekt", a.get("objekt"))):
            ziel = ORDNER / "motive" / f"{aufgabe}{art}.png"
            if not text or (ziel.exists() and not neu):
                continue
            beginn = time.monotonic()
            erzeugen(c, b["motive"], text).save(ziel)
            print(f"{ziel.relative_to(WURZEL)}: {time.monotonic() - beginn:.0f} s", flush=True)


def freistellen(objekt: Image.Image, schwelle: int = 238) -> Image.Image:
    """Das Objekt vom weißen Grund lösen (grob, wie ein schnell freigestelltes Bild): fast Weißes wird durchsichtig."""
    rgba = objekt.convert("RGBA")
    rgba.putdata([(r, g, b, 0 if min(r, g, b) >= schwelle else 255) for r, g, b, _ in rgba.get_flattened_data()])
    return rgba.crop(rgba.getbbox()) if rgba.getbbox() else rgba


def vorlage(aufgabe: str, a: dict, motiv: Image.Image, farbe: tuple[int, int, int]) -> Image.Image:
    """Was das Modell sieht, wie die Bildbearbeitung es schickt."""
    v = motiv.copy()
    if aufgabe in ("entfernen", "fuellen"):
        if not a.get("flaeche"):
            raise SystemExit(f"{aufgabe}: in beispiele.toml fehlt die Fläche")
        ImageDraw.Draw(v).polygon([tuple(p) for p in a["flaeche"]], fill=farbe)
    elif aufgabe == "erweitern":
        r = int(a["rand"])
        v = Image.new("RGB", motiv.size, farbe)
        v.paste(motiv.resize((motiv.width - 2 * r, motiv.height - 2 * r), Image.LANCZOS), (r, r))
    elif aufgabe == "einpassen":
        if not a.get("objekt_kasten"):
            raise SystemExit("einpassen: in beispiele.toml fehlt objekt_kasten")
        x, y, breite, hoehe = a["objekt_kasten"]
        objekt = freistellen(Image.open(ORDNER / "motive" / "einpassen-objekt.png"))
        objekt.thumbnail((breite, hoehe), Image.LANCZOS)
        v.paste(objekt, (x + (breite - objekt.width) // 2, y + hoehe - objekt.height), objekt)
    return v


def anweisung(vorlage_des_modells: dict, wunsch: str) -> str:
    """Die Anweisung aus der Aufgabenvorlage, wie das Atelier sie zusammensetzt (Wunsch schon in der Sprache der Vorlage)."""
    if wunsch and "{wunsch}" in vorlage_des_modells["anweisung"]:
        return vorlage_des_modells["anweisung"].replace("{wunsch}", wunsch)
    return vorlage_des_modells.get("ohne_wunsch") or vorlage_des_modells["anweisung"]


def rechnen(c: httpx.Client, b: dict, nur: str | None, neu: bool) -> None:
    dienste = [d for d in c.get("/v1/dienste").json()["data"] if "bild_bearbeiten" in d.get("faehigkeiten", [])]
    for aufgabe, a in b["aufgabe"].items():
        if nur and aufgabe != nur:
            continue
        motiv = Image.open(ORDNER / "motive" / f"{aufgabe}.png").convert("RGB")
        for d in dienste:
            v_modell = ((d.get("bild") or {}).get("aufgaben") or {}).get(aufgabe)
            if not v_modell:
                continue
            ziel = ORDNER / aufgabe / f"{d['kennung']}.webp"
            if ziel.exists() and not neu:
                continue
            ziel.parent.mkdir(parents=True, exist_ok=True)
            farbe = tuple(v_modell.get("signalfarbe") or SIGNAL)
            v = vorlage(aufgabe, a, motiv, farbe)
            (ORDNER / "vorlagen").mkdir(exist_ok=True)
            v.save(ORDNER / "vorlagen" / f"{aufgabe}.webp", quality=90)
            text = anweisung(v_modell, a.get("wunsch", ""))
            beginn = time.monotonic()
            r = senden(c, {"model": d["kennung"], "prompt": text, "size": b["motive"]["groesse"], "seed": 1,
                           "response_format": "b64_json", "images": [b64(v)]})
            gesamt = time.monotonic() - beginn
            if r.status_code != 200:
                print(f"{aufgabe} / {d['kennung']}: {r.status_code} {r.text[:200]}", flush=True)
                continue
            bild_aus(r).save(ziel, quality=88)
            zusatz = r.json().get("x-leitstelle", {})
            angaben = {"dienst": d["kennung"], "name": d.get("name", ""), "aufgabe": aufgabe, "anweisung": text,
                       "dauer_s": round(zusatz.get("dauer_ms", 0) / 1000, 2), "bis_antwort_s": round(gesamt, 2),
                       "schritte": zusatz.get("schritte"), "gerechnet_am": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            ziel.with_suffix(".json").write_text(json.dumps(angaben, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{aufgabe} / {d['kennung']}: {angaben['dauer_s']} s im Dienst, {gesamt:.0f} s bis zur Antwort", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Portfolio der Bildmodelle über die Leitstelle")
    p.add_argument("befehl", choices=["motive", "rechnen"])
    p.add_argument("--schluessel-datei", required=True, type=Path)
    p.add_argument("--aufgabe", help="nur diese Aufgabe rechnen")
    p.add_argument("--neu", action="store_true", help="Vorhandenes neu erzeugen")
    a = p.parse_args()
    adresse = os.environ.get("LEITSTELLE_ADRESSE")
    if not adresse:
        raise SystemExit("LEITSTELLE_ADRESSE fehlt")
    b = lesen()
    schluessel = a.schluessel_datei.expanduser().read_text().strip()
    with httpx.Client(base_url=adresse, headers={"Authorization": f"Bearer {schluessel}"}, timeout=httpx.Timeout(3600, connect=10)) as c:
        motive(c, b, a.neu) if a.befehl == "motive" else rechnen(c, b, a.aufgabe, a.neu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
