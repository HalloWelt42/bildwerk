"""Vergleichstafel aus dem Portfolio (docs/beispiele): je Aufgabe eine Zeile mit dem unmarkierten Ausgangsbild, der
Vorlage, wie die Modelle sie sehen, und dem Ergebnis jedes genannten Dienstes. Mit --erzeugen vergleicht sie Erzeuger
an den Motiven: links die Beschreibung, daneben die 8-Bit-Ergebnisse der Messreihen (docs/messungen/bits).

    .venv/bin/python tools/vergleichstafel.py "Qwen-Image 2.1 gegen Qwen-Image-Edit" vergleich.webp \\
        --aufgabe entfernen --aufgabe fuellen --dienst bild-fuellen --dienst bildwerk-qwen-image-edit-schnell
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beispiele  # noqa: E402
from bittafel import (  # noqa: E402
    BILDTEXT, BREITE, KOPF, LUECKE, ZEILENKOPF, ZIEL, Zelle, ausgang_zellen, eingepasst, schrift, textfeld)


def ergebnis(aufgabe: str, dienst: str) -> Zelle:
    ordner = beispiele.ORDNER / aufgabe
    angaben = json.loads((ordner / f"{dienst}.json").read_text(encoding="utf-8"))
    return Zelle(f"{angaben.get('name', dienst)}: {angaben['dauer_s']:.0f} s", eingepasst(Image.open(ordner / f"{dienst}.webp")))


def erzeugt(aufgabe: str, angabe: str) -> Zelle:
    """Das 8-Bit-Ergebnis eines Erzeugers aus seiner Messreihe, mit Schritten und Dauer; angabe ist die Kennung der
    Reihe, wahlweise mit lesbarem Namen dahinter (bild-erzeugen=FLUX 3 Image Klein)."""
    dienst, _, name = angabe.partition("=")
    reihe = json.loads((ZIEL / f"{dienst}.json").read_text(encoding="utf-8"))
    # ohne eigene Schrittzahl lief die Reihe mit der Vorgabe des Modells, die nur der Name nennen kann
    schritte = f", {reihe['schritte']} Schritte" if reihe.get("schritte") else ""
    return Zelle(f"{name or reihe['modell']}{schritte}: {reihe['bits']['8']['dauer_s'][aufgabe]:.0f} s",
                 eingepasst(Image.open(ZIEL / dienst / "8bit" / aufgabe / f"{dienst}.webp")))


def tafel(titel: str, aufgaben_namen: list[str], dienste: list[str], ziel: Path, erzeugen: bool = False,
          spalten: int | None = None) -> Path:
    aufgaben = beispiele.lesen()["aufgabe"]
    zeilen = []
    for name in aufgaben_namen:
        a = aufgaben[name]
        if erzeugen:
            ergebnisse = [erzeugt(name, d) for d in dienste]
            vorn = [Zelle("Beschreibung", textfeld(a["motiv"], ergebnisse[0].bild.height))]
            zeilen.append((f"Motiv {a['titel']}", vorn + ergebnisse))
            continue
        ergebnisse = [ergebnis(name, d) for d in dienste]
        kopf = f"{a['titel']} ({name})" + (f": {a['wunsch']}" if a.get("wunsch") else "")
        zeilen.append((kopf, ausgang_zellen(name, a, ergebnisse[0].bild.height) + ergebnisse))
    # höchstens "spalten" Kacheln je Bildzeile; was nicht passt, bricht in die nächste Zeile um (Überschrift nur oben)
    spalten = spalten or max(len(z) for _, z in zeilen)
    reihen = [(kopf if s == 0 else None, zellen[s:s + spalten]) for kopf, zellen in zeilen
              for s in range(0, len(zellen), spalten)]
    bild_h = max(c.bild.height for _, z in zeilen for c in z)
    hoehen = [(ZEILENKOPF if kopf is not None else 0) + BILDTEXT + bild_h + LUECKE for kopf, _ in reihen]
    bild = Image.new("RGB", (spalten * BREITE + (spalten + 1) * LUECKE, KOPF - 30 + sum(hoehen)), "white")
    d = ImageDraw.Draw(bild)
    d.text((LUECKE, 10), titel, fill="black", font=schrift(26))
    y = KOPF - 30
    for (kopf, zellen), hoehe in zip(reihen, hoehen):
        if kopf is not None:
            d.text((LUECKE, y + 6), kopf, fill="black", font=schrift(20))
            y += ZEILENKOPF
        for s, zelle in enumerate(zellen):
            x = LUECKE + s * (BREITE + LUECKE)
            d.text((x, y), zelle.text, fill="#333", font=schrift(16))
            bild.paste(zelle.bild, (x, y + BILDTEXT))
        y += hoehe - (ZEILENKOPF if kopf is not None else 0)
    bild.save(ziel, quality=85)
    return ziel


def main() -> int:
    p = argparse.ArgumentParser(description="Vergleichstafel mehrerer Dienste aus dem Portfolio, mit Ausgangsbild")
    p.add_argument("titel")
    p.add_argument("ziel", type=Path)
    p.add_argument("--aufgabe", action="append", required=True)
    p.add_argument("--dienst", action="append", required=True)
    p.add_argument("--erzeugen", action="store_true", help="Erzeuger an den Motiven statt Bearbeiter am Ausgangsbild")
    p.add_argument("--spalten", type=int, help="höchstens so viele Kacheln je Bildzeile, Rest bricht um")
    a = p.parse_args()
    print(tafel(a.titel, a.aufgabe, a.dienst, a.ziel, a.erzeugen, a.spalten))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
