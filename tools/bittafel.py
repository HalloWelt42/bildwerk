"""Vergleichstafel einer Messreihe 4 gegen 8 Bit (tools/bitvergleich.py): je Aufgabe eine Zeile mit dem Ausgang
(beim Bearbeiten die Vorlage, wie das Modell sie sieht, beim Erzeugen die Beschreibung), daneben 8 Bit und 4 Bit.

    .venv/bin/python tools/bittafel.py bild-fuellen "Qwen-Image 2.1 mit Maske"

Ablage: docs/messungen/bits/<dienst>-tafel.webp
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beispiele  # noqa: E402

ZIEL = beispiele.WURZEL / "docs" / "messungen" / "bits"
BITS = ("8", "4")
#: Breite eines Bildes in der Tafel, Abstand dazwischen und Höhe der Beschriftungen
BREITE, LUECKE, KOPF, ZEILENKOPF, BILDTEXT = 440, 12, 70, 34, 26
SCHRIFTEN = ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf")


def schrift(groesse: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in SCHRIFTEN:
        try:
            return ImageFont.truetype(name, groesse)
        except OSError:
            continue
    return ImageFont.load_default(size=groesse)


def eingepasst(bild: Image.Image) -> Image.Image:
    return bild.convert("RGB").resize((BREITE, round(bild.height * BREITE / bild.width)))


def beschreibung(text: str, hoehe: int) -> Image.Image:
    """Beim Erzeugen gibt es kein Ausgangsbild: die Beschreibung, aus der das Modell zeichnet."""
    feld = Image.new("RGB", (BREITE, hoehe), "#f1f1f1")
    zeilen = textwrap.wrap(" ".join(text.split()), width=48)
    ImageDraw.Draw(feld).multiline_text((12, 12), "\n".join(zeilen), fill="#222", font=schrift(17), spacing=5)
    return feld


def tafel(dienst: str, titel: str) -> Path:
    reihe = json.loads((ZIEL / f"{dienst}.json").read_text(encoding="utf-8"))
    aufgaben = beispiele.lesen()["aufgabe"]
    namen = [a for a in aufgaben if (ZIEL / dienst / "8bit" / a / f"{dienst}.webp").is_file()]
    zeilen = []
    for a in namen:
        paar = [eingepasst(Image.open(ZIEL / dienst / f"{b}bit" / a / f"{dienst}.webp")) for b in BITS]
        # bearbeitet: bitvergleich legt je Aufgabe die Angaben des Auftrags daneben; erzeugt: nur das Bild
        bearbeitet = (ZIEL / dienst / "8bit" / a / f"{dienst}.json").is_file()
        ausgang = (eingepasst(Image.open(beispiele.ORDNER / "vorlagen" / f"{a}.webp")) if bearbeitet
                   else beschreibung(aufgaben[a]["motiv"], paar[0].height))
        zeilen.append((a, bearbeitet, ausgang, paar))
    zeile_h = ZEILENKOPF + BILDTEXT + max(z[3][0].height for z in zeilen) + LUECKE
    bild = Image.new("RGB", (3 * BREITE + 4 * LUECKE, KOPF + len(zeilen) * zeile_h), "white")
    d = ImageDraw.Draw(bild)
    d.text((LUECKE, 10), titel, fill="black", font=schrift(26))
    werte = "      ".join(f"{b} Bit: Ablage {reihe['bits'][b]['ablage_gb']} GB, Spitze {reihe['bits'][b]['spitze_gb']} GB"
                          for b in BITS)
    d.text((LUECKE, 42), werte, fill="#444", font=schrift(18))
    for i, (a, bearbeitet, ausgang, paar) in enumerate(zeilen):
        y = KOPF + i * zeile_h
        # beim Erzeugen entsteht nur das Motiv der Aufgabe, ihr Wunsch gilt erst beim Bearbeiten
        wunsch = aufgaben[a].get("wunsch") if bearbeitet else ""
        kopf = f"{aufgaben[a]['titel']} ({a})" if bearbeitet else f"Motiv {aufgaben[a]['titel']}"
        d.text((LUECKE, y + 6), kopf + (f": {wunsch}" if wunsch else ""), fill="black", font=schrift(20))
        ab = reihe["abweichung_4_zu_8"].get(a, {})
        texte = ["Ausgang, wie das Modell es sieht" if bearbeitet else "Beschreibung (Erzeugen)"]
        texte += [f"{b} Bit: {reihe['bits'][b]['dauer_s'][a]:.0f} s" for b in BITS]
        texte[2] += f"   (PSNR zu 8 Bit {ab.get('psnr_db')} dB)"
        for s, (text, teil) in enumerate(zip(texte, [ausgang, *paar])):
            x = LUECKE + s * (BREITE + LUECKE)
            d.text((x, y + ZEILENKOPF), text, fill="#333", font=schrift(17))
            bild.paste(teil, (x, y + ZEILENKOPF + BILDTEXT))
    ziel = ZIEL / f"{dienst}-tafel.webp"
    bild.save(ziel, quality=85)
    return ziel


def main() -> int:
    p = argparse.ArgumentParser(description="Vergleichstafel einer Messreihe 4 gegen 8 Bit mit Ausgang")
    p.add_argument("dienst", help="Kennung der Reihe, etwa bild-fuellen")
    p.add_argument("titel", help="Überschrift, etwa der Name des Modells")
    a = p.parse_args()
    print(tafel(a.dienst, a.titel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
