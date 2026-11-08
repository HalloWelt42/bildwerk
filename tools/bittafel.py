"""Vergleichstafel einer Messreihe 4 gegen 8 Bit (tools/bitvergleich.py), je Aufgabe eine Zeile:

- Bearbeiten: das unmarkierte Ausgangsbild, die Vorlage mit Markierung (was das Modell sieht), 8 Bit, 4 Bit
- Erzeugen: die Beschreibung, aus der das Modell zeichnet, 8 Bit, 4 Bit

    .venv/bin/python tools/bittafel.py bild-fuellen "Qwen-Image 2.1 mit Maske"

Ablage: docs/messungen/bits/<dienst>-tafel.webp
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beispiele  # noqa: E402

ZIEL = beispiele.WURZEL / "docs" / "messungen" / "bits"
BITS = ("8", "4")
#: Breite eines Bildes in der Tafel, Abstand dazwischen und Höhe der Beschriftungen
BREITE, LUECKE, KOPF, ZEILENKOPF, BILDTEXT = 400, 12, 70, 34, 26
SCHRIFTEN = ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf")


@dataclass
class Zelle:
    text: str
    bild: Image.Image


def schrift(groesse: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in SCHRIFTEN:
        try:
            return ImageFont.truetype(name, groesse)
        except OSError:
            continue
    return ImageFont.load_default(size=groesse)


def eingepasst(bild: Image.Image, massstab_breite: int | None = None) -> Image.Image:
    """Auf Tafelbreite; mit massstab_breite im selben Maßstab wie ein Bild dieser Breite (ein kleineres Ausgangsbild
    beim Erweitern bleibt so sichtbar kleiner), mittig auf Weiß."""
    faktor = BREITE / (massstab_breite or bild.width)
    klein = bild.convert("RGB").resize((round(bild.width * faktor), round(bild.height * faktor)))
    if klein.width == BREITE:
        return klein
    feld = Image.new("RGB", (BREITE, round(klein.height * massstab_breite / bild.width)), "white")
    feld.paste(klein, ((BREITE - klein.width) // 2, (feld.height - klein.height) // 2))
    return feld


def textfeld(text: str, hoehe: int) -> Image.Image:
    feld = Image.new("RGB", (BREITE, hoehe), "#f1f1f1")
    zeilen = textwrap.wrap(" ".join(text.split()), width=44)
    ImageDraw.Draw(feld).multiline_text((12, 12), "\n".join(zeilen), fill="#222", font=schrift(17), spacing=5)
    return feld


def ausgang_zellen(aufgabe: str, a: dict, hoehe: int) -> list[Zelle]:
    """Was vor dem Modell da war: das unmarkierte Ausgangsbild und die Vorlage, wie das Modell sie sieht."""
    vorlage = Image.open(beispiele.ORDNER / "vorlagen" / f"{aufgabe}.webp")
    # beispiele legt ein eigenes Ausgangsbild nur ab, wo es sich von der Vorlage unterscheidet; sonst ist es das Motiv
    eigenes = beispiele.ORDNER / "ausgang" / f"{aufgabe}.webp"
    if eigenes.is_file():
        return [Zelle("Ausgangsbild", eingepasst(Image.open(eigenes), vorlage.width)),
                Zelle("Vorlage (was das Modell sieht)", eingepasst(vorlage))]
    return [Zelle("Ausgangsbild (so bekommt es das Modell)", eingepasst(vorlage)),
            Zelle("Keine Markierung, nur die Anweisung", textfeld(a.get("wunsch") or "", hoehe))]


def tafel(dienst: str, titel: str) -> Path:
    reihe = json.loads((ZIEL / f"{dienst}.json").read_text(encoding="utf-8"))
    aufgaben = beispiele.lesen()["aufgabe"]
    zeilen: list[tuple[str, list[Zelle]]] = []
    for name, a in aufgaben.items():
        if not (ZIEL / dienst / "8bit" / name / f"{dienst}.webp").is_file():
            continue
        ab = reihe["abweichung_4_zu_8"].get(name, {})
        ergebnisse = [Zelle(f"{b} Bit: {reihe['bits'][b]['dauer_s'][name]:.0f} s", eingepasst(
            Image.open(ZIEL / dienst / f"{b}bit" / name / f"{dienst}.webp"))) for b in BITS]
        ergebnisse[1].text += f"   (PSNR zu 8 Bit {ab.get('psnr_db')} dB)"
        hoehe = ergebnisse[0].bild.height
        # bearbeitet: bitvergleich legt je Aufgabe die Angaben des Auftrags daneben; erzeugt: nur das Bild
        if (ZIEL / dienst / "8bit" / name / f"{dienst}.json").is_file():
            kopf = f"{a['titel']} ({name})" + (f": {a['wunsch']}" if a.get("wunsch") else "")
            vorn = ausgang_zellen(name, a, hoehe)
        else:
            # beim Erzeugen entsteht nur das Motiv der Aufgabe, ihr Wunsch gilt erst beim Bearbeiten
            kopf, vorn = f"Motiv {a['titel']}", [Zelle("Beschreibung (Erzeugen)", textfeld(a["motiv"], hoehe))]
        zeilen.append((kopf, vorn + ergebnisse))
    spalten = max(len(z) for _, z in zeilen)
    zeile_h = ZEILENKOPF + BILDTEXT + max(c.bild.height for _, z in zeilen for c in z) + LUECKE
    bild = Image.new("RGB", (spalten * BREITE + (spalten + 1) * LUECKE, KOPF + len(zeilen) * zeile_h), "white")
    d = ImageDraw.Draw(bild)
    d.text((LUECKE, 10), titel, fill="black", font=schrift(26))
    werte = "      ".join(f"{b} Bit: Ablage {reihe['bits'][b]['ablage_gb']} GB, Spitze {reihe['bits'][b]['spitze_gb']} GB"
                          for b in BITS)
    d.text((LUECKE, 42), werte, fill="#444", font=schrift(18))
    for i, (kopf, zellen) in enumerate(zeilen):
        y = KOPF + i * zeile_h
        d.text((LUECKE, y + 6), kopf, fill="black", font=schrift(20))
        for s, zelle in enumerate(zellen):
            x = LUECKE + s * (BREITE + LUECKE)
            d.text((x, y + ZEILENKOPF), zelle.text, fill="#333", font=schrift(16))
            bild.paste(zelle.bild, (x, y + ZEILENKOPF + BILDTEXT))
    ziel = ZIEL / f"{dienst}-tafel.webp"
    bild.save(ziel, quality=85)
    return ziel


def main() -> int:
    p = argparse.ArgumentParser(description="Vergleichstafel einer Messreihe 4 gegen 8 Bit mit Ausgangsbild")
    p.add_argument("dienst", help="Kennung der Reihe, etwa bild-fuellen")
    p.add_argument("titel", help="Überschrift, etwa der Name des Modells")
    a = p.parse_args()
    print(tafel(a.dienst, a.titel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
