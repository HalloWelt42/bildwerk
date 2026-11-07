"""Beispiele im Steckbrief (ATL-45 des Ateliers): je Aufgabe zeigt ein Modell, was es an einem festen Motiv macht, damit
eine Anwendung die Modelle bei der Wahl vergleichen kann. Das Portfolio rechnet tools/beispiele.py über die Leitstelle
(docs/beispiele); hier werden daraus kleine Bilder für den Steckbrief.

Klein heißt: höchstens BREITE x HOEHE als WebP, als data-URL im Steckbrief; je Aufgabe Vorher (was das Modell sah,
markiert wie in der Bildbearbeitung) und Nachher (sein Ergebnis) und die Dauer im Dienst beim Rechnen des Portfolios.
"""

from __future__ import annotations

import base64
import io
import json
import tomllib
from pathlib import Path

from PIL import Image

from .bildformat import Aufgabe, Aufgabenvorlage, Beispiel

ORDNER = Path(__file__).resolve().parent.parent / "docs" / "beispiele"
#: Größe der Bilder im Steckbrief (Seitenverhältnis bleibt); Qualität der WebP-Fassung
BREITE, HOEHE, QUALITAET = 384, 256, 78


def data_url(datei: Path) -> str:
    """Ein Bild verkleinert als WebP in einer data-URL."""
    bild = Image.open(datei).convert("RGB")
    bild.thumbnail((BREITE, HOEHE), Image.LANCZOS)
    puffer = io.BytesIO()
    bild.save(puffer, format="WEBP", quality=QUALITAET)
    return "data:image/webp;base64," + base64.b64encode(puffer.getvalue()).decode("ascii")


def motivtitel(ordner: Path = ORDNER) -> dict[str, str]:
    """Kurze Titel der Motive je Aufgabe aus der Beschreibung des Portfolios (ohne: leer)."""
    try:
        roh = tomllib.loads((ordner / "beispiele.toml").read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {name: str(werte.get("titel", "")) for name, werte in roh.get("aufgabe", {}).items()}


def beispiel(aufgabe: Aufgabe, dienst: str, ordner: Path = ORDNER) -> Beispiel | None:
    """Das Beispiel eines Dienstes für eine Aufgabe, wenn das Portfolio es hat."""
    nachher, angaben = ordner / aufgabe.value / f"{dienst}.webp", ordner / aufgabe.value / f"{dienst}.json"
    vorher = ordner / "vorlagen" / f"{aufgabe.value}.webp"
    if not (nachher.is_file() and vorher.is_file()):
        return None
    try:
        dauer = json.loads(angaben.read_text(encoding="utf-8")).get("dauer_s")
    except (OSError, ValueError):
        dauer = None
    return Beispiel(motiv=motivtitel(ordner).get(aufgabe.value, ""), vorher=data_url(vorher), nachher=data_url(nachher), dauer_s=dauer)


def mit_beispielen(aufgaben: dict[Aufgabe, Aufgabenvorlage], dienst: str, ordner: Path = ORDNER) -> dict[Aufgabe, Aufgabenvorlage]:
    """Die Aufgabenvorlagen eines Dienstes, je Aufgabe mit ihrem Beispiel, soweit das Portfolio eines hat."""
    return {a: v.model_copy(update={"beispiel": beispiel(a, dienst, ordner)}) for a, v in aufgaben.items()}
