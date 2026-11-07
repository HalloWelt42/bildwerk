"""Gemeinsame Teile der Dienste zum Entfernen (bild-entfernen-schnell, bild-entfernen): die markierte Fläche aus der
Signalfarbe lesen, einen Ausschnitt um sie herum wählen und das Gefüllte nur dort zurücksetzen, wo markiert ist.

Das Atelier markiert die zu entfernende Fläche deckend in der Signalfarbe der Aufgabenvorlage und rechnet die Vorlage
weich auf die Rechengröße um; an den Rändern mischt sich die Signalfarbe deshalb mit dem Bild. Jedes Modell rechnet nur
einen Ausschnitt; alles außerhalb der Maske bleibt Pixel für Pixel.
"""

from __future__ import annotations

from PIL import Image, ImageFilter

#: Abstand eines Pixels von der Signalfarbe (RGB, euklidisch), bis zu dem es als markiert gilt
NAEHE = 90
#: so weich wird das Gefüllte an der Kante der Maske zurückgesetzt (Pixel)
WEICH = 2
#: Umgebung um die Maske im Ausschnitt: so viel ihrer größeren Kante je Seite, mindestens so viele Pixel
UMGEBUNG_ANTEIL, UMGEBUNG_MIN = 0.6, 48


def maske_aus_signalfarbe(bild: Image.Image, farbe: tuple[int, int, int], naehe: int = NAEHE, saum: int = 3) -> Image.Image:
    """Die markierte Fläche (Pixel nahe der Signalfarbe), um `saum` Pixel gewachsen, als Graustufenbild 0/255."""
    import numpy as np

    px = np.asarray(bild.convert("RGB"), dtype=np.int32)
    abstand = np.sqrt(((px - np.asarray(farbe, dtype=np.int32)) ** 2).sum(axis=2))
    maske = Image.fromarray(np.where(abstand <= naehe, 255, 0).astype("uint8"), "L")
    return wachsen(maske, saum)


def wachsen(maske: Image.Image, saum: int) -> Image.Image:
    """Die Maske um `saum` Pixel nach außen wachsen lassen (in Schritten, damit große Säume schnell bleiben)."""
    while saum > 0:
        schritt = min(saum, 8)
        maske = maske.filter(ImageFilter.MaxFilter(2 * schritt + 1))
        saum -= schritt
    return maske


def ausschnitt(maske: Image.Image, anteil: float = UMGEBUNG_ANTEIL, minimum: int = UMGEBUNG_MIN) -> tuple[int, int, int, int] | None:
    """Quadratischer Ausschnitt um die Maske mit Umgebung, im Bild gehalten; None, wenn nichts markiert ist."""
    kasten = maske.getbbox()
    if not kasten:
        return None
    x0, y0, x1, y1 = kasten
    kante = max(x1 - x0, y1 - y0)
    k = min(max(maske.width, maske.height), kante + 2 * max(minimum, round(kante * anteil)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax = min(max(0, cx - k // 2), max(0, maske.width - k))
    ay = min(max(0, cy - k // 2), max(0, maske.height - k))
    return ax, ay, min(maske.width, ax + k), min(maske.height, ay + k)


def zurueckmischen(original: Image.Image, gefuellt: Image.Image, maske: Image.Image, kasten: tuple[int, int, int, int],
                   weich: int = WEICH) -> Image.Image:
    """Den gefüllten Ausschnitt nur dort ins Original setzen, wo markiert ist (Kante weich)."""
    ergebnis = original.convert("RGB").copy()
    breite, hoehe = kasten[2] - kasten[0], kasten[3] - kasten[1]
    teil = gefuellt.convert("RGB").resize((breite, hoehe), Image.LANCZOS)
    teilmaske = maske.crop(kasten)
    ergebnis.paste(teil, kasten[:2], teilmaske.filter(ImageFilter.GaussianBlur(weich)) if weich else teilmaske)
    return ergebnis
