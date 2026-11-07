"""Dienst bild-entfernen-schnell: LaMa (big-lama) als Core-ML-Paket, Entfernen in unter einer Sekunde (ATL-33 des Ateliers).

LaMa füllt eine markierte Fläche mit dem, was dahinter läge (Hintergründe, Wiederholmuster, Personen und Masten weg);
Neues erfindet es nicht, dafür sind Qwen-Image-Edit und seine schnelle Fassung da. Der Dienst heißt nach seiner
Fähigkeit (Entfernen) und Eigenschaft (schnell); das Modell behält im Steckbrief seinen Namen.

Schnittstelle wie jeder Bilddienst: `/v1/images/edits` mit genau einer Vorlage. Die zu entfernende Fläche steht in der
Vorlage deckend in der Signalfarbe aus der Aufgabenvorlage (`lama.aufgaben.toml`), so wie bei Qwen; der Dienst liest die
Maske daraus, rechnet einen Ausschnitt um sie herum auf der Arbeitsgröße des Pakets (800 x 800) und setzt nur die
markierte Fläche weich zurück. Alles andere bleibt Pixel für Pixel. Die Anweisung braucht LaMa nicht.

Gewichte: das Paket `LaMa.mlpackage` (fp32, Umwandlung des originalen big-lama, Apache-2.0) aus Hugging Face, von
`bildwerk umwandeln lama` einmal für diesen Mac übersetzt (`.mlmodelc`) in die eigene Ablage; der Dienst lädt nur das
Übersetzte. Das Original kommt erst nach einer Messung über die Leitstelle in den Papierkorb (wie bei Qwen).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from ..bildformat import Aufgabe, Bildauftrag, Groessenregel, Modellangaben, Rechenwerk, Schrittregel, aufgaben_lesen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, aus_ordner, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen
from .qwen_image_edit import original_in_papierkorb

PORT = 8797
#: Name des Dienstes im Maschinenraum und für die Leitstelle: Fähigkeit und Eigenschaft (HELF-189)
DIENST = "bild-entfernen-schnell"
PRUEFLISTE = Path(__file__).with_name("lama.json")
PRUEFLISTE_QUELLE = Path(__file__).with_name("lama_quelle.json")
LOKAL = "bild/coreml/lama-big-800.mlmodelc"
AUFGABEN = Path(__file__).with_name("lama.aufgaben.toml")
MESSUNG = Path(__file__).resolve().parents[2] / "docs" / "messungen" / "lama.json"

#: Arbeitsgröße des Pakets (fest); größere und kleinere Ausschnitte werden darauf gebracht
ARBEIT = 800
#: Abstand eines Pixels von der Signalfarbe (RGB, euklidisch), bis zu dem es als markiert gilt: das Atelier rechnet die
#: Vorlage weich auf die Rechengröße um, an den Rändern der Fläche mischt sich die Signalfarbe mit dem Bild
NAEHE = 90
#: um so viele Pixel wächst die Maske (Saum der weich umgerechneten Kante), dann so weich wird sie zurückgesetzt
SAUM, WEICH = 3, 2
#: Umgebung um die Maske im Ausschnitt: so viel ihrer größeren Kante je Seite, mindestens so viele Pixel
UMGEBUNG_ANTEIL, UMGEBUNG_MIN = 0.6, 48

ANGABEN = Modellangaben(
    kennung="lama",
    name="LaMa (big)",
    aliase=["big-lama", "lama-big", DIENST, "mlboydaisuke/LaMa-CoreML"],
    zweck="Markierte Flächen entfernen und mit dem füllen, was dahinter läge, in unter einer Sekunde; erfindet nichts Neues",
    kann_bearbeiten=True,
    vorlagen_min=1,
    vorlagen_max=1,
    eignung=["schnell"],
    sprachen=[],
    # big-lama: 51 Millionen Parameter (Gewichte fp32, 216 MB)
    parameter_mrd=0.051,
    textkodierer="keiner",
    lizenz="Apache-2.0",
    lizenz_hinweis="Apache-2.0, frei nutzbar.",
    erschienen=date(2021, 9, 15),
    # Rechengröße frei (gerechnet wird ein Ausschnitt auf 800 x 800); Raster 8 wie bei den anderen Bildwegen üblich
    groessen=Groessenregel(raster=8, kante_min=64, kante_max=4096, pixel_min=64 * 64, pixel_max=4096 * 4096, vorgabe="1024x1024"),
    schritte=Schrittregel(vorgabe=1, kleinste=1, groesste=1),
    guidance=1.0,
    beschreibung_max_token=0,
    # LaMa rechnet immer gleich: eine Variante
    anzahl_max=1,
    aufgaben=aufgaben_lesen(AUFGABEN),
)


def maske_aus_signalfarbe(bild: Image.Image, farbe: tuple[int, int, int], naehe: int = NAEHE, saum: int = SAUM) -> Image.Image:
    """Die markierte Fläche (Pixel nahe der Signalfarbe), um `saum` Pixel gewachsen, als Graustufenbild 0/255."""
    import numpy as np

    px = np.asarray(bild.convert("RGB"), dtype=np.int32)
    abstand = np.sqrt(((px - np.asarray(farbe, dtype=np.int32)) ** 2).sum(axis=2))
    maske = Image.fromarray(np.where(abstand <= naehe, 255, 0).astype("uint8"), "L")
    return maske.filter(ImageFilter.MaxFilter(2 * saum + 1)) if saum else maske


def ausschnitt(maske: Image.Image) -> tuple[int, int, int, int] | None:
    """Quadratischer Ausschnitt um die Maske mit Umgebung, im Bild gehalten; None, wenn nichts markiert ist."""
    kasten = maske.getbbox()
    if not kasten:
        return None
    x0, y0, x1, y1 = kasten
    kante = max(x1 - x0, y1 - y0)
    k = min(max(maske.width, maske.height), kante + 2 * max(UMGEBUNG_MIN, round(kante * UMGEBUNG_ANTEIL)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax = min(max(0, cx - k // 2), max(0, maske.width - k))
    ay = min(max(0, cy - k // 2), max(0, maske.height - k))
    return ax, ay, min(maske.width, ax + k), min(maske.height, ay + k)


def zurueckmischen(original: Image.Image, gefuellt: Image.Image, maske: Image.Image, kasten: tuple[int, int, int, int]) -> Image.Image:
    """Den gefüllten Ausschnitt nur dort ins Original setzen, wo markiert ist (Kante weich)."""
    ergebnis = original.convert("RGB").copy()
    breite, hoehe = kasten[2] - kasten[0], kasten[3] - kasten[1]
    teil = gefuellt.convert("RGB").resize((breite, hoehe), Image.LANCZOS)
    weich = maske.crop(kasten).filter(ImageFilter.GaussianBlur(WEICH)) if WEICH else maske.crop(kasten)
    ergebnis.paste(teil, kasten[:2], weich)
    return ergebnis


def umwandeln(original_loeschen: bool) -> int:
    """Das Paket aus dem Zwischenspeicher für diesen Mac übersetzen (.mlmodelc) in die eigene Ablage, Prüfliste schreiben,
    Probeladen. Mit `original_loeschen` kommt das Original nur nach einer Messung über die Leitstelle in den Papierkorb."""
    import coremltools as ct

    quelle = Pruefliste.lesen(PRUEFLISTE_QUELLE)
    wo = ablage()
    stand = pruefen(quelle, wo)
    if not stand.vollstaendig:
        print(f"{stand.meldung}. Zuerst: bildwerk gewichte {ANGABEN.kennung} --laden")
        return 1
    ziel = wo.parent / LOKAL
    if not ziel.is_dir():
        paket = quelle.ordner(wo) / "LaMa.mlpackage"
        # der Zwischenspeicher hält Verweise auf gemeinsame Dateien; der Übersetzer braucht echte Dateien: kurz kopieren
        with tempfile.TemporaryDirectory(prefix="bildwerk-lama-") as ordner:
            echt = Path(ordner) / "LaMa.mlpackage"
            shutil.copytree(paket, echt, symlinks=False)
            print(f"übersetze nach {ziel} ...", flush=True)
            uebersetzt = Path(ct.utils.compile_model(str(echt)))
            ziel.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(uebersetzt), str(ziel))
    liste = aus_ordner(ziel, quelle, LOKAL)
    liste.schreiben(PRUEFLISTE)
    print(f"{PRUEFLISTE.name}: {len(liste.dateien)} Dateien, {liste.groesse / 1e6:.0f} MB in {ziel}", flush=True)
    ct.models.CompiledMLModel(str(ziel), compute_units=ct.ComputeUnit.ALL)
    print("Probeladen gelungen")
    if original_loeschen:
        if not gerechnet():
            print(f"Original bleibt: erst über die Leitstelle messen (tools/messen.py {ANGABEN.kennung}), dann erneut mit --original-loeschen")
            return 3
        print(f"Original im Papierkorb: {original_in_papierkorb(wo, quelle.repo)}")
    return 0


#: Rechenwerk der Gewichte: kein mflux, sondern ein Core-ML-Paket (die Prüfung der mflux-Teile gilt hier nicht)
RECHENWERK = "coreml"


def herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster des Pakets (nur LaMa.mlpackage, nie Code aus dem Repo)."""
    liste = Pruefliste.lesen(PRUEFLISTE_QUELLE)
    return liste.repo, liste.revision, ["LaMa.mlpackage/*"]


def gerechnet() -> bool:
    """Belegt eine Messung über die Leitstelle, dass der Dienst rechnet (mindestens ein Lauf mit Bild)?"""
    import json
    try:
        laeufe = json.loads(MESSUNG.read_text(encoding="utf-8")).get("laeufe", [])
    except (OSError, ValueError):
        return False
    return any(x.get("vorlage") and x.get("dauer_s", 0) > 0 for x in laeufe)


class Motor:
    """Hält das übersetzte LaMa-Paket; geladen wird nur aus der eigenen Ablage, nie aus dem Netz."""

    angaben = ANGABEN
    #: die Gewichte rechnen nur in voller Genauigkeit (LaMa zerfällt mit fp16, laut Modellkarte der MLX-Fassung)
    quantisierung = "fp32"

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.pruefliste = Pruefliste.lesen(PRUEFLISTE) if PRUEFLISTE.is_file() else Pruefliste(
            repo="mlboydaisuke/LaMa-CoreML", revision="", lizenz=ANGABEN.lizenz, lokal=LOKAL, dateien=[])
        self.m: Any = None
        self.eingaenge: dict[str, str] = {}
        self.signalfarbe: tuple[int, int, int] = tuple(ANGABEN.aufgaben[Aufgabe.ENTFERNEN].signalfarbe or (255, 0, 255))  # type: ignore[assignment]

    def gewichte(self) -> Gewichtsstand:
        stand = pruefen(self.pruefliste, ablage())
        if not self.pruefliste.dateien:
            stand.fehlen.append("eigene Ablage (noch nicht übersetzt)")
        return stand

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Laden und übersetzen von Hand: bildwerk gewichte {ANGABEN.kennung} --laden, "
                                 f"dann bildwerk umwandeln {ANGABEN.kennung}")
        import coremltools as ct

        # die Rechenwerke, mit denen die Umwandlung geprüft ist (Modellkarte: andere können die Zahlen still verändern)
        self.m = ct.models.CompiledMLModel(stand.ordner, compute_units=ct.ComputeUnit.ALL)
        self.eingaenge = {e.name: e.type.WhichOneof("Type") for e in self.m._spec.description.input} if getattr(self.m, "_spec", None) else {}

    def rechenwerk(self) -> Rechenwerk:
        from importlib import metadata
        try:
            return Rechenwerk(name="coremltools", fassung=metadata.version("coremltools"))
        except metadata.PackageNotFoundError:
            return Rechenwerk(name="coremltools", fassung="fehlt")

    def fuellen(self, bild: Image.Image, maske: Image.Image) -> Image.Image:
        """Ein Ausschnitt (Arbeitsgröße) mit Maske (L, 0/255) durch das Modell; liefert den gefüllten Ausschnitt."""
        aus = self.m.predict({"image": bild.convert("RGB"), "mask": maske.convert("L")})
        ergebnis = next(iter(aus.values()))
        if isinstance(ergebnis, Image.Image):
            return ergebnis.convert("RGB")
        import numpy as np
        feld = np.asarray(ergebnis)
        feld = feld.reshape(feld.shape[-3:]) if feld.ndim == 4 else feld
        if feld.shape[0] in (3, 4) and feld.shape[-1] not in (3, 4):
            feld = feld.transpose(1, 2, 0)
        return Image.fromarray(np.clip(feld[..., :3], 0, 255).astype("uint8"), "RGB")

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        if len(auftrag.vorlagen) != 1:
            raise ValueError("LaMa rechnet mit genau einer Vorlage")
        import io
        bild = Image.open(io.BytesIO(auftrag.vorlagen[0])).convert("RGB")
        if bild.size != (auftrag.breite, auftrag.hoehe):
            bild = bild.resize((auftrag.breite, auftrag.hoehe), Image.LANCZOS)
        maske = maske_aus_signalfarbe(bild, self.signalfarbe)
        kasten = ausschnitt(maske)
        if kasten is None:
            raise ValueError(f"Keine markierte Fläche: die zu entfernende Fläche muss deckend in der Signalfarbe {self.signalfarbe} stehen")
        if abbruch.is_set():
            raise Abgebrochen("vor dem Rechnen")
        teil = bild.crop(kasten).resize((ARBEIT, ARBEIT), Image.LANCZOS)
        teilmaske = maske.crop(kasten).resize((ARBEIT, ARBEIT), Image.NEAREST)
        # die Signalfarbe nicht ins Modell geben (falls das Paket die Fläche nicht selbst ausblendet): dort neutral grau
        teil.paste((128, 128, 128), (0, 0, ARBEIT, ARBEIT), teilmaske)
        gefuellt = self.fuellen(teil, teilmaske)
        return [zurueckmischen(bild, gefuellt, maske, kasten)]
