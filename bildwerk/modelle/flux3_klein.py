"""Dienst bild-erzeugen: FLUX 3 Image Klein von Black Forest Labs über mflux, Bilder aus Text in 16 Schritten (ATL-47 des
Ateliers). Das Modell kann auch nach Vorlagen bearbeiten; das bleibt zurückgestellt (Nutzer 07.11.2026), weil es dafür
statt kurzer Anweisungen eine vollständige Beschreibung des Zielbilds braucht und sein Aufbereiter in mflux nicht läuft.

5,6 Mrd. Parameter im Bildteil, Qwen3-VL-4B als Textkodierer; destilliert, 16 Schritte, keine Führung. Am besten mit
vollständigen Szenenbeschreibungen (Bildausschnitt, Gegenstand, Hintergrund, Licht).

Lizenz: FLUX Kommunity License v1.0: nicht-kommerziell frei, kommerzielle Bilder nur für Nutzer unter 5 Mio. US-Dollar
Jahresumsatz; der Textkodierer steht unter Apache-2.0. Die Gewichte gibt es auf Hugging Face erst nach Zustimmung.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Bildauftrag, Groessenregel, Modellangaben, Schrittregel
from ..eigene_ablage import liste_zu, ort
from ..eigene_ablage import lesen as ablage_lesen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung
from .moss_image import Abbruchwache

PORT = 8803
#: Name des Dienstes im Maschinenraum und für die Leitstelle: die Fähigkeit (HELF-189)
DIENST = "bild-erzeugen"
#: das Original (wie der gepinnte Stand von mflux es verlangt)
PRUEFLISTE_QUELLE = Path(__file__).with_name("flux3_klein.json")
#: die eigene Ablage (MR-4) je Bitbreite; nur daraus lädt der Dienst, in der Bitbreite seiner Einstellung (Vorgabe 8 Bit)
BASIS, BITS = "flux3-image-klein", 8
LOKAL = ort(BASIS, BITS)
PRUEFLISTE = liste_zu(Path(__file__), "flux3_klein", BITS)

ANGABEN = Modellangaben(
    kennung="flux3-klein",
    name="FLUX 3 Image Klein",
    aliase=[DIENST, "black-forest-labs/flux-3-image-klein", "flux3-image-klein"],
    zweck="Bilder aus Text in 16 Schritten, auch mit Schrift im Bild und Anordnung über Rahmen; am besten mit vollständigen "
          "Szenenbeschreibungen",
    kann_bearbeiten=False,
    bearbeiten_grund="Bearbeiten nach Vorlage ist zurückgestellt: es braucht eine vollständige Beschreibung des Zielbilds.",
    eignung=[],
    sprachen=["en"],
    parameter_mrd=5.6,
    textkodierer="Qwen3-VL-4B",
    lizenz="FLUX Kommunity License v1.0",
    lizenz_hinweis="Nicht-kommerziell frei; kommerziell nur für Nutzer unter 5 Mio. US-Dollar Jahresumsatz.",
    erschienen=date(2026, 10, 15),
    # mflux: Kanten durch 16; ausgelegt auf rund 1 Megapixel
    groessen=Groessenregel(raster=16, kante_min=256, kante_max=2048, pixel_min=512 * 512, pixel_max=2048 * 2048, vorgabe="1024x1024",
                           empfohlen=["1024x1024", "1344x768", "768x1344", "1152x896", "896x1152"]),
    # destilliert auf 16 Schritte, ohne Führung
    schritte=Schrittregel(vorgabe=16, kleinste=1, groesste=50),
    guidance=1.0,
    beschreibung_max_token=512,
)


def herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster, wie der gepinnte Stand von mflux sie für FLUX 3 Image Klein verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.flux3.weights.flux3_weight_definition import Flux3KleinWeightDefinition

    repo = ModelConfig.flux3_image_klein().model_name
    return repo, ModelRevisions.get(repo), Flux3KleinWeightDefinition.get_download_patterns()


def bauen(bits: int = BITS) -> Any:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux3.variants.txt2img.flux3_klein import Flux3Klein

    return Flux3Klein(quantize=bits, model_config=ModelConfig.flux3_image_klein())


def aus_ablage(ordner: str) -> Any:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux3.variants.txt2img.flux3_klein import Flux3Klein

    return Flux3Klein(model_path=ordner, model_config=ModelConfig.flux3_image_klein())


def umwandeln(original_loeschen: bool, bits: int = BITS) -> int:
    """Die eigene Ablage einer Bitbreite bauen (bildwerk umwandeln flux3-klein [--bits 4]); das Original bleibt (MR-4)."""
    from ..eigene_ablage import umwandeln as bauen_und_speichern

    if original_loeschen:
        print("Quellen und Duplikate räumen wir später gemeinsam ab (MR-4); das Original bleibt.")
    return bauen_und_speichern(ANGABEN.kennung, Pruefliste.lesen(PRUEFLISTE_QUELLE), liste_zu(Path(__file__), "flux3_klein", bits),
                               ort(BASIS, bits), lambda: bauen(bits), aus_ablage)


class Motor:
    """Hält Flux3Klein aus mflux; geladen wird nur aus der eigenen Ablage in der Bitbreite der Einstellung."""

    angaben = ANGABEN

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.quantisierung: Quantisierung = einstellungen.quantisierung
        bits = self.quantisierung.bits or BITS
        self.pruefliste = ablage_lesen(liste_zu(Path(__file__), "flux3_klein", bits), "black-forest-labs/flux-3-image-klein",
                                       ANGABEN.lizenz, ort(BASIS, bits))
        self.m: Any = None
        self._wache = Abbruchwache()

    def gewichte(self) -> Gewichtsstand:
        stand = pruefen(self.pruefliste, ablage())
        if not self.pruefliste.dateien:
            stand.fehlen.append("eigene Ablage (noch nicht umgewandelt)")
        return stand

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Von Hand: bildwerk gewichte {ANGABEN.kennung} --laden, "
                                 f"dann bildwerk umwandeln {ANGABEN.kennung} --bits {self.quantisierung.bits or BITS}")
        self.m = aus_ablage(stand.ordner)
        self.m.callbacks.register(self._wache)

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        if auftrag.vorlagen:
            raise ValueError(ANGABEN.bearbeiten_grund)
        self._wache.signal = abbruch
        try:
            bilder = []
            for seed in auftrag.seeds:
                if abbruch.is_set():
                    raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                ergebnis = self.m.generate_image(seed=seed, prompt=auftrag.beschreibung, num_inference_steps=auftrag.schritte,
                                                 width=auftrag.breite, height=auftrag.hoehe)
                bilder.append(ergebnis.image)
            return bilder
        finally:
            self._wache.signal = None
            if hasattr(self.m, "prompt_cache"):
                self.m.prompt_cache.clear()                   # mflux behält jede Textkodierung; im Dauerbetrieb wüchse das
