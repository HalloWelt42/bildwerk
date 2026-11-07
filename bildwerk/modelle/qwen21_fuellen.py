"""Dienst bild-fuellen: Qwen-Image 2.1 Turbo mit dem ControlNet-Union von alibaba-pai über mflux, Füllen mit echter Maske
(ATL-47 des Ateliers). Was markiert ist, rechnet das Modell neu (nach Anweisung: etwas hinein oder nur Hintergrund, also
Entfernen); alles andere bleibt. Damit kann es Entfernen und Füllen, ohne dass sich Bereiche außerhalb der Markierung
verändern (anders als Bearbeiten nach Anweisung).

Ansteuerung wie `mflux-generate-qwen-2.1-controlnet --image-path ... --mask-image ...`: Quelle und Maske (weiß neu,
schwarz bleibt), ohne Kontrollbild, Turbo in 8 Schritten ohne Führung. Gerechnet wird ein Ausschnitt um die Markierung
auf der Arbeitsgröße; eingesetzt wird nur in der Markierung (wie bei den anderen Entfernen-Diensten).

Schnittstelle wie jeder Bilddienst: `/v1/images/edits` mit genau einer Vorlage, die Fläche deckend in der Signalfarbe
der Aufgabenvorlage (`qwen21_fuellen.aufgaben.toml`); die Anweisung setzt die Anwendung aus der Vorlage zusammen.

Lizenz: Qwen Research License (Modell und ControlNet): nur nichtkommerziell, für Forschung und Erprobung.
"""

from __future__ import annotations

import io
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Aufgabe, Bildauftrag, Groessenregel, Modellangaben, Schrittregel, aufgaben_lesen
from ..eigene_ablage import liste_zu, ort
from ..eigene_ablage import lesen as ablage_lesen
from ..entfernen import ausschnitt, maske_aus_signalfarbe, wachsen, zurueckmischen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung
from .moss_image import Abbruchwache

PORT = 8802
#: Name des Dienstes im Maschinenraum und für die Leitstelle: die Fähigkeit (HELF-189)
DIENST = "bild-fuellen"
#: das Original (wie der gepinnte Stand von mflux es verlangt) und, als Zusatz, das ControlNet-Union, das die Maske liest
PRUEFLISTE_QUELLE = Path(__file__).with_name("qwen21_fuellen.json")
ZUSATZ_PRUEFLISTEN = [Path(__file__).with_name("qwen21_fuellen_controlnet.json")]
#: die eigene Ablage (MR-4) je Bitbreite, samt ControlNet unter controlnet/; nur daraus lädt der Dienst, in der
#: Bitbreite seiner Einstellung (BILDWERK_QUANTISIERUNG, Vorgabe 8 Bit)
BASIS, BITS = "qwen-image-2.1-turbo-fuellen", 8
LOKAL = ort(BASIS, BITS)
PRUEFLISTE = liste_zu(Path(__file__), "qwen21_fuellen", BITS)
AUFGABEN = Path(__file__).with_name("qwen21_fuellen.aufgaben.toml")

#: Arbeitsgröße des Ausschnitts (Kanten im Raster von 16)
ARBEIT = 1024
#: um so viele Pixel wächst die Markierung vor dem Rechnen, damit keine Reste am Rand bleiben
SAUM = 6

ANGABEN = Modellangaben(
    kennung="qwen21-fuellen",
    name="Qwen-Image 2.1 Turbo mit Maske",
    aliase=[DIENST, "Qwen/Qwen-Image-2.1-Turbo", "qwen-image-2.1-turbo-controlnet"],
    zweck="Markierte Flächen nach Anweisung neu füllen oder entfernen; außerhalb der Markierung bleibt alles, 8 Schritte",
    kann_bearbeiten=True,
    vorlagen_min=1,
    vorlagen_max=1,
    eignung=[],
    sprachen=["en", "zh"],
    parameter_mrd=7.0,
    textkodierer="Qwen3-VL-8B",
    lizenz="Qwen Research License",
    lizenz_hinweis="Nur nicht-kommerzielle Nutzung (Forschung und Erprobung).",
    erschienen=date(2026, 9, 20),
    groessen=Groessenregel(raster=8, kante_min=64, kante_max=4096, pixel_min=64 * 64, pixel_max=4096 * 4096, vorgabe="1024x1024"),
    schritte=Schrittregel(vorgabe=8, kleinste=8, groesste=8),
    guidance=1.0,
    beschreibung_max_token=512,
    aufgaben=aufgaben_lesen(AUFGABEN),
)


def herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster, wie der gepinnte Stand von mflux sie für Qwen-Image 2.1 Turbo verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

    repo = ModelConfig.qwen_image_21_turbo_controlnet_union().model_name
    return repo, ModelRevisions.get(repo), QwenImage21WeightDefinition.get_download_patterns()


def herkunft_controlnet() -> tuple[str, str]:
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions

    repo = ModelConfig.qwen_image_21_turbo_controlnet_union().controlnet_model
    return repo, ModelRevisions.get(repo)


def bauen(bits: int = BITS) -> Any:
    """Qwen-Image 2.1 Turbo mit ControlNet aus dem Original, quantisiert (für die eigene Ablage)."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet

    zusatz = Pruefliste.lesen(ZUSATZ_PRUEFLISTEN[0])
    return QwenImage21Controlnet(quantize=bits, model_config=ModelConfig.qwen_image_21_turbo_controlnet_union(),
                                 controlnet_path=str(zusatz.ordner(ablage()) / zusatz.dateien[0].pfad))


def aus_ablage(ordner: str) -> Any:
    """Aus der eigenen Ablage: das ControlNet liegt dort unter controlnet/ und kommt mit."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet

    return QwenImage21Controlnet(model_path=ordner, model_config=ModelConfig.qwen_image_21_turbo_controlnet_union())


def umwandeln(original_loeschen: bool, bits: int = BITS) -> int:
    """Die eigene Ablage einer Bitbreite bauen (bildwerk umwandeln qwen21-fuellen [--bits 4]); das Original bleibt (MR-4)."""
    from ..eigene_ablage import umwandeln as bauen_und_speichern

    if original_loeschen:
        print("Quellen und Duplikate räumen wir später gemeinsam ab (MR-4); das Original bleibt.")
    return bauen_und_speichern(ANGABEN.kennung, Pruefliste.lesen(PRUEFLISTE_QUELLE), liste_zu(Path(__file__), "qwen21_fuellen", bits),
                               ort(BASIS, bits), lambda: bauen(bits), aus_ablage)


class Motor:
    """Hält QwenImage21Controlnet aus mflux; geladen wird nur aus der eigenen Ablage in der Bitbreite der Einstellung."""

    angaben = ANGABEN

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.quantisierung: Quantisierung = einstellungen.quantisierung
        bits = self.quantisierung.bits or BITS
        self.pruefliste = ablage_lesen(liste_zu(Path(__file__), "qwen21_fuellen", bits), "Qwen/Qwen-Image-2.1-Turbo", ANGABEN.lizenz,
                                       ort(BASIS, bits))
        self.signalfarbe: dict[Aufgabe, tuple[int, int, int]] = {
            a: tuple(v.signalfarbe) for a, v in ANGABEN.aufgaben.items() if v.signalfarbe}  # type: ignore[misc]
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
        if len(auftrag.vorlagen) != 1:
            raise ValueError("Füllen rechnet mit genau einer Vorlage")
        bild = Image.open(io.BytesIO(auftrag.vorlagen[0])).convert("RGB")
        if bild.size != (auftrag.breite, auftrag.hoehe):
            bild = bild.resize((auftrag.breite, auftrag.hoehe), Image.LANCZOS)
        farbe = next(iter(self.signalfarbe.values()), (255, 0, 255))
        maske = wachsen(maske_aus_signalfarbe(bild, farbe), SAUM)
        kasten = ausschnitt(maske)
        if kasten is None:
            raise ValueError(f"Keine markierte Fläche: die Fläche muss deckend in der Signalfarbe {farbe} stehen")
        teil = bild.crop(kasten).resize((ARBEIT, ARBEIT), Image.LANCZOS)
        teilmaske = maske.crop(kasten).resize((ARBEIT, ARBEIT), Image.NEAREST)
        teil.paste((128, 128, 128), (0, 0, ARBEIT, ARBEIT), teilmaske)     # die Signalfarbe nicht ins Modell geben
        self._wache.signal = abbruch
        try:
            bilder = []
            for seed in auftrag.seeds:
                if abbruch.is_set():
                    raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                ergebnis = self.m.generate_image(seed=seed, prompt=auftrag.beschreibung, num_inference_steps=auftrag.schritte,
                                                 width=ARBEIT, height=ARBEIT, guidance=ANGABEN.guidance,
                                                 image_path=teil, mask_image=teilmaske.convert("RGB"))
                bilder.append(zurueckmischen(bild, ergebnis.image.convert("RGB"), maske, kasten))
            return bilder
        finally:
            self._wache.signal = None
            if hasattr(self.m, "prompt_cache"):
                self.m.prompt_cache.clear()                   # mflux behält jede Textkodierung; im Dauerbetrieb wüchse das
