"""Dienst qwen-image-edit-schnell: Qwen-Image-Edit-2511 mit den Lightning-Gewichten von lightx2v (ATL-21 des Ateliers).

Dieselben Grundgewichte wie qwen-image-edit (eigene 8-Bit-Ablage), dazu ein LoRA von 0,85 GB, das beim Laden
eingerechnet wird: 8 Schritte statt 40 und Stärke 1, also ohne den zweiten Durchgang für die Gegenanweisung; das ist
rund ein Zehntel der Rechenzeit. Mit hoher Stärke melden Nutzer Artefakte, deshalb fest 1. Ob die Qualität reicht,
zeigt die Messung über die Leitstelle (docs/messungen/qwen-image-edit-schnell.json).

Lizenz: Apache-2.0 (Grundmodell und LoRA), die Bilder sind frei nutzbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..bildformat import Schrittregel
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Einstellungen
from . import qwen_image_edit as basis

PORT = 8796
#: ein Zusatz (LoRA) auf die Gewichte dieses Modells: die Prüfliste hier nennt nur ihn
ZUSATZ_ZU = "qwen-image-edit"
#: das LoRA aus Hugging Face (die Grundgewichte prüft das Grundmodul)
PRUEFLISTE = Path(__file__).with_name("qwen_image_edit_schnell.json")

ANGABEN = basis.ANGABEN.model_copy(update={
    "kennung": "qwen-image-edit-schnell",
    "name": "Qwen-Image-Edit-2511 schnell",
    "aliase": ["qwen-image-edit-2511-lightning", "qwen-edit-schnell"],
    "zweck": "Bilder nach Anweisung bearbeiten wie Qwen-Image-Edit-2511, mit Lightning in 8 Schritten (rund ein Zehntel der Zeit)",
    "eignung": ["schnell"],
    # Lightning ist auf 8 Schritte gelernt; weniger oder mehr verschlechtert das Bild
    "schritte": Schrittregel(vorgabe=8, kleinste=8, groesste=8),
    "guidance": 1.0,
})


class OhneGegendurchgang:
    """Bei Stärke 1 hängt ein Schritt nicht von der Gegenanweisung ab: mflux rechnet neg + 1 * (pos - neg) = pos und
    skaliert auf die Norm von pos zurück, also genau pos. Trotzdem ruft es den Bildteil je Schritt zweimal auf (erst
    positiv, dann negativ); das kostet die Hälfte der Zeit. Dieser Vorsatz rechnet nur den ersten Aufruf jedes Schritts
    und gibt ihn beim zweiten zurück. `neu()` vor jedem Bild, damit die Paare stimmen."""

    def __init__(self, bildteil: Any) -> None:
        self._bildteil, self._zaehler, self._letzter = bildteil, 0, None

    def neu(self) -> None:
        self._zaehler, self._letzter = 0, None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._zaehler += 1
        if self._zaehler % 2 == 0 and self._letzter is not None:
            return self._letzter
        self._letzter = self._bildteil(*args, **kwargs)
        return self._letzter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bildteil, name)


def mflux_herkunft() -> tuple[str, str, list[str]]:
    liste = Pruefliste.lesen(PRUEFLISTE)
    return liste.repo, liste.revision, [d.pfad for d in liste.dateien]


class Motor(basis.Motor):
    """Wie Qwen-Image-Edit, beim Laden mit dem Lightning-LoRA eingerechnet."""

    angaben = ANGABEN

    def __init__(self, einstellungen: Einstellungen) -> None:
        super().__init__(einstellungen)
        self.lora = Pruefliste.lesen(PRUEFLISTE)

    def gewichte(self) -> Gewichtsstand:
        stand = super().gewichte()
        lora = pruefen(self.lora, ablage())
        stand.fehlen.extend(f"LoRA: {f}" for f in lora.fehlen)
        return stand

    def laden(self) -> None:
        super().laden()
        if self.angaben.guidance == 1.0:                 # nur dann ist der Gegendurchgang wirkungslos
            self.m.transformer = OhneGegendurchgang(self.m.transformer)

    def erzeugen(self, auftrag: Any, abbruch: Any) -> Any:
        if isinstance(self.m.transformer, OhneGegendurchgang):
            self.m.transformer.neu()
        return super().erzeugen(auftrag, abbruch)

    def zusatzgewichte(self) -> dict[str, Any]:
        datei = self.lora.ordner(ablage()) / self.lora.dateien[0].pfad
        return {"lora_paths": [str(datei)], "lora_scales": [1.0]}
