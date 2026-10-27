"""Die gemeinsame Schnittstelle aller Bilddienste im üblichen Bildformat: Anfrage, Antwort, Steckbrief.

Jeder Dienst des Bildwerks spricht genau diese Formen, gleich welches Modell er hält. So braucht die Leitstelle einen
einzigen Bildweg für alle Bildmodelle; was ein Modell kann (Größen, Schritte, Bearbeiten), sagt sein Steckbrief.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Größter Startwert: in JSON als ganze Zahl verlustfrei auch für Leser, die Zahlen als Gleitkomma lesen (wie die Leitstelle)
SEED_GRENZE = 2**53
ANZAHL_MAX = 4
GROESSE = re.compile(r"(\d{1,5})x(\d{1,5})")


class Groessenregel(BaseModel):
    """Welche Bildgrößen ein Modell annimmt: Kanten im Raster, je Kante und für die Fläche eine Grenze."""

    raster: int = Field(gt=0, description="jede Kante ist ein Vielfaches davon")
    kante_min: int
    kante_max: int
    pixel_min: int
    pixel_max: int
    vorgabe: str
    empfohlen: list[str] = Field(default_factory=list, description="Größen, in denen das Modell am besten zeichnet")

    def kanten(self, groesse: str) -> tuple[int, int]:
        """Breite und Höhe aus "BxH"; ValueError mit dem Grund, wenn das Modell die Größe nicht annimmt."""
        t = GROESSE.fullmatch(groesse.strip())
        if not t:
            raise ValueError(f"Größe als BreitexHöhe angeben, zum Beispiel {self.vorgabe}, nicht {groesse!r}.")
        breite, hoehe = int(t.group(1)), int(t.group(2))
        if breite % self.raster or hoehe % self.raster:
            raise ValueError(f"Breite und Höhe müssen durch {self.raster} teilbar sein, nicht {breite}x{hoehe}.")
        if not all(self.kante_min <= k <= self.kante_max for k in (breite, hoehe)):
            raise ValueError(f"Jede Kante {self.kante_min} bis {self.kante_max}, nicht {breite}x{hoehe}.")
        if not self.pixel_min <= breite * hoehe <= self.pixel_max:
            raise ValueError(f"Die Fläche muss zwischen {self.pixel_min} und {self.pixel_max} Pixeln liegen, "
                             f"{breite}x{hoehe} hat {breite * hoehe}.")
        return breite, hoehe


class Schrittregel(BaseModel):
    vorgabe: int
    kleinste: int
    groesste: int


class Modellangaben(BaseModel):
    """Was ein Bildmodell ist und kann; fest je Modell, von Hand nach Modellkarte und Rechenwerk."""

    kennung: str = Field(description="Kurzname des Modells; als `model` in Anfragen")
    name: str
    aliase: list[str] = Field(default_factory=list, description="weitere Namen, die als `model` dasselbe meinen")
    zweck: str
    faehigkeiten: list[str] = Field(default_factory=lambda: ["bild_erzeugen"])
    kann_bearbeiten: bool = False
    bearbeiten_grund: str = Field(default="", description="warum das Modell nicht bearbeitet (Antwort 501)")
    eignung: list[str] = Field(default_factory=list, description="Wörter der Leitstelle: schnell, gruendlich, ...")
    sprachen: list[str] = Field(default_factory=list, description="Sprachen der Beschreibung, die das Modell gut versteht (ISO 639-1)")
    parameter_mrd: float = Field(description="Parameter des Bildteils (Transformer) in Milliarden")
    textkodierer: str
    lizenz: str
    erschienen: date
    groessen: Groessenregel
    schritte: Schrittregel
    guidance: float = Field(description="Stärke der Führung durch die Beschreibung (CFG), wie die Modellkarte sie vorgibt")
    beschreibung_max_token: int

    def meint(self, model: str) -> bool:
        return model in (self.kennung, self.name, *self.aliase)


class Quelle(BaseModel):
    repo: str
    revision: str


class Rechenwerk(BaseModel):
    name: str = "mflux"
    fassung: str
    commit: str | None = None


class Steckbrief(Modellangaben):
    """Die Angaben des Modells und wie dieser Dienst es hält."""

    quelle: Quelle
    quantisierung: str
    rechenwerk: Rechenwerk
    anzahl_max: int = ANZAHL_MAX
    zustand: str


class Modelleintrag(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bildwerk"
    steckbrief: Steckbrief


class Modellliste(BaseModel):
    object: str = "list"
    data: list[Modelleintrag]


class Bildanfrage(BaseModel):
    """`POST /v1/images/generations`. Unbekannte Felder des üblichen Formats (quality, style, user) werden überlesen."""

    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1, max_length=20000)
    size: str | None = Field(default=None, description="BreitexHöhe; fehlt sie, gilt die Vorgabe des Modells")
    n: int = Field(default=1, ge=1, le=ANZAHL_MAX)
    seed: int | None = Field(default=None, ge=0, lt=SEED_GRENZE, description="Bild i rechnet mit seed + i; fehlt er, wird einer gezogen")
    steps: int | None = Field(default=None, description="Schritte; fehlen sie, gilt die Vorgabe des Modells")
    model: str | None = None
    response_format: str = "b64_json"

    @field_validator("prompt")
    @classmethod
    def _nicht_leer(cls, wert: str) -> str:
        if not wert.strip():
            raise ValueError("Die Beschreibung ist leer.")
        return wert

    @field_validator("response_format")
    @classmethod
    def _format(cls, wert: str) -> str:
        if wert != "b64_json":
            raise ValueError("Bilder kommen nur als b64_json (PNG), nie als Adresse.")
        return wert


class Bilddaten(BaseModel):
    b64_json: str
    revised_prompt: str | None = None


class Bildzusatz(BaseModel):
    modell: str
    seed: int                         # Startwert des ersten Bildes
    seeds: list[int]                  # je Bild
    schritte: int
    groesse: str
    dauer_ms: int                     # Rechenzeit im Dienst, ohne Warten auf die Sperre


class Bildantwort(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    created: int
    data: list[Bilddaten]
    x_bildwerk: Bildzusatz = Field(alias="x-bildwerk")


class Bildauftrag(BaseModel):
    """Was der Motor eines Modells rechnet: geprüft, mit festen Startwerten je Bild."""

    beschreibung: str
    breite: int
    hoehe: int
    seeds: list[int]
    schritte: int
