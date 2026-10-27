"""Gemeinsamer Unterbau aller Bildwerk-Dienste.

Jeder Dienst ist ein eigener Prozess mit genau einem Bildmodell. Er lädt es beim Start; solange es lädt, antwortet
`/gesundheit` mit 503, danach mit 200. Laufen heißt damit: das Modell liegt im Speicher. Gestartet und angehalten wird
er vom Maschinenraum, wenn die Leitstelle es verlangt; von selbst startet er nie, und Anwendungen erreichen ihn nur
über die Leitstelle.

Grafik: In einem Prozess rechnet immer nur ein Auftrag zugleich (eine Sperre). Der Zwischenspeicher für Grafikpuffer
von MLX wächst ohne Grenze; er wird begrenzt und nach jeder Rechnung geleert, das Modell bleibt davon unberührt.

Feste Zeitgrenzen für Arbeit gibt es nicht. Geht die Verbindung des Aufrufers zu (die Leitstelle bricht einen Auftrag
ab), endet die Rechnung beim nächsten Schritt; so hält ein verlassener Auftrag die Grafik nicht minutenlang fest.

Probemodus (`--probe`): der Dienst lädt nichts und rechnet nichts, beantwortet aber `/gesundheit` (200, Zustand
`probe`) und `/v1/models`; für den Probelauf der Einrichtung und für Prüfungen ohne Modell.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__

log = logging.getLogger("bildwerk")
ABBRUCH_TAKT_S = 0.5              # so oft wird nachgesehen, ob die Verbindung des Aufrufers noch steht


class Quantisierung(str, Enum):
    """Wie die Gewichte im Speicher liegen; mflux quantisiert beim Laden (Transformer und Textkodierer)."""

    BF16 = "bf16"
    Q8 = "8bit"
    Q6 = "6bit"
    Q5 = "5bit"
    Q4 = "4bit"
    Q3 = "3bit"

    @property
    def bits(self) -> int | None:
        return None if self is Quantisierung.BF16 else int(self.value.removesuffix("bit"))


class Einstellungen(BaseSettings):
    """Je Dienst über Umgebungsvariablen mit der Vorsilbe BILDWERK_ (Dienstdatei im Maschinenraum, `[umgebung]`)."""

    model_config = SettingsConfigDict(env_prefix="BILDWERK_", env_file=None)

    adresse: str = Field(default="127.0.0.1", description="nur dieser Rechner; erreichbar ist der Dienst über die Leitstelle")
    quantisierung: Quantisierung = Field(default=Quantisierung.Q8, description="bf16 oder 8bit, 6bit, 5bit, 4bit, 3bit")
    mlx_zwischenspeicher_gb: float = Field(default=2.0, ge=0, description="Grenze für behaltene Grafikpuffer von MLX")


class Zustand(str, Enum):
    LAEDT = "laedt"
    BEREIT = "bereit"
    GESTOERT = "gestoert"
    PROBE = "probe"                   # Probemodus: antwortet, lädt und rechnet nie


class Gesundheit(BaseModel):
    dienst: str
    fassung: str
    modell: str
    zustand: Zustand
    meldung: str = ""
    gewichte_vollstaendig: bool


class Abgebrochen(RuntimeError):
    """Der Aufrufer ist gegangen; die Rechnung endet beim nächsten Schritt."""


class GewichteFehlen(RuntimeError):
    """Die Gewichte liegen nicht vollständig in der Modellablage; der Dienst lädt nie selbst aus dem Netz."""


class Rechner:
    """Hält das eine Modell des Dienstes und die Sperre, unter der gerechnet wird."""

    def __init__(self, dienst: str, modell: str, einstellungen: Einstellungen, probe: bool = False) -> None:
        self.dienst, self.modell, self.einstellungen = dienst, modell, einstellungen
        self.zustand = Zustand.PROBE if probe else Zustand.LAEDT
        self.meldung = "Probemodus: kein Modell geladen, Rechnen abgelehnt" if probe else ""
        self._sperre = asyncio.Lock()
        self._mlx_begrenzt = False

    def _mlx_begrenzen(self) -> None:
        if self._mlx_begrenzt:
            return
        self._mlx_begrenzt = True
        try:
            import mlx.core as mx
            mx.set_cache_limit(int(self.einstellungen.mlx_zwischenspeicher_gb * 1024**3))
        except Exception as f:
            log.debug("MLX-Zwischenspeicher nicht begrenzbar: %s", f)

    @staticmethod
    def _mlx_leeren() -> None:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    async def laden(self, lader: Callable[[], Any]) -> None:
        self._mlx_begrenzen()
        try:
            await asyncio.to_thread(lader)
            self.zustand, self.meldung = Zustand.BEREIT, ""
            log.info("%s: Modell %s geladen", self.dienst, self.modell)
        except GewichteFehlen as f:
            self.zustand, self.meldung = Zustand.GESTOERT, str(f)[:500]
            log.error("%s: %s", self.dienst, f)
        except Exception as f:
            self.zustand, self.meldung = Zustand.GESTOERT, f"{type(f).__name__}: {f}"[:500]
            log.error("%s: Modell %s nicht ladbar\n%s", self.dienst, self.modell, traceback.format_exc())

    async def rechnen(self, funktion: Callable[..., Any], abbruch: threading.Event, *args: Any) -> Any:
        """Eine Rechnung zugleich, in einem Thread; danach behaltene Grafikpuffer freigeben. `abbruch` beendet sie
        beim nächsten Schritt; wer schon vor dem Start gegangen ist, rechnet gar nicht erst."""
        if self.zustand != Zustand.BEREIT:
            grund = f": {self.meldung}" if self.meldung else ""
            raise HTTPException(503, f"Modell {self.modell} ist nicht bereit ({self.zustand.value}){grund}")
        self._mlx_begrenzen()
        async with self._sperre:
            if abbruch.is_set():
                raise Abgebrochen("vor dem Start abgebrochen")
            try:
                return await asyncio.to_thread(funktion, *args, abbruch)
            finally:
                self._mlx_leeren()


async def verbindung_beobachten(request: Request, abbruch: threading.Event) -> None:
    """Setzt `abbruch`, sobald der Aufrufer die Verbindung schließt (bis die Aufgabe beendet wird)."""
    while not abbruch.is_set():
        if await request.is_disconnected():
            abbruch.set()
            return
        await asyncio.sleep(ABBRUCH_TAKT_S)


def anwendung(dienst: str, rechner: Rechner, lader: Callable[[], Any], gewichte_vollstaendig: Callable[[], bool],
              titel: str) -> FastAPI:
    """FastAPI-Anwendung eines Dienstes: lädt das Modell beim Start im Hintergrund (nicht im Probemodus),
    Gesundheit, Fehler als JSON."""

    @asynccontextmanager
    async def lebenslauf(_app: FastAPI):
        aufgabe = None if rechner.zustand == Zustand.PROBE else asyncio.create_task(rechner.laden(lader))
        try:
            yield
        finally:
            if aufgabe:
                aufgabe.cancel()

    app = FastAPI(title=f"Bildwerk {titel}", version=__version__, lifespan=lebenslauf)
    app.state.rechner = rechner

    @app.exception_handler(Abgebrochen)
    async def abgebrochen(request: Request, f: Abgebrochen):
        log.info("%s: Auftrag abgebrochen (%s)", request.url.path, f)
        return JSONResponse(status_code=499, content={"detail": f"abgebrochen: {f}"})

    @app.exception_handler(Exception)
    async def fehler(request: Request, f: Exception):
        log.error("Fehler in %s: %s\n%s", request.url.path, f, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": type(f).__name__, "message": str(f)[:500]})

    def gesundheit() -> JSONResponse:
        g = Gesundheit(dienst=dienst, fassung=__version__, modell=rechner.modell, zustand=rechner.zustand, meldung=rechner.meldung,
                       gewichte_vollstaendig=gewichte_vollstaendig())
        # 503, solange das Modell nicht rechnen kann: der Maschinenraum wartet beim Start darauf
        bereit = rechner.zustand in (Zustand.BEREIT, Zustand.PROBE)
        return JSONResponse(g.model_dump(mode="json"), status_code=200 if bereit else 503)

    app.add_api_route("/gesundheit", gesundheit, methods=["GET"])
    app.add_api_route("/health", gesundheit, methods=["GET"], include_in_schema=False)
    return app


def starten(app: FastAPI, port: int, einstellungen: Einstellungen) -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=einstellungen.adresse, port=port, log_level="warning", access_log=False)
