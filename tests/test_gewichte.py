"""Prüfliste und Prüfung der Gewichte (mit kleinen Scheindateien), dazu: passen Prüfliste, Steckbrief und der gepinnte
Stand von mflux zusammen. Kein Modell wird geladen; mflux wird nur importiert."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import tomllib
from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects

from bildwerk.gewichte import Datei, Pruefliste, pruefen
from bildwerk.modelle import MODELLE, modul, moss_image

WURZEL = Path(__file__).resolve().parent.parent


def schein_liste(ablage: Path) -> Pruefliste:
    gross, klein = b"\x01" * 1000, b'{"a": 1}'
    liste = Pruefliste(repo="Schein/Modell", revision="abc123", lizenz="Apache-2.0", dateien=[
        Datei(pfad="transformer/teil.safetensors", groesse=len(gross), sha256=hashlib.sha256(gross).hexdigest()),
        Datei(pfad="transformer/config.json", groesse=len(klein), git_sha1=hashlib.sha1(b"blob %d\0" % len(klein) + klein).hexdigest()),
    ])
    ordner = liste.ordner(ablage)
    (ordner / "transformer").mkdir(parents=True)
    (ordner / "transformer/teil.safetensors").write_bytes(gross)
    (ordner / "transformer/config.json").write_bytes(klein)
    return liste


def test_pruefen(tmp_path: Path):
    liste = schein_liste(tmp_path)
    assert liste.ordner(tmp_path) == tmp_path / "models--Schein--Modell" / "snapshots" / "abc123"
    stand = pruefen(liste, tmp_path, pruefsummen=True)
    assert stand.vollstaendig and "vollständig" in stand.meldung
    # gleich groß, anderer Inhalt: nur die Prüfung mit Prüfsummen sieht es
    (liste.ordner(tmp_path) / "transformer/teil.safetensors").write_bytes(b"\x02" * 1000)
    assert pruefen(liste, tmp_path).vollstaendig
    stand = pruefen(liste, tmp_path, pruefsummen=True)
    assert stand.falsch == ["transformer/teil.safetensors"] and not stand.vollstaendig
    # abgebrochener Download: kürzer als verlangt
    (liste.ordner(tmp_path) / "transformer/config.json").write_bytes(b"{")
    stand = pruefen(liste, tmp_path)
    assert stand.fehlen == ["transformer/config.json"] and "unvollständig" in stand.meldung
    assert not pruefen(liste, tmp_path / "leer").vollstaendig


@pytest.mark.parametrize("kennung", sorted(MODELLE))
def test_pruefliste_passt_zu_mflux(kennung: str):
    """Repo, Revision und Dateien der Prüfliste sind genau das, was der gepinnte Stand von mflux lädt; nach einem neuen
    Pin schlägt das fehl, bis `bildwerk pruefliste <modell>` die Liste erneuert hat."""
    m = modul(kennung)
    liste = Pruefliste.lesen(m.PRUEFLISTE)
    repo, revision, muster = m.mflux_herkunft()
    assert (liste.repo, liste.revision, liste.lizenz) == (repo, revision, m.ANGABEN.lizenz)
    pfade = [d.pfad for d in liste.dateien]
    assert sorted(filter_repo_objects(pfade, allow_patterns=muster)) == sorted(pfade)
    assert all(d.sha256 or d.git_sha1 for d in liste.dateien)
    assert not [p for p in pfade if p.endswith(".py")]                  # nie Code aus dem Modell-Repo
    for teil in ("transformer", "text_encoder", "vae"):                  # jeder Teil mit Gewichten und Konfiguration
        assert any(p.startswith(f"{teil}/") and p.endswith(".safetensors") for p in pfade)
        assert f"{teil}/config.json" in pfade


def test_mflux_auf_dem_gepinnten_commit():
    pin = tomllib.loads((WURZEL / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["mflux"][0]
    commit = re.search(r"@([0-9a-f]{40})$", pin).group(1)
    herkunft = json.loads(importlib.metadata.distribution("mflux").read_text("direct_url.json"))
    assert herkunft["vcs_info"]["commit_id"] == commit, "installiertes mflux weicht vom Pin ab: ./installieren.sh"


def test_moss_image_angaben_wie_mflux():
    """Grenzen und Vorgaben im Steckbrief sind die von mflux (ohne Modell: nur Konfiguration und Zeitplan)."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.moss_image import MOSSImage
    from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator
    from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler

    a = moss_image.ANGABEN
    config = ModelConfig.moss_image_sft()
    assert config.max_sequence_length == a.beschreibung_max_token and config.supports_guidance
    assert (MOSSImage.DEFAULT_STEPS, MOSSImage.DEFAULT_GUIDANCE) == (a.schritte.vorgabe, a.guidance)
    assert MOSSImageLatentCreator.SPATIAL_SCALE == a.groessen.raster
    MOSSImageScheduler.compute_sigmas(a.schritte.groesste)
    with pytest.raises(ValueError):
        MOSSImageScheduler.compute_sigmas(a.schritte.groesste + 1)
    for groesse in a.groessen.empfohlen + [a.groessen.vorgabe]:
        breite, hoehe = a.groessen.kanten(groesse)
        MOSSImageLatentCreator.check_size(height=hoehe, width=breite)
