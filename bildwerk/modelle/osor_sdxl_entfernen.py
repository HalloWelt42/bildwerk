"""Dienst bild-entfernen-osor: OSOR auf SDXL-Inpainting, Entfernen in einem Schritt samt Schatten und Spiegelung
(ATL-43 des Ateliers, Vergleich der Mittelstufe mit bild-entfernen). Name vorläufig, bis die Messung entscheidet.

OSOR (One-Step Object Removal, ECCV 2026) ist ein LoRA (Rang 256) auf der UNet von SDXL-Inpainting 0.1 mit einem fünften
Ausgangskanal: neben der Vorhersage des Rauschens eine Alpha-Fläche, die sagt, wo entfernt werden soll; sie reicht über
die Maske hinaus, wo Schatten und Spiegelung liegen, und verzeiht ungenaue Masken. Gerechnet wird ein einziger Schritt
bei t = 400 mit festem Text ("Remove the instance of object"), wie im Code der Autoren (MIT); das Ergebnis wird über die
Alpha-Fläche (und mindestens die Maske) ins Bild gesetzt.

Rechenwerk: PyTorch auf der Grafik des Macs (MPS) mit diffusers und peft; der Code der Autoren ist nur auf CUDA geprüft.
Der Text wird beim Laden einmal mit den beiden Textkodierern von SDXL kodiert, danach sind sie wieder frei.

Lizenz: Gewichte CreativeML Open RAIL++-M (SDXL-Inpainting und OSOR-SDXL): kommerzielle Nutzung erlaubt, aber mit den
Nutzungsauflagen der Lizenz.
"""

from __future__ import annotations

import io
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Aufgabe, Bildauftrag, Groessenregel, Modellangaben, Rechenwerk, Schrittregel, aufgaben_lesen
from ..entfernen import ausschnitt, maske_aus_signalfarbe, zurueckmischen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen

PORT = 8799
DIENST = "bild-entfernen-osor"
#: das Grundmodell SDXL-Inpainting 0.1 (fp16-Fassung) und, als Zusatz, die Gewichte von OSOR-SDXL (Phase II)
PRUEFLISTE = Path(__file__).with_name("osor_sdxl_entfernen.json")
ZUSATZ_PRUEFLISTEN = [Path(__file__).with_name("osor_sdxl_entfernen_osor.json")]
AUFGABEN = Path(__file__).with_name("osor_sdxl_entfernen.aufgaben.toml")
OSOR = "osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth"
GRUNDMUSTER = ["model_index.json", "scheduler/scheduler_config.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
               "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/config.json",
               "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
               "tokenizer/*", "tokenizer_2/*"]

#: wie die Autoren gelernt haben: fester Text, ein Schritt bei t = 400, LoRA Rang 256 auf diesen Modulen
TEXT = "Remove the instance of object"
SCHRITT_T = 400
LORA_RANG = 256
LORA_MODULE = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
#: Arbeitsgröße des Ausschnitts (Kanten im Raster von 16; SDXL rechnet nativ um 1024)
ARBEIT = 1024

RECHENWERK = "torch"

ANGABEN = Modellangaben(
    kennung="osor-sdxl-entfernen",
    name="OSOR-SDXL",
    aliase=[DIENST, "QinmingZhou/OSOR", "osor-sdxl"],
    zweck="Markierte Flächen in einem Schritt entfernen, samt Schatten und Spiegelung; verzeiht ungenaue Masken",
    kann_bearbeiten=True,
    vorlagen_min=1,
    vorlagen_max=1,
    eignung=[],
    sprachen=[],
    parameter_mrd=2.6,
    textkodierer="CLIP ViT-L und bigG (Text fest)",
    lizenz="CreativeML Open RAIL++-M",
    lizenz_hinweis="Kommerziell nutzbar mit den Nutzungsauflagen der Lizenz CreativeML Open RAIL++-M.",
    erschienen=date(2026, 6, 30),
    groessen=Groessenregel(raster=8, kante_min=64, kante_max=4096, pixel_min=64 * 64, pixel_max=4096 * 4096, vorgabe="1024x1024"),
    schritte=Schrittregel(vorgabe=1, kleinste=1, groesste=1),
    guidance=1.0,
    beschreibung_max_token=0,
    aufgaben=aufgaben_lesen(AUFGABEN),
)


def herkunft() -> tuple[str, str, list[str]]:
    liste = Pruefliste.lesen(PRUEFLISTE)
    return liste.repo, liste.revision, GRUNDMUSTER


def geraet() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Motor:
    """Hält VAE, UNet mit eingerechnetem OSOR-LoRA und den kodierten Text; geladen nur aus der Modellablage."""

    angaben = ANGABEN
    quantisierung = "bf16"

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.pruefliste = Pruefliste.lesen(PRUEFLISTE)
        self.zusatz = Pruefliste.lesen(ZUSATZ_PRUEFLISTEN[0])
        self.signalfarbe: tuple[int, int, int] = tuple(ANGABEN.aufgaben[Aufgabe.ENTFERNEN].signalfarbe or (255, 0, 255))  # type: ignore[assignment]
        self.unet: Any = None
        self.vae: Any = None
        self.ablauf: Any = None
        self.text: Any = None

    def gewichte(self) -> Gewichtsstand:
        stand = pruefen(self.pruefliste, ablage())
        stand.fehlen.extend(f"OSOR: {f}" for f in pruefen(self.zusatz, ablage()).fehlen)
        return stand

    def rechenwerk(self) -> Rechenwerk:
        from importlib import metadata
        return Rechenwerk(name="torch", fassung=f"{metadata.version('torch')} (diffusers {metadata.version('diffusers')})")

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Laden von Hand: bildwerk gewichte {ANGABEN.kennung} --laden")
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from peft.tuners.tuners_utils import BaseTunerLayer

        grund, wo = str(self.pruefliste.ordner(ablage())), geraet()
        self.text = self._text_kodieren(grund, wo)
        self.vae = AutoencoderKL.from_pretrained(grund, subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(wo).eval()
        self.ablauf = DDPMScheduler.from_pretrained(grund, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(grund, subfolder="unet", variant="fp16", torch_dtype=torch.float32)
        unet.add_adapter(LoraConfig(r=LORA_RANG, lora_alpha=LORA_RANG, init_lora_weights="gaussian", target_modules=LORA_MODULE))
        alt = unet.conv_out
        unet.conv_out = torch.nn.Conv2d(alt.in_channels, 5, alt.kernel_size, alt.stride, alt.padding)   # fünfter Kanal: Alpha
        # nur Tensoren, kein ausführbarer Inhalt aus der Datei
        roh = torch.load(self.zusatz.ordner(ablage()) / OSOR, map_location="cpu", weights_only=True)
        zustand = {k.replace("module.", "").replace("_orig_mod.", "").removeprefix("unet."): v for k, v in roh.items()}
        ergebnis = unet.load_state_dict(zustand, strict=False)
        if ergebnis.unexpected_keys or not any("lora_" in k for k in zustand) or "conv_out.weight" not in zustand:
            raise GewichteFehlen(f"OSOR passt nicht zur UNet: {len(ergebnis.unexpected_keys)} unbekannte Schlüssel, etwa {ergebnis.unexpected_keys[:3]}")
        for modul in unet.modules():                         # LoRA einrechnen: kein Umweg je Schritt
            if isinstance(modul, BaseTunerLayer):
                modul.merge()
        self.unet = unet.to(wo, dtype=torch.bfloat16).eval()

    def _text_kodieren(self, grund: str, wo: str) -> tuple[Any, Any]:
        """Den festen Text einmal mit beiden Textkodierern von SDXL kodieren (wie cache_prompts.py der Autoren)."""
        import torch
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        with torch.no_grad():
            t1 = CLIPTokenizer.from_pretrained(grund, subfolder="tokenizer")
            k1 = CLIPTextModel.from_pretrained(grund, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32).to(wo)
            e1 = k1(t1(TEXT, padding="max_length", max_length=t1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(wo),
                    output_hidden_states=True).hidden_states[-2]
            t2 = CLIPTokenizer.from_pretrained(grund, subfolder="tokenizer_2")
            k2 = CLIPTextModelWithProjection.from_pretrained(grund, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float32).to(wo)
            a2 = k2(t2(TEXT, padding="max_length", max_length=t2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(wo),
                    output_hidden_states=True)
            text = (torch.cat([e1, a2.hidden_states[-2]], dim=-1).to(torch.bfloat16), a2.text_embeds.to(torch.bfloat16))
        del k1, k2
        return text

    def entfernen(self, teil: Image.Image, teilmaske: Image.Image, seed: int) -> tuple[Image.Image, Image.Image]:
        """Ein Ausschnitt (Arbeitsgröße) mit Maske durch OSOR; liefert das Bild und die Alpha-Fläche (L) in Arbeitsgröße."""
        import numpy as np
        import torch
        import torch.nn.functional as F

        wo = geraet()
        with torch.no_grad():
            bild = torch.from_numpy(np.asarray(teil, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(wo) * 2 - 1
            maske = torch.from_numpy(np.asarray(teilmaske, dtype=np.float32) / 255.0)[None, None].to(wo)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            z_lq = self.vae.encode(bild).latent_dist.sample(generator=gen) * self.vae.config.scaling_factor
            m_lat = F.max_pool2d(maske, kernel_size=8, stride=8)
            rauschen = torch.randn(z_lq.shape, generator=gen).to(wo)
            t = torch.full((1,), SCHRITT_T, dtype=torch.long, device=wo)
            z = self.ablauf.add_noise(z_lq, rauschen, t)
            z_in = torch.cat([z, m_lat, z_lq], dim=1).to(torch.bfloat16)
            h, b = teil.height, teil.width
            zeiten = torch.tensor([[h, b, 0, 0, h, b]], dtype=torch.bfloat16, device=wo)
            text, gebuendelt = self.text
            aus = self.unet(z_in, t, encoder_hidden_states=text, added_cond_kwargs={"text_embeds": gebuendelt, "time_ids": zeiten},
                            return_dict=False)[0].float()
            eps, alpha = aus[:, :4], torch.sigmoid(aus[:, 4:])
            z_aus = self.ablauf.step(eps.cpu(), int(SCHRITT_T), z.float().cpu()).pred_original_sample.to(wo)
            z_aus = z_aus * alpha + z_lq * (1 - alpha)
            x = ((self.vae.decode(z_aus / self.vae.config.scaling_factor).sample + 1) / 2).clamp(0, 1)
            alpha_gross = F.interpolate(alpha, size=(h, b), mode="bilinear")
        ergebnis = Image.fromarray((x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"), "RGB")
        return ergebnis, Image.fromarray((alpha_gross[0, 0].cpu().numpy() * 255).round().astype("uint8"), "L")

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        from PIL import ImageChops

        if len(auftrag.vorlagen) != 1:
            raise ValueError("Entfernen rechnet mit genau einer Vorlage")
        bild = Image.open(io.BytesIO(auftrag.vorlagen[0])).convert("RGB")
        if bild.size != (auftrag.breite, auftrag.hoehe):
            bild = bild.resize((auftrag.breite, auftrag.hoehe), Image.LANCZOS)
        maske = maske_aus_signalfarbe(bild, self.signalfarbe)
        kasten = ausschnitt(maske)
        if kasten is None:
            raise ValueError(f"Keine markierte Fläche: die zu entfernende Fläche muss deckend in der Signalfarbe {self.signalfarbe} stehen")
        breite, hoehe = kasten[2] - kasten[0], kasten[3] - kasten[1]
        teil = bild.crop(kasten).resize((ARBEIT, ARBEIT), Image.LANCZOS)
        teilmaske = maske.crop(kasten).resize((ARBEIT, ARBEIT), Image.NEAREST)
        teil.paste((128, 128, 128), (0, 0, ARBEIT, ARBEIT), teilmaske)     # die Signalfarbe nicht ins Modell geben
        bilder = []
        for seed in auftrag.seeds:
            if abbruch.is_set():
                raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
            gefuellt, alpha = self.entfernen(teil, teilmaske, seed)
            # eingesetzt wird, wo OSOR entfernen will (auch Schatten), mindestens aber die markierte Fläche
            flaeche = Image.new("L", bild.size)
            flaeche.paste(ImageChops.lighter(alpha, teilmaske).resize((breite, hoehe), Image.BILINEAR), kasten[:2])
            bilder.append(zurueckmischen(bild, gefuellt, flaeche, kasten))
        return bilder
