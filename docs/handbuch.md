---
zweck: "Die Bilddienste des Hauses, je Bildmodell ein eigener Prozess mit derselben Schnittstelle: Bilder aus Text und nach Bildvorlagen für die Apps, vermittelt von der Leitstelle."
geraet: [mac]
abhaengigkeiten: [maschinenraum]
daten:
  - "Mac: die Gewichte in der Modellablage `~/Maschinenraum/modelle/huggingface` (MOSS-Image-SFT, 27,2 GB; FLUX.2 klein 9B, 34,7 GB) und die eigene 8-Bit-Ablage von Qwen-Image-Edit-2511 in `~/Maschinenraum/modelle/bild/mflux/qwen-image-edit-2511-8bit`; LaMa für schnelles Entfernen übersetzt in `~/Maschinenraum/modelle/bild/coreml/lama-big-800.mlmodelc` (207 MB); FLUX.2 klein 4B (16 GB) und das LoRA erase-v1 (0,2 GB) für bild-entfernen in der Modellablage"
  - "Mac: Dienstdateien `~/Maschinenraum/dienste/bildwerk-*.toml` , `bild-entfernen-schnell.toml`, `bild-entfernen.toml`, `bild-entfernen-osor.toml`, `bild-fuellen.toml` und `bild-erzeugen.toml` (je Modell eine)"
sicherung:
  - "nichts Eigenes zu sichern: die Dienstdateien schreibt die Installation neu, die Gewichte lassen sich neu laden; die Modellablage gehört zur Sicherung des Macs (siehe Grundlage Mac)"
  - "das Repo liegt bisher nur auf dem Mac (keine entfernte Ablage): der Ordner gehört zur Sicherung des Macs"
schluessel_orte: []
einrichten: einrichten.sh
stand: 397c39f
---

# Bildwerk

## Zweck

Das Bildwerk rechnet Bilder für die Apps: je Bildmodell ein Dienst, ein eigener Prozess mit genau einem Modell, der es
beim Start lädt und den Speicher beim Anhalten wieder frei gibt. Ein Bilddienst ist eine Rechenressource wie jedes
andere Modell: er startet nie von selbst. Der [[Maschinenraum]] startet und hält ihn an, wenn die [[Leitstelle]] es
verlangt; sie entscheidet über Laden, Entladen und Vorrang, und Apps (auch der Assistent) erreichen ihn nur über sie.

| Dienst | Fähigkeit | Modell |
|---|---|---|
| moss-image | bild_erzeugen | MOSS-Image-SFT (Apache-2.0), 8 Bit, gerechnet mit mflux (MLX); Bilder aus Text, kein Bearbeiten |
| flux2-klein | bild_erzeugen | FLUX.2 klein 9B (FLUX Non-Commercial License: Bilder nur nicht-kommerziell nutzen), 8 Bit, mflux (MLX); schnell in 4 Schritten, Bilder aus Text und nach bis zu 4 Vorlagen (Stil, Bezug) |
| qwen-image-edit | bild_erzeugen | Qwen-Image-Edit-2511 (Apache-2.0, frei nutzbar), eigene 8-Bit-Ablage, mflux (MLX); bearbeitet nach Anweisung und zeichnet nach 1 bis 3 Vorlagen neu, rechnet nur mit Vorlage |
| qwen-image-edit-schnell | bild_erzeugen | dasselbe Modell mit dem Lightning-LoRA von lightx2v (Apache-2.0, 0,85 GB): 8 Schritte, Stärke 1, rund ein Zehntel der Rechenzeit; Grundgewichte aus der 8-Bit-Ablage von qwen-image-edit |
| bild-entfernen-schnell | bild_erzeugen | LaMa big (Apache-2.0, frei nutzbar) als Core-ML-Paket über coremltools: entfernt markierte Flächen in 0,3 bis 0,4 s, erfindet nichts Neues; genau eine Vorlage mit der Fläche in der Signalfarbe |
| bild-entfernen | bild_erzeugen | FLUX.2 klein 4B mit dem LoRA erase-v1 (beides Apache-2.0, frei nutzbar) über mflux: entfernt markierte Flächen samt Schatten und Spiegelung in einigen Sekunden; Mittelstufe zwischen LaMa und Qwen; genau eine Vorlage mit der Fläche in der Signalfarbe |
| bild-entfernen-osor | bild_erzeugen | OSOR auf SDXL-Inpainting (Lizenz CreativeML Open RAIL++-M: kommerziell mit Nutzungsauflagen) in PyTorch: entfernt in einem Schritt samt Schatten und Spiegelung; Vergleich der Mittelstufe, Name vorläufig; genau eine Vorlage mit der Fläche in der Signalfarbe |
| bild-fuellen | bild_erzeugen | Qwen-Image 2.1 Turbo mit ControlNet-Union (Qwen Research License: nur nicht-kommerziell) über mflux: füllt die markierte Fläche nach Anweisung mit echter Maske, außerhalb bleibt alles; Entfernen und Füllen |
| bild-erzeugen | bild_erzeugen | FLUX 3 Image Klein (FLUX Kommunity License: nicht-kommerziell frei, kommerziell unter 5 Mio. US-Dollar Jahresumsatz) über mflux: Bilder aus Text in 16 Schritten; Bearbeiten zurückgestellt |

Sichtbar heißt der Dienst nach seinem Modell ("MOSS-Image", `name` der Dienstdatei); Maschinenraum, Leitstelle und
Apps setzen die Aufgabe klein dahinter ("MOSS-Image · Bilderzeugung"). Die Kennung `bildwerk-moss-image` bleibt.

Alle Dienste sprechen dieselbe Schnittstelle im üblichen Bildformat (`POST /v1/images/generations`,
`POST /v1/images/edits`, `GET /v1/models` mit Steckbrief, `GET /gesundheit`); Einzelheiten im README des Repos. Port
und Prüfadresse stehen unten im Block "Auf einen Blick" (aus den Dienstdateien).

## Was sie braucht

- Mac nach [[Grundlage Mac]]: Python ab 3.11, die Modellablage.
- [[Maschinenraum]]: die Installation trägt die Dienste in `~/Maschinenraum/dienste/` ein; gestartet werden sie dort
  auf Geheiß der [[Leitstelle]], nie von der Installation selbst.
- Die Pakete kommen mit festen Ständen aus `einschraenkungen.txt`, mflux auf dem Commit aus `pyproject.toml`.
- Die Gewichte je Modell in der Modellablage, vollständig nach der Prüfliste des Modells; für MOSS-Image rund 27 GB, für
  FLUX.2 klein 9B rund 35 GB.

## Neu einrichten

Zwei Wege. Aus der Sicherung: den Ordner {{ordner}} und die Modellablage zurückspielen (siehe [[Grundlage Mac]]); dann
fehlen nur Umgebung und Dienstdatei. Frisch: Ordner zurückspielen, Installation, die Gewichte einmal laden.

### 1. Ordner und Installation

```
git clone {{repo}} {{ordner}}
cd {{ordner}} && ./einrichten.sh
```

`./einrichten.sh` ruft `./installieren.sh`: `.venv` im Ordner mit den Ständen aus `einschraenkungen.txt` und mflux auf
dem gepinnten Commit und coremltools, dann je Modell eine Dienstdatei aus `deploy/` nach `~/Maschinenraum/dienste/`
(`bildwerk-<modell>.toml`, neue Dienste nach ihrer Fähigkeit wie `bild-entfernen-schnell.toml`), aber nur,
wenn es sie noch nicht gibt (Änderungen von Hand bleiben). Gestartet und geladen wird nichts.

Probe: `ls ~/Maschinenraum/dienste/bildwerk-*.toml ~/Maschinenraum/dienste/bild-entfernen*.toml` nennt
`bildwerk-moss-image.toml`, `bildwerk-flux2-klein.toml`, `bildwerk-qwen-image-edit.toml`,
`bildwerk-qwen-image-edit-schnell.toml`, `bild-entfernen-schnell.toml`, `bild-entfernen.toml`, `bild-entfernen-osor.toml`, `bild-fuellen.toml` und `bild-erzeugen.toml`, und der Maschinenraum zeigt die Dienste unter
Dienste als aus.

### 2. Gewichte

Aus der Sicherung liegen sie in der Modellablage. Frisch einmal laden (rund 27 GB je Modell aus dem Netz, nur Gewichte,
Konfiguration und Tokenizer, nie Code des Modell-Repos), je Modell:
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --laden` (ebenso `flux2-klein`;
dessen Repo verlangt einmal die Annahme der Lizenz bei Hugging Face). Die Dienstdatei sperrt
danach das Netz (`HF_HUB_OFFLINE=1`); der Dienst lädt nie selbst.

Qwen-Image-Edit-2511 läuft aus einer eigenen, schon quantisierten Ablage: einmal das Original laden (57,7 GB), dann
umwandeln, alles mit derselben Umgebung:
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte qwen-image-edit --laden`, danach
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk umwandeln qwen-image-edit`. Das Umwandeln lädt das
Modell einmal in bfloat16 (rund 58 GB Speicher); vorher große Sprachmodelle über die Leitstelle entladen und danach
genau so wieder laden. Erst wenn eine Messung über die Leitstelle (`tools/messen.py qwen-image-edit`) belegt, dass die
eigene Ablage rechnet, legt derselbe Befehl mit `--original-loeschen` das Original in den Papierkorb (nicht endgültig).
Die schnelle Fassung braucht diese Ablage und dazu nur das LoRA (0,85 GB):
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte qwen-image-edit-schnell --laden`.
Aus der Sicherung liegt die 8-Bit-Ablage schon in der Modellablage.

LaMa (schnelles Entfernen) ebenso über eine eigene Ablage: das Core-ML-Paket laden (216 MB), dann für diesen Mac
übersetzen: `HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte lama --laden`, danach
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk umwandeln lama`; nach der Messung über die Leitstelle
(`tools/messen.py lama --vorlage <Bild mit magenta markierter Fläche>`) legt `--original-loeschen` das Paket in den
Papierkorb. Aus der Sicherung liegt die übersetzte Ablage schon in der Modellablage.

bild-entfernen (FLUX.2 klein 4B mit erase-v1) lädt Grundgewichte (16 GB) und LoRA samt fertig kodiertem Text (0,2 GB)
mit einem Befehl: `HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte flux2-klein-entfernen --laden`.
Daraus baut `HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk umwandeln flux2-klein-entfernen` die eigene
8-Bit-Ablage mit eingerechnetem LoRA und kodiertem Text (`~/Maschinenraum/modelle/bild/mflux/flux2-klein-4b-erase-8bit`, 8 GB);
der Dienst lädt nur daraus. bild-fuellen ebenso: `bildwerk gewichte qwen21-fuellen --laden` (Qwen-Image 2.1 Turbo 32,5 GB und
ControlNet 7,6 GB), dann `bildwerk umwandeln qwen21-fuellen` (`bild/mflux/qwen-image-2.1-turbo-fuellen-8bit`). Die Originale
bleiben liegen, bis Quellen und Duplikate gemeinsam abgeräumt werden (MR-4). MOSS-Image ebenso: `bildwerk umwandeln moss-image`
(`bild/mflux/moss-image-sft-8bit`, 14 GB statt 27 GB). FLUX 3 Image Klein (bild-erzeugen): erst auf Hugging Face der Lizenz zustimmen (Konto
des Macs), dann `bildwerk gewichte flux3-klein --laden` (20,3 GB) und `bildwerk umwandeln flux3-klein`. Die eigenen Ablagen gibt es je Bitbreite (`--bits 4` oder `8`,
Ordner `...-4bit` bzw. `...-8bit`); welche ein Dienst lädt, sagt `BILDWERK_QUANTISIERUNG` in seiner Dienstdatei (MR-5).
bild-entfernen-osor ebenso SDXL-Inpainting 0.1 (fp16, 6,9 GB) und OSOR-SDXL Phase II (3,1 GB):
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte osor-sdxl-entfernen --laden`.

Probe: `HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --pruefsummen` (ebenso
`flux2-klein` und `qwen-image-edit`) meldet "vollständig" und "Prüfsummen stimmen".

### 3. Starten über die Leitstelle

Die [[Leitstelle]] startet den Dienst über den Maschinenraum, sobald eine App ein Bild mit diesem Modell verlangt. Ohne
Modell lässt sich die Schnittstelle im Probemodus prüfen: `.venv/bin/bildwerk moss-image --probe --port 18791`, danach
mit Strg-C beenden.

Probe: im Probemodus meldet `curl -s http://127.0.0.1:18791/gesundheit` den Zustand `probe` und
`"gewichte_vollstaendig":true`; nach einem Start durch die Leitstelle meldet `curl -s {{pruefadresse:bildwerk-moss-image}}`
den Zustand `bereit`.

## Sicherung und Rückspielen

Das Bildwerk hält keine eigenen Daten: die Dienstdateien schreibt die Installation neu, erzeugte Bilder gehen direkt in
der Antwort an den Aufrufer. Was eine Neueinrichtung dauern lässt, sind die Gewichte in der Modellablage (rund 27 GB je
Modell); sie gehören zur Sicherung des Macs (siehe [[Grundlage Mac]]). Das Repo liegt bisher nur auf dem Mac und ist
damit nur über die Sicherung des Macs wiederherstellbar.

## Wenn etwas klemmt

- Der Dienst meldet `gestoert` mit "Gewichte unvollständig": die Gewichte fehlen in der Modellablage oder ein Download
  brach ab; Schritt 2.
- Der Dienst bleibt bei `laedt`: das Laden liest rund 27 GB und quantisiert auf 8 Bit; das dauert. Endet er nach der
  Startzeit der Dienstdatei, steht der Grund im Protokoll des Dienstes im Maschinenraum.
- Antwort `400`: Größe, Schritte oder Vorlagen passen nicht zum Modell (Grenzen im Steckbrief unter `/v1/models`); `501`
  bei Bearbeiten: das Modell kann es nicht (MOSS-Image; Vorlagen nimmt FLUX.2 klein).
- Speicher knapp: jeder Dienst braucht in 8 Bit rund 24 GB (Anmeldung in der Dienstdatei); weniger mit
  `BILDWERK_QUANTISIERUNG = "4bit"` unter `[umgebung]` der Dienstdatei. Die Leitstelle hält Dienste an, die sie nicht
  braucht.
