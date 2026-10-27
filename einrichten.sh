#!/usr/bin/env bash
# Einrichten des Bildwerks. Schritte und Proben der Neueinrichtung stehen in docs/handbuch.md.
#
#   ./einrichten.sh                                       wie ./installieren.sh: Umgebung mit den erprobten Paketständen und
#                                                         mflux auf dem gepinnten Commit, je Modell eine Dienstdatei im
#                                                         Maschinenraum (nur wenn sie fehlt)
#   ./einrichten.sh --probe --ziel DIR --port-versatz N   Probelauf des Handbuchs: dieselbe Installation in DIR, die
#                                                         Dienstdateien in einen eigenen Ordner DIR/maschinenraum (nie in
#                                                         den echten), dann jeder Bilddienst im Probemodus (lädt und rechnet
#                                                         nichts) auf seinem Port plus N, bis er antwortet
#   ./einrichten.sh --probe --ziel DIR --abbauen          beendet die Bilddienste des Probelaufs
set -euo pipefail
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARTEN_S=120

probe_aufbauen() {
  local ziel="$1" versatz="$2" dienste="$1/maschinenraum/dienste" erwartet anzahl datei modell port adresse modelle erste=""
  mkdir -p "$dienste"
  MASCHINENRAUM_ORDNER="$ziel/maschinenraum" "$ziel/installieren.sh"
  erwartet="$(find "$ziel/deploy" -name 'bildwerk-*.toml' | wc -l | tr -d ' ')"
  anzahl="$(find "$dienste" -name 'bildwerk-*.toml' | wc -l | tr -d ' ')"
  if [ "$anzahl" != "$erwartet" ] || [ "$anzahl" = 0 ]; then
    echo "Die Installation schrieb $anzahl statt $erwartet Dienstdateien" >&2
    return 1
  fi
  for datei in "$dienste"/bildwerk-*.toml; do
    modell="$(basename "$datei" .toml)"; modell="${modell#bildwerk-}"
    # Port aus der Dienstdatei, verschoben um den Versatz
    port=$(( $(sed -n 's/^port = //p' "$datei") + versatz ))
    HF_HUB_OFFLINE=1 nohup "$ziel/.venv/bin/python" -m bildwerk "$modell" --probe --port "$port" > "$ziel/$modell.log" 2>&1 &
    echo $! > "$ziel/$modell.pid"
    adresse="http://127.0.0.1:$port/gesundheit"
    warten "$adresse" "$ziel/$modell.log" || return 1
    # Ausgaben erst ganz lesen, dann suchen (grep -q in einer Röhre beendet curl unter pipefail vorzeitig)
    modelle="$(curl -sf "http://127.0.0.1:$port/v1/models" || true)"
    [[ "$modelle" == *'"steckbrief"'* ]] || { echo "$modell: /v1/models ohne Steckbrief" >&2; return 1; }
    [ -n "$erste" ] || erste="$adresse"
  done
  echo "ANTWORTET $erste"
}

warten() {
  local ende=$(( $(date +%s) + WARTEN_S )) gesundheit
  while [ "$(date +%s)" -lt "$ende" ]; do
    gesundheit="$(curl -sf "$1" || true)"
    [[ "$gesundheit" == *'"zustand":"probe"'* ]] && return 0
    sleep 1
  done
  echo "Kein Bilddienst unter $1 nach $WARTEN_S s" >&2
  tail -n 40 "$2" >&2 || true
  return 1
}

probe_abbauen() {
  local pid datei
  for datei in "$1"/*.pid; do
    [ -f "$datei" ] || continue
    pid="$(cat "$datei")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$datei"
  done
}

probe=0; abbauen=0; ziel=""; versatz=""
while [ $# -gt 0 ]; do
  case "$1" in
    --probe) probe=1 ;;
    --abbauen) abbauen=1 ;;
    --ziel) ziel="${2:?--ziel braucht einen Ordner}"; shift ;;
    --port-versatz) versatz="${2:?--port-versatz braucht eine Zahl}"; shift ;;
    *) echo "unbekannte Angabe: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$probe" = 0 ]; then
  exec "$HIER/installieren.sh"
elif [ -z "$ziel" ] || [ ! -d "$ziel" ]; then
  echo "--probe braucht --ziel mit einem vorhandenen Ordner" >&2; exit 2
elif [ "$abbauen" = 1 ]; then
  probe_abbauen "$(cd "$ziel" && pwd)"
else
  probe_aufbauen "$(cd "$ziel" && pwd)" "${versatz:?--port-versatz fehlt}"
fi
