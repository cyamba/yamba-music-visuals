#!/usr/bin/env bash
# Downloads the two (SIL Open Font License) fonts used on the title cards.
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)/musicviz/fonts"
mkdir -p "$dir"
base=https://raw.githubusercontent.com/google/fonts/main/ofl
curl -sSfL -o "$dir/Cinzel.ttf"    "$base/cinzel/Cinzel%5Bwght%5D.ttf"
curl -sSfL -o "$dir/Cormorant.ttf" "$base/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf"
echo "fonts saved to $dir"
