#!/bin/bash
# Build the animated Rain file with this Mac's Blender and open it with autoplay.
# Usage: bash scripts/run_rain.sh "/path/to/Rain v3.3/rain_v3.2.blend"
set -e
cd "$(dirname "$0")/.."
SRC="${1:?path to rain_v3.2.blend}"
OUT="$(dirname "$SRC")/rain_mocap_local.blend"
B=/Applications/Blender.app/Contents/MacOS/Blender
"$B" -b -y --python pipeline/build_blend.py -- "$SRC" profiles/rain.json data/targets_rain.json "$OUT"
"$B" -y "$OUT" --python scripts/autoplay.py
