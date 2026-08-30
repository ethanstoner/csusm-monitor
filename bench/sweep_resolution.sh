#!/usr/bin/env bash
# Sweep inference resolution for LocateAnything-3B.
#
# MoonViT tokenises at the image's native resolution, so a 1920x1080 campus
# frame produces far more vision tokens than the model card's batch-4 A100
# figures assume. This finds the point where latency and VRAM become survivable
# without losing the objects the queries are about. Run inside WSL2.
set -e
FRAMES="${1:-bench/frames}"
OUT="${2:-bench/results}"
QUERY="${3:-trash can}"
N="${4:-4}"
mkdir -p "$OUT"
# Greedy decoding throughout. The runtime samples at temperature 0.7 by
# default, which makes the same frame return different counts, and a sweep run
# that way measures the sampler as much as the resolution.
for SIDE in 1920 1600 1440 1280 960 640 448; do
  echo "=================== max_side=$SIDE ==================="
  python bench/la3b_bench.py \
    --frames "$FRAMES" --limit "$N" --query "$QUERY" --temperature 0 \
    --max-side "$SIDE" --out "$OUT/sweep_${SIDE}.json" \
    --annotate "$OUT/sweep_${SIDE}_annotated" 2>&1 | tail -14
done
