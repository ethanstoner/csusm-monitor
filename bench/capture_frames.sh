#!/usr/bin/env bash
# Capture raw (unannotated) frames from a live CSUSM HLS stream at a fixed
# interval, for use as a frozen benchmark set. Annotated snapshots from
# data/snapshots are unusable here: the drawn boxes cue a grounding model.
URL="${1:?stream url}"
OUT="${2:?output dir}"
COUNT="${3:-40}"
SLEEP="${4:-15}"
mkdir -p "$OUT"
for i in $(seq -w 1 "$COUNT"); do
  ffmpeg -y -i "$URL" -frames:v 1 -q:v 2 "$OUT/frame_$i.jpg" -loglevel error 2>/dev/null
  sleep "$SLEEP"
done
