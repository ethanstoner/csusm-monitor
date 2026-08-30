"""Run the production YOLOv8n person detector over a fixed frame set.

Emits one JSON record per frame so the same frames can be scored by a second
backend and the two compared directly. This deliberately calls the same
`detect_people` the monitor runs in production, thresholds included — a
comparison against a differently-tuned YOLO would measure the tuning, not the
model.

Usage:
    python bench/yolo_bench.py bench/frames bench/results/yolo.json
"""
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.detector import detect_people  # noqa: E402

# Ignore the first N inferences when computing latency stats: the first call
# builds CUDA/CPU kernels and is not representative of steady state.
WARMUP_FRAMES = 3


def main(frames_dir: str, out_path: str) -> None:
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {frames_dir}")

    records = []
    for i, path in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"skip unreadable {path.name}")
            continue
        t0 = time.perf_counter()
        count, boxes = detect_people(frame)
        latency_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "frame": path.name,
            "count": count,
            "boxes": boxes,
            "latency_ms": round(latency_ms, 1),
            "warmup": i < WARMUP_FRAMES,
        })
        print(f"{path.name}: {count} people, {latency_ms:.0f}ms")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "backend": "yolov8n",
        "frames_dir": frames_dir,
        "records": records,
    }, indent=2))
    print(f"\nwrote {len(records)} records to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/frames",
         sys.argv[2] if len(sys.argv) > 2 else "bench/results/yolo.json")
