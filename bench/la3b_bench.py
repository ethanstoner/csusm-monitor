"""Run NVIDIA LocateAnything-3B over a fixed frame set and record what it costs.

This exists to answer the first question in ROADMAP.md — single-frame latency on
this machine, at the real stream resolution — before any architecture is built
around the model. It is deliberately separate from the monitor: it imports the
model's own `batch_utils` runtime straight out of the HF snapshot and touches
nothing in `backend/`.

Runs under WSL2 (the model is Linux-only), not the Windows venv.

Usage:
    python bench/la3b_bench.py --frames bench/frames --out bench/results/la3b.json \
        --query person --repeats 1 --annotate bench/results/la3b_annotated
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# The monitor's own decoder, so the benchmark measures the code that would
# actually ship rather than a second implementation that agrees with it today.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.vlm import parse_boxes  # noqa: E402

HF_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/*/"
)

# The first inference builds CUDA kernels and pays one-off allocator cost; it is
# reported but excluded from the steady-state statistics.
WARMUP_FRAMES = 2


def resolve_snapshot() -> Path:
    matches = sorted(glob.glob(HF_SNAPSHOT_GLOB))
    if not matches:
        raise SystemExit(
            "LocateAnything-3B snapshot not found. Run:\n"
            "  hf download nvidia/LocateAnything-3B"
        )
    return Path(matches[-1])


def annotate(image, boxes, label, out_path, scale=(1.0, 1.0)):
    """Draw decoded boxes so the coordinate order can be checked by eye.

    Boxes arrive in original-frame pixels; `scale` maps them onto whatever
    (possibly downscaled) image is being drawn.
    """
    from PIL import ImageDraw
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    sx, sy = scale
    for b in boxes:
        draw.rectangle([b["x1"] * sx, b["y1"] * sy, b["x2"] * sx, b["y2"] * sy],
                       outline=(0, 255, 0), width=3)
        draw.text((b["x1"] * sx + 4, max(0, b["y1"] * sy - 12)), label, fill=(0, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def count_spread(records: list[dict]) -> dict:
    """How much the count moves when the same frame is scored repeatedly.

    With sampling on, this is not zero, and a counting product cannot quietly
    ignore that. Reported as the fraction of frames that ever disagreed with
    themselves, plus the largest single disagreement seen.
    """
    by_frame: dict[str, list[int]] = {}
    for r in records:
        by_frame.setdefault(r["frame"], []).append(r["count"])
    repeated = {f: c for f, c in by_frame.items() if len(c) > 1}
    if not repeated:
        return {}
    unstable = [f for f, c in repeated.items() if len(set(c)) > 1]
    return {
        "frames_repeated": len(repeated),
        "frames_disagreeing_with_themselves": len(unstable),
        "max_spread": max(max(c) - min(c) for c in repeated.values()),
    }


def summarise(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    ordered = sorted(latencies)
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 1),
        "median_ms": round(ordered[len(ordered) // 2], 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
        "max_ms": round(ordered[-1], 1),
        "mean_ms": round(sum(ordered) / len(ordered), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="bench/frames")
    ap.add_argument("--out", default="bench/results/la3b.json")
    ap.add_argument("--query", default="person",
                    help="category query; join several with the model's </c> separator")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "magi", "la_flash"])
    ap.add_argument("--scheduler", default="eager")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    # The shipped runtime defaults to temperature 0.7 / top-p 0.9, i.e. it
    # samples. That is fine for a demo and wrong for a counter: the same frame
    # can return a different number twice in a row. --repeats measures exactly
    # that, and --temperature 0 turns it off.
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--repeats", type=int, default=1,
                    help="score every frame this many times, to measure run-to-run spread")
    ap.add_argument("--coord-order", default="xyxy", choices=["xyxy", "xxyy"])
    ap.add_argument("--limit", type=int, default=0, help="only score the first N frames")
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale the frame so its longest side is at most this many px "
                         "(0 = native). MoonViT tokenises at native resolution, so this is "
                         "the dominant cost knob.")
    ap.add_argument("--annotate", default="", help="directory to write annotated frames to")
    args = ap.parse_args()

    snapshot = resolve_snapshot()
    os.environ.setdefault("LA_FLASH_MODEL", str(snapshot))
    os.environ["LA_FLASH_ATTN"] = args.attn
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = args.scheduler
    sys.path.insert(0, str(snapshot))

    import torch
    from PIL import Image
    from batch_utils import generate_batch_hybrid, load

    print(f"snapshot: {snapshot}")
    print(f"device:   {torch.cuda.get_device_name(0)}")

    t0 = time.perf_counter()
    load()
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s")

    frames = sorted(Path(args.frames).glob("*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    # Sampling knobs are passed explicitly, never left to the runtime's
    # defaults, so a result file always records the settings that produced it.
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "scheduler": args.scheduler,
        "temperature": args.temperature,
        "top_p": args.top_p if args.top_p > 0 else None,
        "repetition_penalty": args.repetition_penalty,
    }

    records, steady = [], []
    inference = 0
    for path in frames:
        image = Image.open(path).convert("RGB")
        native = (image.width, image.height)
        if args.max_side and max(native) > args.max_side:
            scale = args.max_side / max(native)
            image = image.resize((round(image.width * scale), round(image.height * scale)),
                                 Image.LANCZOS)
        for rep in range(args.repeats):
            torch.cuda.synchronize()
            t = time.perf_counter()
            texts = generate_batch_hybrid([(image, args.query)], **gen_kwargs)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t) * 1000
            raw = texts[0]
            # Boxes are decoded against the ORIGINAL frame size: the model's
            # 0-1000 grid is resolution-independent, so downscaling for speed
            # still yields full-resolution boxes.
            boxes = parse_boxes(raw, native[0], native[1], order=args.coord_order)
            is_warmup = inference < WARMUP_FRAMES
            if not is_warmup:
                steady.append(latency_ms)
            records.append({
                "frame": path.name,
                "repeat": rep,
                "inference_size": [image.width, image.height],
                "count": len(boxes),
                "boxes": boxes,
                "latency_ms": round(latency_ms, 1),
                "warmup": is_warmup,
                "raw_response": raw,
            })
            if args.annotate and rep == 0:
                annotate(image, boxes, args.query, Path(args.annotate) / path.name,
                         scale=(image.width / native[0], image.height / native[1]))
            print(f"{path.name}[{rep}]: {len(boxes)} {args.query}, {latency_ms:.0f}ms")
            inference += 1

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    result = {
        "backend": "LocateAnything-3B",
        "snapshot": str(snapshot),
        "device": torch.cuda.get_device_name(0),
        "query": args.query,
        "max_side": args.max_side,
        "attn": args.attn,
        "scheduler": args.scheduler,
        "coord_order": args.coord_order,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "repeats": args.repeats,
        "count_spread": count_spread(records),
        "model_load_s": round(load_s, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "latency": summarise(steady),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsteady-state latency: {result['latency']}")
    print(f"peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
