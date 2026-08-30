"""HTTP wrapper around NVIDIA LocateAnything-3B.

Runs as its own process, on its own machine if you like, and speaks one route:
POST /locate {"image": <base64 jpeg>, "query": "person"} -> {"raw": "...", "latency_ms": n}.

Three reasons it is not imported into the monitor:

1. The model is Linux-only. The monitor runs on Windows here, so the two cannot
   share a process even in principle — this side lives in WSL2.
2. A 3B VLM does not fit the monitor's 5-second detection budget. Behind HTTP it
   can be slow, be restarted, or be missing entirely without the person-count
   loop noticing.
3. The weights are NVIDIA-licensed for academic and non-profit research only.
   Keeping them behind a boundary means the monitor itself stays MIT-licensed
   code that merely *can* call a grounding service, rather than a repo that
   ships one.

Deliberately returns the model's raw text rather than parsed boxes: coordinate
decoding and the dropping of malformed output live in `backend/vlm.py`, where
they are covered by tests that run on any machine without a GPU.

Run (inside WSL2, in the venv that has torch + transformers):
    python services/locate_anything/server.py --host 127.0.0.1 --port 8100
"""
import argparse
import base64
import glob
import io
import logging
import os
import sys
import threading
import time

logger = logging.getLogger("locate-anything")

HF_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/*/"
)

_generate = None
_load_lock = threading.Lock()
# One model, one GPU: concurrent generate() calls would interleave on the same
# weights and the same KV cache. Requests queue instead.
_infer_lock = threading.Lock()


def resolve_snapshot(explicit: str = "") -> str:
    if explicit:
        return explicit
    matches = sorted(glob.glob(HF_SNAPSHOT_GLOB))
    if not matches:
        raise SystemExit(
            "LocateAnything-3B not found locally. Run:\n  hf download nvidia/LocateAnything-3B"
        )
    return matches[-1]


def load_model(snapshot: str, attn: str, scheduler: str):
    """Import the model's own batched runtime and load the weights once."""
    global _generate
    with _load_lock:
        if _generate is not None:
            return
        os.environ["LA_FLASH_MODEL"] = snapshot
        os.environ["LA_FLASH_ATTN"] = attn
        os.environ["LA_FLASH_HYBRID_SCHEDULER"] = scheduler
        sys.path.insert(0, snapshot)

        from batch_utils import generate_batch_hybrid, load

        started = time.perf_counter()
        load()
        logger.info("model loaded in %.1fs", time.perf_counter() - started)
        _generate = generate_batch_hybrid


def build_app(snapshot: str, attn: str, scheduler: str, max_new_tokens: int,
              temperature: float, top_p: float, repetition_penalty: float):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from PIL import Image

    app = FastAPI(title="LocateAnything-3B grounding service")

    @app.get("/health")
    async def health():
        return {"ready": _generate is not None, "snapshot": snapshot, "attn": attn}

    @app.post("/locate")
    async def locate(payload: dict):
        query = (payload or {}).get("query")
        encoded = (payload or {}).get("image")
        if not query or not encoded:
            return JSONResponse(status_code=400, content={"detail": "image and query are required"})
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail": f"undecodable image: {e}"})

        started = time.perf_counter()
        try:
            with _infer_lock:
                texts = _generate([(image, query)], max_new_tokens=max_new_tokens,
                                  scheduler=scheduler, temperature=temperature,
                                  top_p=top_p if top_p > 0 else None,
                                  repetition_penalty=repetition_penalty)
        except Exception:
            logger.exception("generation failed for query %r", query)
            return JSONResponse(status_code=500, content={"detail": "generation failed"})
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info("%r -> %d chars in %.0fms", query, len(texts[0]), latency_ms)
        return {"raw": texts[0], "latency_ms": round(latency_ms, 1),
                "width": image.width, "height": image.height}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="", help="local snapshot path; resolved from the HF cache if omitted")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "magi", "la_flash"])
    ap.add_argument("--scheduler", default="eager")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    # Greedy by default, against the runtime's own 0.7/0.9 defaults. Sampling
    # makes the same frame return a different count on consecutive polls —
    # measured at 4 of 5 frames disagreeing with themselves over 5 repeats — and
    # a time series nobody can reproduce is not worth storing. Greedy cost the
    # same to the millisecond and found more objects, not fewer.
    ap.add_argument("--temperature", type=float, default=0.0)
    # 0.9, matching the configuration every benchmark in bench/README.md was
    # run under. Passing top_p=None alongside temperature=0 made generation run
    # away instead of stopping: 340 boxes returned for a query with 6 correct
    # answers, and a box covering the entire frame. Do not "simplify" this to
    # None on the theory that greedy decoding ignores it.
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    snapshot = resolve_snapshot(args.model)
    logger.info("loading %s (attn=%s)", snapshot, args.attn)
    # Load before binding the port: a service that accepts requests while the
    # weights are still loading just converts a clear startup wait into a pile
    # of client timeouts.
    load_model(snapshot, args.attn, args.scheduler)

    import uvicorn
    app = build_app(snapshot, args.attn, args.scheduler, args.max_new_tokens,
                    args.temperature, args.top_p, args.repetition_penalty)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
