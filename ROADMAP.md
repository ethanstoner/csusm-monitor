# Roadmap — CSUSM Campus Monitor

Items are written down only once they are concrete enough to argue with. Each
one states what it buys, what it costs, and what would have to be true for it
to be worth doing.

---

## Switch detection to NVIDIA LocateAnything-3B

**Status:** investigating. Hardware verified, architecture not designed.

### What it is

[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
is an open-vocabulary visual grounding model — Qwen2.5-3B as the language
model, MoonViT as the vision tower, joined by a 2-layer MLP projector. It emits
coordinate tokens that decode to boxes, using Parallel Box Decoding (block-wise
multi-token prediction) rather than one-box-at-a-time autoregression.

### Why it is worth considering

The monitor currently counts exactly one thing, because YOLOv8n was trained to.
Every question beyond "how many people" needs a different model or a fine-tune.
An open-vocabulary model is prompted in natural language, so the same weights
answer:

- how many **bicycles** are at the rack
- how many tables are **occupied** vs empty
- how long the **queue** at the coffee cart is, as distinct from people merely
  standing nearby
- whether the shuttle is **at** the stop

That is a genuinely different product — "campus occupancy" becomes "ask the
campus a question" — and it needs no retraining or labelled data, which is the
part that would otherwise make it a semester of work rather than a weekend.

There is a second, smaller prize. `StaticObjectFilter` exists only because
YOLO misreads signs and poles as people, and it costs a ~60s warmup after every
restart during which counts are deliberately discarded. A grounding model
reasons semantically and may simply not make that error. **May.** It will make
different ones. That is a thing to measure, not to assume — and the filter does
not come out until measurement says it can.

### Feasibility, as verified

| Requirement | This machine | Verdict |
|---|---|---|
| NVIDIA Ampere / Ada / Hopper / Blackwell | RTX 4090, compute capability 8.9 (Ada) | ✅ supported |
| ~11.7 GB peak VRAM (batch 4, A100 reference) | 24 GB | ✅ comfortable headroom |
| Linux | Windows 11 + WSL2 | ⚠️ needs WSL2 CUDA passthrough, or the Linux box |
| License | NVIDIA License — academic and non-profit research **only** | ⚠️ see below |

**The licence is the constraint that does not go away.** The NVIDIA License
permits use, reproduction and modification "for academic and non-profit
research purposes only"; commercial deployment is reserved to NVIDIA and its
affiliates. As a CSUSM student project this is squarely inside the grant. It
means two things anyway, and both should be written down before any code is
written rather than discovered later:

1. This can never become a paid product without swapping the model out. If that
   is ever a goal, the abstraction below matters more than the model does.
2. The README currently presents this repo as a portfolio piece. Shipping a
   non-commercial-licensed model as the default backend of a public repo
   deserves an explicit note in the README, so nobody clones it and deploys it
   commercially on the strength of the MIT `LICENSE` file at the root — which
   covers *this* code, not the weights it would pull.

### The actual engineering problem

Not the model. The loop.

`detect_people()` (`backend/detector.py:122`) is the seam — it takes a frame and
returns `(count, boxes)`, and every camera thread funnels through it behind a
single `_model_lock` (`detector.py:20`). YOLOv8n is small enough that
serialising inference across cameras costs nothing noticeable on a 5-second
cycle.

A 3B VLM generating coordinate tokens is not in that latency class, and the gap
is not marginal. With N cameras sharing one lock on a `DETECTION_INTERVAL` of
5s, there is a point where inference time per frame exceeds the cycle budget and
the lock backs up permanently — every camera then reports numbers that are
quietly minutes stale, which is worse than reporting nothing, because the
dashboard has no way to show it.

**No latency number is written here on purpose.** The model card reports up to
2.5× throughput over autoregressive decoding and a batch-4 A100 memory figure,
but nothing that translates to single-frame latency on a 4090 under this
workload. That measurement is step one, and it decides the whole design.

### Proposed shape — a third backend, not a replacement

The README already documents a pluggable detection architecture (challenge #5):
`DetectionWorker` (local YOLO) and `FrigateListener` (MQTT) both write the same
schema, and the API merges live counts from whichever is active. LocateAnything
should slot in there rather than replace `detect_people()` in place.

Concretely, the likely split:

- **YOLOv8n stays the hot path** for person counts on the 5s cycle. It is fast,
  it is good enough at the one thing it does, and the trend data has months of
  history computed with it — swapping the model underneath a heatmap silently
  changes what the historical average *means*.
- **LocateAnything runs cold**, on a much longer interval or on demand, for
  open-vocabulary queries. Served as a separate process (vLLM or SGLang, both
  listed as compatible) so it can batch across cameras and be restarted without
  taking the monitor down with it.
- Open-vocabulary results need a schema decision that person-counting did not:
  `detections` stores a bare `count` per camera, which presumes everyone knows
  what is being counted. A prompt-per-camera, or several labels per camera,
  needs its own table rather than overloading that column.

### Open questions, in the order they should be answered

1. **Single-frame latency on the 4090**, hybrid generation mode, at the actual
   stream resolution. Everything else depends on this number.
2. Does it hold up on these specific feeds? Public university HLS streams are
   low-bitrate, compressed, and often at an unhelpful angle — the model card's
   benchmarks are not that. Test on real captured frames before designing
   anything.
3. Does it beat YOLO on the false-positive cases `StaticObjectFilter` was built
   for? Run both over the same saved frames and compare.
4. Where does it run — WSL2 with CUDA passthrough on this machine, or a
   separate Linux host? The Docker path (`docker-compose.yml`) argues for a
   host, but the 4090 is in *this* box, and no other machine here is confirmed
   to have a supported GPU. Check before assuming a host is available;
   otherwise WSL2 is the only option and the answer is already decided.
5. Does it degrade gracefully? A VLM can return malformed coordinates or refuse.
   `detect_people()` has one narrow contract and every caller assumes it holds.

### Definition of done

Not "it runs." A prompt-driven count visible on the dashboard for at least one
camera, measured against YOLO on the same frames, with the latency number
written into this file — and the licence note in the README.

---

## Backlog

Not yet argued through; listed so they are not forgotten.

- Retention beyond 30 days for aggregates, so the heatmap can span a semester
  without keeping every row.
- Alerting on sustained camera outages rather than only marking them in trends.
