# Roadmap — CSUSM Campus Monitor

Items are written down only once they are concrete enough to argue with. Each
one states what it buys, what it costs, and what would have to be true for it
to be worth doing.

---

## Open-vocabulary detection via LocateAnything-3B

**Status:** shipped as the **primary** person detector, with YOLOv8n as an
automatic fallback. Every question this section originally gated the design on
has been measured; what remains open is listed at the bottom and is about
conditions the benchmark sets do not cover, not about the design.

### What it is

[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
is an open-vocabulary visual grounding model — Qwen2.5-3B as the language
model, MoonViT as the vision tower, joined by a 2-layer MLP projector. It emits
coordinate tokens that decode to boxes, using Parallel Box Decoding (block-wise
multi-token prediction) rather than one-box-at-a-time autoregression.

### Why it was worth doing

The monitor counted exactly one thing, because YOLOv8n was trained to. Every
question beyond "how many people" needed a different model or a fine-tune. An
open-vocabulary model is prompted in natural language, so the same weights
answer how many bicycles are at the rack, how many tables are occupied, or how
long the queue is as distinct from people merely standing nearby — with no
retraining and no labelled data, which is the part that would otherwise make it
a semester of work rather than a weekend.

That claim is now demonstrated rather than asserted: prompting `trash can`
against a real frame finds all six receptacles, correctly boxed, and `trash can`
is not a class YOLOv8n has. See [bench/README.md](bench/README.md).

### What the measurements said

The roadmap previously refused to write down a latency number until one was
measured. Measured, on the 4090, at the real stream resolution:

| Question | Answer |
|---|---|
| Single-frame latency, native 1920px | **38.2 s** median — because it needs 25.75 GB and the card has 24 |
| Single-frame latency, 1440px | **1.31 s** median, 13.7 GB peak |
| YOLOv8n on the same frames | **17.6 ms** median |
| Does it hold up on these feeds? | Yes at native resolution (6/6 objects); recall drops to ~4/6 below it |
| Does it degrade gracefully? | Not on its own — the shipped runtime samples at temperature 0.7 and returns different counts for identical pixels. Fixed by decoding greedily. |

Three findings changed the design, and all three are written up with the
evidence in [bench/README.md](bench/README.md): the released box order is
documented two contradictory ways, the default sampling settings make counts
irreproducible, and full recall requires more VRAM than the GPU has.

### The shape it took

An earlier draft of this section argued for keeping YOLO as the hot path and
running the grounding model cold. That was revised once the numbers were in.
Two of the three reasons for it did not survive:

- *"Trend history has months of data computed with YOLO."* It does not any more
  — the 30-day retention sweep removed it. There was nothing left to protect.
- *"1.2 s does not fit a 5-second cycle."* It fits comfortably, for the number
  of cameras this actually has.
- *"It needs a Linux host with a GPU."* This one held, and is why the fallback
  exists rather than why the switch does not.

What shipped:

- **LocateAnything-3B is the primary person detector**, chosen per cycle by
  `PersonCounters`, running out-of-process
  (`services/locate_anything/server.py`) over HTTP so it can be slow, restarted
  or absent without taking the monitor with it.
- **YOLOv8n is the automatic fallback**, not a deleted predecessor. 17.6 ms on
  any machine beats 1,215 ms on one machine when the GPU service is not there,
  and a portfolio repo that shows an empty dashboard to anyone without a 4090
  is not a working portfolio repo.
- **Every row records which model produced it** (`detections.source`), because
  a detector that changes underneath a heatmap has to leave a trail.
  `/api/history/sources` reports the mix; the dashboard names the live one.
- **`StaticObjectFilter` stays on the YOLO path only.** It suppresses whatever
  holds still, which is right for YOLO's failure mode and wrong for a model
  that may not share it — on a grounding model it would equally erase a person
  sitting at a table.
- **Open-vocabulary queries get their own table and their own slow loop.**
  `detections.count` carries no label and every trend query reads it as people;
  a bicycle count sharing that column would change what the heatmap means
  without changing the query that draws it.

### The licence, which does not go away

The NVIDIA License permits use "for academic and non-profit research purposes
only". As a CSUSM student project this is squarely inside the grant. It still
means two things, both now handled rather than discovered later:

1. This can never become a paid product without swapping the model out — hence
   the process boundary and the `VLM_ENABLED` flag, so the monitor runs
   perfectly well without it.
2. The README carries an explicit note that the MIT `LICENSE` at the root covers
   this code and not the weights, so nobody clones the repo and deploys it
   commercially on the strength of it.

### Answered since

**The runaway-generation fix is verified**, not just argued. Passing
`top_p=None` alongside `temperature=0` made generation run to the token budget
(340 boxes for a 6-object query, 32.7 s). With `top_p=0.9`, same frame and same
client: `trash can` 4, `red tent` 1, `lamp post` 1, all in ~1–2 s. Greedy
decoding does not ignore nucleus sampling in this runtime. The client also
discards any answer over `MAX_PLAUSIBLE_BOXES`, which is unit-tested.

**Person counting has been measured against hand-labelled ground truth**, on
three sets totalling 820 frames and 33 people, every frame reviewed by eye
rather than labelling whatever a detector flagged.

| Set | People | YOLOv8n found | LocateAnything found |
|---|---|---|---|
| 180 frames, daylight, unoccluded | 7 | 7 | 7 |
| 240 frames, dusk, edges and shadow | 21 | **10** | 19 |
| 400 frames, sunset to 00:55 | 5 | 3 | **5** |
| **Total** | **33** | **20** | **31** |

The easy set could not separate them. The other two do: **YOLOv8n misses 13 of
33 people**, including two standing at the cart in plain view and two walking
across a lit plaza at night, while LocateAnything misses 2 — both more than half
outside the frame. Neither invents anybody.

**A prompt beat a filter.** Prompted `"person"`, the grounding model found all 21
and invented 6 — four of them the plaza's trash receptacles, the same failure
`StaticObjectFilter` was written for. Prompted `"pedestrian"`, false positives
went to zero at a cost of 2 in recall, and frame-exact accuracy went 97.5% →
99.2%. That is a precision/recall trade made by editing a string, which is the
capability argument for an open-vocabulary detector made concrete.

**Ground truth had its own bugs.** Three of the nine originally-scored false
positives were real people missed during labelling; the model was right and the
grader was wrong. `bench/people2_labels.json` records the correction rather than
hiding it, and cropping every disputed detection is now part of the method.

**The `StaticObjectFilter` question is answered, and the answer is uncomfortable.**
A 400-frame capture running to 00:55 on a lit plaza — no frame dark enough for
the brightness gate to skip — brings the total to **880 benchmark frames across
full daylight, dusk and six hours of night in which raw YOLOv8n produced zero
stationary false positives.** Every detection it made in every set is a real
person. The filter was built to remove signs and poles it never saw.

The methodological point matters more than the result: **the filter's successes
are unmeasurable from production data**, because snapshots and database rows are
both written after filtering. Only `bench/yolo_bench.py`, which calls
`detect_people` raw, could answer this at all. It is also now more expensive than
it was — YOLO is the fallback, so its ~60-second warm-up discard costs data
precisely during a GPU outage, when the system is already degraded.

It has not been deleted. One camera, one scene, one day is thin evidence for
removing a safety mechanism, and the scene changes with the season. The numbers
are in `bench/README.md` so the call gets made against data rather than memory.

### Still open

1. **Crowds and distance, the two untested cases most likely to break.** The
   largest count in any set is 2, and everyone is reasonably near the camera. The
   resolution sweep already showed recall on small distant objects collapsing
   below native resolution, and the only two people `"pedestrian"` still misses
   are both half outside the frame — plausibly the same weakness. A busy weekday
   lunch rush is the measurement that matters and does not exist yet.
2. **Whether native resolution is reachable another way.** Tiling the frame and
   scoring each tile separately would get full recall inside the VRAM budget at
   the cost of overlap handling and N× latency. Worth trying, not yet tried.
3. **Batching across cameras.** The runtime supports it and there is currently
   one live camera, so it buys nothing today.

---

## Backlog

Not yet argued through; listed so they are not forgotten.

- Retention beyond 30 days for aggregates, so the heatmap can span a semester
  without keeping every row.
- The Starbucks camera has been serving a zero-byte manifest since May. It is
  now correctly reported as offline rather than silently stale, but nothing
  alerts on the transition — a camera can die and the only evidence is a badge
  nobody is looking at.
- Decide `StaticObjectFilter`'s fate against the evidence above: keep it as
  cheap insurance against a scene that changes, or drop it and its warm-up
  discard. Either is defensible; leaving it undecided is the only bad option.
