"""Score one or more backend result files against hand-labelled ground truth.

Reports exact-count accuracy plus the two errors that actually matter for an
occupancy monitor, kept separate because they are not interchangeable: missing
people who are there (the dashboard says the cart is free when it is not) and
inventing people who are not (the dashboard sends you elsewhere for nothing).

Usage:
    python bench/compare.py bench/people_labels.json \
        bench/results/yolo_people.json bench/results/la3b_people.json
"""
import json
import sys
from pathlib import Path


def load_counts(path: Path) -> tuple[str, dict[str, int]]:
    data = json.loads(path.read_text())
    counts: dict[str, int] = {}
    for r in data["records"]:
        # Repeat runs collapse to the first scoring of each frame.
        counts.setdefault(r["frame"], r["count"])
    return data.get("backend", path.stem), counts


def score(truth: dict[str, int], counts: dict[str, int]) -> dict:
    frames = sorted(counts)
    exact = missed = invented = 0
    fp_frames, fn_frames = [], []
    for f in frames:
        want, got = truth.get(f, 0), counts[f]
        if want == got:
            exact += 1
        if got < want:
            missed += want - got
            fn_frames.append((f, want, got))
        elif got > want:
            invented += got - want
            fp_frames.append((f, want, got))
    return {
        "frames": len(frames),
        "exact": exact,
        "exact_pct": round(100 * exact / len(frames), 1) if frames else 0.0,
        "people_truth": sum(truth.get(f, 0) for f in frames),
        "people_found": sum(counts.values()),
        "missed": missed,
        "invented": invented,
        "fp_frames": fp_frames,
        "fn_frames": fn_frames,
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    labels = json.loads(Path(sys.argv[1]).read_text())["people"]

    print(f"{'backend':<24}{'frames':>7}{'exact':>8}{'exact%':>8}"
          f"{'people':>8}{'found':>7}{'missed':>8}{'invented':>10}")
    print("-" * 80)
    details = []
    for arg in sys.argv[2:]:
        name, counts = load_counts(Path(arg))
        s = score(labels, counts)
        print(f"{name:<24}{s['frames']:>7}{s['exact']:>8}{s['exact_pct']:>8}"
              f"{s['people_truth']:>8}{s['people_found']:>7}{s['missed']:>8}{s['invented']:>10}")
        details.append((name, s))

    for name, s in details:
        if not (s["fp_frames"] or s["fn_frames"]):
            continue
        print(f"\n{name} — disagreements with ground truth:")
        for f, want, got in s["fn_frames"]:
            print(f"  MISSED   {f}: truth {want}, reported {got}")
        for f, want, got in s["fp_frames"]:
            print(f"  INVENTED {f}: truth {want}, reported {got}")


if __name__ == "__main__":
    main()
