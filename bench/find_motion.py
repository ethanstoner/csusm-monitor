"""Flag frames that differ from a static background, independently of any detector.

Hand-labelling 400 near-identical night frames by eye is unreliable in a way
that biases the result: attention drifts, and the frames you look at hardest are
the ones a detector already flagged. This builds a temporal-median background
and reports frames containing a blob that departs from it, using nothing but
pixels — so the candidate list owes nothing to YOLO or to the grounding model
and can be used to score both.

It is a *candidate finder*, not ground truth. Everything it flags still gets
looked at; its job is to make sure nothing is skipped.

Usage:
    python bench/find_motion.py bench/frames_night --min-area 400
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--min-area", type=int, default=400,
                    help="minimum blob area in px at half resolution")
    ap.add_argument("--threshold", type=int, default=28,
                    help="per-pixel difference from background to count as changed")
    ap.add_argument("--sample", type=int, default=80,
                    help="frames sampled to build the background median")
    args = ap.parse_args()

    paths = sorted(Path(args.frames).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")

    # Background = per-pixel median over a subsample. Median rather than mean so
    # a person standing still in a handful of frames cannot pull the background
    # towards themselves and erase their own detection.
    step = max(1, len(paths) // args.sample)
    stack = []
    for p in paths[::step]:
        img = cv2.imread(str(p))
        stack.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    background = np.median(np.stack(stack), axis=0).astype(np.uint8)
    bg_grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    print(f"background from {len(stack)} frames")

    flagged = []
    for p in paths:
        img = cv2.imread(str(p))
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(grey, bg_grey)
        _, mask = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
        # Close gaps so one person does not become five specks.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) >= args.min_area]
        if blobs:
            areas = sorted((int(cv2.contourArea(c)) for c in blobs), reverse=True)
            flagged.append((p.name, len(blobs), areas[:4]))

    print(f"\n{len(flagged)} of {len(paths)} frames contain a blob >= {args.min_area}px:")
    for name, n, areas in flagged:
        print(f"  {name}  blobs={n}  areas={areas}")


if __name__ == "__main__":
    main()
