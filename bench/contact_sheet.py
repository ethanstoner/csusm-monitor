"""Tile a frame directory into contact sheets for human labelling.

Ground truth for the people benchmark cannot be "the frames YOLO flagged" —
that bakes YOLO's misses into the labels and makes its recall look perfect by
construction. Every frame gets looked at instead, and these sheets are how.

Usage:
    python bench/contact_sheet.py bench/frames_people bench/results/sheets --cols 6 --rows 5
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cell-width", type=int, default=440)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    cw = args.cell_width
    ch = round(cw * 9 / 16)
    per_sheet = args.cols * args.rows
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sheet_no, start in enumerate(range(0, len(frames), per_sheet), start=1):
        chunk = frames[start:start + per_sheet]
        sheet = Image.new("RGB", (args.cols * cw, args.rows * ch), (12, 16, 24))
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(chunk):
            img = Image.open(path).convert("RGB").resize((cw, ch), Image.LANCZOS)
            x, y = (i % args.cols) * cw, (i // args.cols) * ch
            sheet.paste(img, (x, y))
            # Stem only — the label has to be readable at thumbnail size.
            tag = path.stem.split("_")[-1]
            draw.rectangle([x, y, x + 34, y + 14], fill=(0, 0, 0))
            draw.text((x + 3, y + 2), tag, fill=(0, 255, 120))
        dest = out_dir / f"sheet_{sheet_no:02d}.jpg"
        sheet.save(dest, quality=88)
        print(f"{dest}  ({len(chunk)} frames)")


if __name__ == "__main__":
    main()
