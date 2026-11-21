"""Download a public labelled drone dataset and convert it to the challenge format.

Source: "Seraphim" drone detection dataset on Hugging Face (public, CC BY 4.0, credit it
when you present). 83,483 images at 640x640, one class (drone), YOLO labels, collected by
its authors from 23 open datasets. No account needed.

  1) look before you download (labels only, ~11 MB):
       python get_public_drones.py --analyze
  2) download one part and convert it (test = 8,349 images, ~0.9 GB):
       python get_public_drones.py --parts test --out-dir <folder> --max-side 64 --max-images 3000
     more parts: --parts test,train1,train2,train3,train4  (train parts are 1.5-2.6 GB each)

Every kept box becomes class 0 (drone_air). Needs only pillow. See README.md for the full recipe.
Tested on locally built zips in the layout the dataset page describes, not yet on the real download.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from challenge_data import clip_box, print_summary, summarize, write_dataset  # noqa: E402

BASE = "https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset/resolve/main/"
PARTS = {  # name: (images zip, label zips; the first must exist, the others are used if present)
    "test": ("test/images/batch_001.zip", ["test/labels/batch_001.zip"]),
    "train1": ("train/images/batch_001.zip", ["train/labels/batch_001.zip"]),
    "train2": ("train/images/batch_002.zip", ["train/labels/batch_001.zip", "train/labels/batch_002.zip"]),
    "train3": ("train/images/batch_003.zip", ["train/labels/batch_001.zip", "train/labels/batch_003.zip"]),
    "train4": ("train/images/batch_004.zip", ["train/labels/batch_001.zip", "train/labels/batch_004.zip"]),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def download(rel: str, cache: Path, optional: bool = False) -> Path | None:
    """Fetch one file of the dataset into the cache (resumes a partial download). None = optional file absent."""
    dst = cache / rel.replace("/", "__")
    if dst.exists():
        return dst
    cache.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "python-urllib"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + rel, headers=headers), timeout=60) as r:
                if have and r.status != 206:
                    have = 0  # server restarted from zero
                total = have + int(r.headers.get("Content-Length") or 0)
                done, t0, last = have, time.time(), 0.0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 5:
                            last = time.time()
                            speed = (done - have) / max(1e-6, last - t0) / 1e6
                            print(f"  {rel}: {done / 1e6:7.0f} / {total / 1e6:.0f} MB  ({speed:.1f} MB/s)", flush=True)
            if total and done < total:
                raise ConnectionError(f"stopped at {done} of {total} bytes")
            part.rename(dst)
            return dst
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # already complete
                part.rename(dst)
                return dst
            if exc.code == 404:  # will not fix itself
                if optional:
                    return None
                raise SystemExit(f"not found on the server: {BASE + rel}") from exc
            print(f"  download problem ({exc}); attempt {attempt}/5")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  download problem ({exc}); attempt {attempt}/5")
        time.sleep(3 * attempt)
    raise SystemExit(f"could not download {BASE + rel}")


def parse_yolo(text: str) -> list[tuple[float, float, float, float]] | None:
    """Return normalised (x, y, w, h) boxes. Handles box lines and polygon lines. None = unreadable."""
    boxes = []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        try:
            v = [float(t) for t in p[1:]]
        except ValueError:
            return None
        if len(v) == 4:
            cx, cy, w, h = v
            boxes.append((cx - w / 2, cy - h / 2, w, h))
        elif len(v) >= 6 and len(v) % 2 == 0:  # polygon: take its bounding box
            xs, ys = v[0::2], v[1::2]
            boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
        else:
            return None
    return boxes


def read_labels(zip_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith(".txt"):
                boxes = parse_yolo(z.read(name).decode("utf-8", "replace"))
                if boxes is not None:
                    out[Path(name).stem] = boxes
    return out


_label_cache: dict[str, dict[str, list[tuple[float, float, float, float]]]] = {}


def labels_for(part: str, cache: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    merged: dict[str, list[tuple[float, float, float, float]]] = {}
    for i, rel in enumerate(PARTS[part][1]):
        if rel not in _label_cache:
            path = download(rel, cache, optional=i > 0)
            _label_cache[rel] = read_labels(path) if path else {}
        merged.update(_label_cache[rel])
    return merged


def analyze(labels: dict[str, list[tuple[float, float, float, float]]], title: str, size: int = 640) -> None:
    buckets = Counter()
    per_image = Counter()
    for boxes in labels.values():
        per_image[min(len(boxes), 4)] += 1
        if not boxes:
            continue
        biggest = max(math.sqrt(w * h) * size for _, _, w, h in boxes)
        buckets["under 16 px" if biggest < 16 else "16-32 px" if biggest < 32 else "32-64 px" if biggest < 64 else "64-128 px" if biggest < 128 else "over 128 px"] += 1
    n = len(labels)
    print(f"{title}: {n} labelled images (sizes assume {size}x{size} images)")
    for k in ("under 16 px", "16-32 px", "32-64 px", "64-128 px", "over 128 px"):
        print(f"  largest drone {k:>12}: {buckets[k]:6d} images ({buckets[k] / max(1, n):5.1%})")
    print(f"  no drone: {per_image[0]} | 1 drone: {per_image[1]} | 2: {per_image[2]} | 3: {per_image[3]} | 4+: {per_image[4]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true", help="only download the label files and print drone sizes")
    ap.add_argument("--parts", default="test", help=f"comma list from {list(PARTS)}")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--cache-dir", type=Path, default=Path(__file__).resolve().parent.parent / "_downloads",
        help="where the zip files are kept (default: WorkSpace/_downloads, ignored by git)",
    )
    ap.add_argument("--prefix", default="ser")
    ap.add_argument("--max-side", type=float, default=64.0, help="skip images whose largest drone is bigger than this (px); 0 = keep all")
    ap.add_argument("--min-side", type=float, default=4.0, help="ignore boxes smaller than this (px)")
    ap.add_argument("--max-images", type=int, default=4000, help="images with drones to keep, over all parts; 0 = all")
    ap.add_argument("--negatives-ratio", type=float, default=0.1, help="drone-free images kept, as a share of images with drones")
    ap.add_argument("--start-id", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.analyze:
        for split in ("test", "train"):
            merged: dict[str, list[tuple[float, float, float, float]]] = {}
            for p in PARTS:
                if p.startswith(split):
                    merged.update(labels_for(p, args.cache_dir))
            analyze(merged, f"{split} labels")
        return
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (or use --analyze)")
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    for p in parts:
        if p not in PARTS:
            raise SystemExit(f"unknown part {p!r}; choose from {list(PARTS)}")

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_part = math.ceil(args.max_images / len(parts)) if args.max_images else 0
    images, anns = [], []
    img_id = ann_id = args.start_id
    why = Counter()
    used: set[str] = set()
    for p in parts:
        labels = labels_for(p, args.cache_dir)
        with zipfile.ZipFile(download(PARTS[p][0], args.cache_dir)) as z:
            members = [n for n in z.namelist() if n.lower().endswith(IMG_EXT)]
            rng.shuffle(members)
            kept_pos = kept_neg = 0
            for name in members:
                if per_part and kept_pos >= per_part:
                    break
                stem = Path(name).stem
                if stem not in labels:
                    why["no label file"] += 1
                    continue
                norm = labels[stem]
                if not norm and kept_neg >= args.negatives_ratio * max(1, kept_pos):
                    why["extra drone-free image"] += 1
                    continue
                data = z.read(name)
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
                        if not name.lower().endswith((".jpg", ".jpeg")):
                            buf = io.BytesIO()
                            im.convert("RGB").save(buf, "JPEG", quality=92)
                            data = buf.getvalue()
                except OSError:
                    why["unreadable image"] += 1
                    continue
                boxes = [b for b in (clip_box([x * w, y * h, bw * w, bh * h], w, h) for x, y, bw, bh in norm) if b]
                sides = [math.sqrt(b[2] * b[3]) for b in boxes]
                if args.max_side and any(s > args.max_side for s in sides):
                    why["drone too large"] += 1
                    continue
                boxes = [b for b, s in zip(boxes, sides) if s >= args.min_side]
                if norm and not boxes:
                    why["only unusably small boxes"] += 1
                    continue
                file_name = f"{args.prefix}_{p}_{re.sub(r'[^A-Za-z0-9._-]', '_', stem)[:120]}.jpg"
                if file_name in used:
                    why["duplicate name"] += 1
                    continue
                used.add(file_name)
                (args.out_dir / file_name).write_bytes(data)
                images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})
                for b in boxes:
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": 0, "bbox": b, "area": round(b[2] * b[3], 2), "iscrowd": 0})
                    ann_id += 1
                img_id += 1
                kept_pos += bool(boxes)
                kept_neg += not boxes
            print(f"{p}: kept {kept_pos} images with drones and {kept_neg} without, out of {len(members)}")
    write_dataset(args.out_dir, images, anns)
    print("left out:", dict(why))
    print_summary(str(args.out_dir), summarize(args.out_dir))
    print("Look at ~50 images before training on them: these labels come from community datasets.")


if __name__ == "__main__":
    main()
