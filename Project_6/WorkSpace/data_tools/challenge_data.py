"""Bring any labelled data into the challenge format, merge it into train, and check it.

Challenge format = a folder of JPEGs + _annotations.coco.json with categories
0 drone_air, 1 airplane, 2 helicopter and boxes [x, y, w, h] in pixels.

  check     validate a folder and print what is inside
  convert   a COCO export (Roboflow, CVAT, ...) -> challenge format, with class remapping
  from-mot  a video + a MOT text file (tracker output or ground truth) -> challenge format
  negatives a folder of images with NO aircraft -> challenge format (no labelling needed)
  merge     challenge dataset + extra folders -> a new dataset with a bigger train/

Needs only pillow (and ffmpeg on PATH for from-mot).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

NAMES =["drone_air", "airplane", "helicopter"]
CATEGORIES = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(NAMES)]
ANN_FILE = "_annotations.coco.json"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataset(out_dir: Path, images: list[dict], anns: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ANN_FILE).write_text(
        json.dumps({"images": images, "annotations": anns, "categories": CATEGORIES}), encoding="utf-8"
    )


def place(src: Path, dst: Path) -> None:
    """Hard-link when possible (instant, no extra disk), else copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def guess_class(name: str) -> int | None:
    n = name.lower().replace("-", "_").replace(" ", "_")
    if "heli" in n:
        return 2
    if any(k in n for k in ("drone", "uav", "quadcopter", "multirotor", "fpv")):
        return 0
    if any(k in n for k in ("airplane", "aeroplane", "plane", "jet", "fixed_wing_aircraft")):
        return 1
    return None


def clip_box(b: list[float], w: int, h: int) -> list[float] | None:
    x0, y0 = max(0.0, float(b[0])), max(0.0, float(b[1]))
    x1, y1 = min(float(w), float(b[0]) + float(b[2])), min(float(h), float(b[1]) + float(b[3]))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


# ---------------------------------------------------------------- check
def summarize(folder: Path, *, strict: bool = True) -> dict:
    doc = read_json(folder / ANN_FILE)
    problems: list[str] = []
    cats = sorted((c["id"], c["name"]) for c in doc.get("categories", []))
    if cats != list(enumerate(NAMES)):
        problems.append(f"categories are {cats}, expected {list(enumerate(NAMES))}")
    images = {im["id"]: im for im in doc["images"]}
    if len(images) != len(doc["images"]):
        problems.append("duplicate image ids")
    if len({im["file_name"] for im in doc["images"]}) != len(doc["images"]):
        problems.append("duplicate file names")
    missing = [im["file_name"] for im in doc["images"] if not (folder / im["file_name"]).is_file()]
    if missing:
        problems.append(f"{len(missing)} image files missing (first: {missing[0]})")
    if len({a["id"] for a in doc["annotations"]}) != len(doc["annotations"]):
        problems.append("duplicate annotation ids")
    per_class: dict[int, list[float]] = defaultdict(list)
    bad_boxes = 0
    with_box: set[int] = set()
    for a in doc["annotations"]:
        im = images.get(a["image_id"])
        x, y, w, h = a["bbox"]
        if im is None or w <= 0 or h <= 0 or x < -0.5 or y < -0.5 or x + w > im["width"] + 0.5 or y + h > im["height"] + 0.5:
            bad_boxes += 1
            continue
        with_box.add(a["image_id"])
        per_class[int(a["category_id"])].append(math.sqrt(w * h))
    if bad_boxes:
        problems.append(f"{bad_boxes} boxes are empty, outside the image, or point to a missing image")
    if strict and doc["images"]:
        sample = random.Random(0).sample(doc["images"], min(25, len(doc["images"])))
        wrong = 0
        for im in sample:
            if (folder / im["file_name"]).is_file():
                with Image.open(folder / im["file_name"]) as pil:
                    wrong += pil.size != (im["width"], im["height"])
        if wrong:
            problems.append(f"{wrong}/{len(sample)} sampled images have a different size than the JSON says")
    return {"images": len(images), "negatives": len(images) - len(with_box), "per_class": per_class, "problems": problems}


def print_summary(title: str, s: dict) -> None:
    print(f"{title}: {s['images']} images, {sum(len(v) for v in s['per_class'].values())} boxes, {s['negatives']} images without target")
    print(f"  {'class':>10} {'boxes':>6} {'<16px':>6} {'16-32':>6} {'32-96':>6} {'>96px':>6}")
    for cid, name in enumerate(NAMES):
        v = s["per_class"].get(cid, [])
        print(
            f"  {name:>10} {len(v):6d} {sum(x < 16 for x in v):6d} {sum(16 <= x < 32 for x in v):6d} "
            f"{sum(32 <= x < 96 for x in v):6d} {sum(x >= 96 for x in v):6d}"
        )
    for p in s["problems"]:
        print(f"  PROBLEM: {p}")
    if not s["problems"]:
        print("  format OK")


def cmd_check(args: argparse.Namespace) -> None:
    s = summarize(args.folder)
    print_summary(str(args.folder), s)
    if s["problems"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- convert
def cmd_convert(args: argparse.Namespace) -> None:
    doc = read_json(args.coco)
    img_dir = args.images_dir or args.coco.parent
    explicit: dict[str, int | None] = {}
    for pair in filter(None, (args.map or "").split(",")):
        src, _, dst = pair.partition("=")
        dst = dst.strip()
        if dst not in NAMES and dst != "":
            raise SystemExit(f"--map target {dst!r} must be one of {NAMES} or empty (to drop)")
        explicit[src.strip().lower()] = NAMES.index(dst) if dst else None

    cat_map: dict[int, int | None] = {}
    box_count = Counter(a["category_id"] for a in doc["annotations"])
    print("class mapping (check it!):")
    for c in doc["categories"]:
        key = str(c["name"]).strip().lower()
        target = explicit[key] if key in explicit else guess_class(key)
        cat_map[c["id"]] = target
        shown = NAMES[target] if target is not None else "(ignored: its boxes are removed)"
        print(f"  {c['name']!r:>28} [{box_count.get(c['id'], 0):6d} boxes] -> {shown}")

    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in doc["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    rng = random.Random(args.seed)
    positives, negatives = [], []
    for im in doc["images"]:
        mapped = [a for a in anns_by_img.get(im["id"], []) if cat_map.get(a["category_id"]) is not None]
        (positives if mapped else negatives).append(im)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if args.max_images:
        positives = positives[: args.max_images]
    negatives = negatives[: int(len(positives) * args.negatives_ratio)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_images, out_anns = [], []
    img_id = ann_id = args.start_id
    skipped = dropped_small = 0
    for im in positives + negatives:
        src = img_dir / im["file_name"]
        if not src.is_file():
            skipped += 1
            continue
        with Image.open(src) as pil:
            w, h = pil.size
        if (w, h) != (im.get("width", w), im.get("height", h)):
            skipped += 1  # JSON and file disagree (rotation flag, resized copy...): boxes cannot be trusted
            continue
        name = f"{args.prefix}_{Path(im['file_name']).stem}{src.suffix.lower()}"
        kept = []
        for a in anns_by_img.get(im["id"], []):
            cid = cat_map.get(a["category_id"])
            box = clip_box(a["bbox"], w, h) if cid is not None else None
            if box is None:
                continue
            if math.sqrt(box[2] * box[3]) < args.min_side:
                dropped_small += 1
                continue
            kept.append((cid, box))
        place(src, args.out_dir / name)
        out_images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        for cid, box in kept:
            out_anns.append({"id": ann_id, "image_id": img_id, "category_id": cid, "bbox": box, "area": round(box[2] * box[3], 2), "iscrowd": 0})
            ann_id += 1
        img_id += 1
    write_dataset(args.out_dir, out_images, out_anns)
    print(f"skipped {skipped} images (file missing or size mismatch), removed {dropped_small} boxes under {args.min_side} px")
    print_summary(str(args.out_dir), summarize(args.out_dir))


# ---------------------------------------------------------------- from-mot
def cmd_from_mot(args: argparse.Namespace) -> None:
    rows: dict[int, list[tuple[int, list[float], int]]] = defaultdict(list)
    for line in args.mot.read_text(encoding="utf-8").splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) < 6:
            continue
        conf = float(p[6]) if len(p) > 6 and p[6] else 1.0
        if conf < args.min_conf:
            continue
        cls = int(float(p[7])) - 1 if len(p) > 7 and p[7] and args.force_class is None else NAMES.index(args.force_class or "drone_air")
        if not 0 <= cls < len(NAMES):
            continue
        rows[int(float(p[0]))].append((int(float(p[1])), [float(v) for v in p[2:6]], cls))

    if args.min_track_len > 1:  # auto-labelling: a real target is followed for many frames, a false alarm is not
        length = Counter(tid for boxes in rows.values() for tid, _, _ in boxes)
        rows = {f: [b for b in boxes if length[b[0]] >= args.min_track_len] for f, boxes in rows.items()}
        kept = sum(v >= args.min_track_len for v in length.values())
        print(f"tracks kept: {kept} of {len(length)} (those seen in fewer than {args.min_track_len} frames are dropped)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / "_frames_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(args.video), "-fps_mode", "passthrough", "-q:v", "2", str(tmp / "f_%06d.jpg")],
        check=True,
    )
    out_images, out_anns = [], []
    img_id = ann_id = args.start_id
    for frame_no, fpath in enumerate(sorted(tmp.glob("f_*.jpg")), 1):
        boxes = rows.get(frame_no, [])
        if (frame_no - 1) % args.every != 0 or (not boxes and not args.keep_empty):
            continue
        with Image.open(fpath) as pil:
            w, h = pil.size
        name = f"{args.prefix}_{frame_no:06d}.jpg"
        shutil.copyfile(fpath, args.out_dir / name)
        out_images.append({"id": img_id, "file_name": name, "width": w, "height": h, "frame_index": frame_no - 1})
        for tid, b, cls in boxes:
            box = clip_box(b, w, h)
            if box is None:
                continue
            out_anns.append({"id": ann_id, "image_id": img_id, "category_id": cls, "bbox": box, "area": round(box[2] * box[3], 2), "iscrowd": 0, "track_id": tid})
            ann_id += 1
        img_id += 1
    shutil.rmtree(tmp)
    write_dataset(args.out_dir, out_images, out_anns)
    print_summary(str(args.out_dir), summarize(args.out_dir))


# ---------------------------------------------------------------- negatives
def cmd_negatives(args: argparse.Namespace) -> None:
    files = sorted(p for p in args.images_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if args.max_images:
        random.Random(args.seed).shuffle(files)
        files = sorted(files[: args.max_images])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for k, src in enumerate(files):
        with Image.open(src) as pil:
            rgb = ImageOps.exif_transpose(pil).convert("RGB")  # bake in phone rotation
        name = f"{args.prefix}_{k:06d}.jpg"
        rgb.save(args.out_dir / name, quality=92)
        images.append({"id": args.start_id + k, "file_name": name, "width": rgb.width, "height": rgb.height})
    write_dataset(args.out_dir, images, [])
    print_summary(str(args.out_dir), summarize(args.out_dir))


# ---------------------------------------------------------------- merge
def cmd_merge(args: argparse.Namespace) -> None:
    sources = [args.base / "train"] + list(args.add)
    if args.final_include_valid:
        sources.append(args.base / "valid")
        print("WARNING: valid/ is going into train/. Scores measured on valid are no longer honest for this model.")
    out_train = args.out / "train"
    out_train.mkdir(parents=True, exist_ok=True)
    images, anns, seen = [], [], set()
    img_id = ann_id = 1
    for src in sources:
        doc = read_json(src / ANN_FILE)
        cats = sorted((c["id"], c["name"]) for c in doc["categories"])
        if cats != list(enumerate(NAMES)):
            raise SystemExit(f"{src}: not in challenge format (categories {cats}); run 'convert' first")
        remap = {}
        for im in doc["images"]:
            if im["file_name"] in seen:
                raise SystemExit(f"file name clash on {im['file_name']} from {src}; use a different --prefix")
            seen.add(im["file_name"])
            place(src / im["file_name"], out_train / im["file_name"])
            remap[im["id"]] = img_id
            images.append({**im, "id": img_id})
            img_id += 1
        for a in doc["annotations"]:
            if a["image_id"] in remap:
                anns.append({**a, "id": ann_id, "image_id": remap[a["image_id"]]})
                ann_id += 1
        print(f"  + {len(doc['images']):5d} images from {src}")
    write_dataset(out_train, images, anns)
    for split in ("valid", "test"):  # unchanged, so scores stay comparable
        if (args.base / split).is_dir():
            (args.out / split).mkdir(parents=True, exist_ok=True)
            for f in (args.base / split).iterdir():
                if f.is_file():
                    place(f, args.out / split / f.name)
    print_summary(str(out_train), summarize(out_train))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate a folder in challenge format")
    c.add_argument("folder", type=Path)
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("convert", help="COCO export -> challenge format")
    c.add_argument("--coco", type=Path, required=True, help="the source COCO json")
    c.add_argument("--images-dir", type=Path, default=None, help="default: folder of the json")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--prefix", required=True, help="short unique name for this source, e.g. rf1")
    c.add_argument("--map", default="", help='override, e.g. "UAV=drone_air,aircraft=airplane,bird=" (empty = ignore)')
    c.add_argument("--max-images", type=int, default=0, help="keep at most this many images with targets")
    c.add_argument("--negatives-ratio", type=float, default=0.15, help="target-free images kept, as a share of images with targets")
    c.add_argument("--min-side", type=float, default=3.0, help="remove boxes smaller than this (px)")
    c.add_argument("--start-id", type=int, default=200000)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_convert)

    c = sub.add_parser("from-mot", help="video + MOT txt -> challenge format")
    c.add_argument("--video", type=Path, required=True)
    c.add_argument("--mot", type=Path, required=True, help="frame,id,x,y,w,h,conf,class(1-based),...")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--prefix", required=True)
    c.add_argument("--every", type=int, default=3, help="keep one frame out of N (neighbours are near-duplicates)")
    c.add_argument("--min-conf", type=float, default=0.0)
    c.add_argument("--min-track-len", type=int, default=1, help="auto-labelling: drop tracks seen in fewer frames than this")
    c.add_argument("--force-class", choices=NAMES, default=None, help="ignore the class column")
    c.add_argument("--keep-empty", action="store_true", help="also keep frames without box (only if you are sure they are empty)")
    c.add_argument("--start-id", type=int, default=300000)
    c.set_defaults(func=cmd_from_mot)

    c = sub.add_parser("negatives", help="folder of target-free images -> challenge format")
    c.add_argument("--images-dir", type=Path, required=True, help="images that contain NO drone, airplane or helicopter")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--prefix", required=True)
    c.add_argument("--max-images", type=int, default=0)
    c.add_argument("--start-id", type=int, default=400000)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_negatives)

    c = sub.add_parser("merge", help="challenge dataset + extra folders -> new dataset")
    c.add_argument("--base", type=Path, required=True, help="the challenge dataset/ folder")
    c.add_argument("--add", type=Path, action="append", default=[], help="extra folder in challenge format (repeatable)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--final-include-valid", action="store_true", help="LAST model only: also train on valid/")
    c.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
