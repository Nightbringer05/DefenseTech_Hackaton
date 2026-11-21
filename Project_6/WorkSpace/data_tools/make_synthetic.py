"""Generate extra labelled training images by pasting real targets, shrunk, onto real frames.

Every label is exact because we place the target ourselves. Uses ONLY the train split.

    python make_synthetic.py --dataset-dir ../../InputData/dataset --out-dir ../extra_data/synth --n 800

Output is a folder in the challenge format (JPEGs + _annotations.coco.json, ids 0/1/2).
Needs only numpy and pillow.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CATEGORIES = [
    {"id": 0, "name": "drone_air", "supercategory": "none"},
    {"id": 1, "name": "airplane", "supercategory": "none"},
    {"id": 2, "name": "helicopter", "supercategory": "none"},
]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32) / 255.0


def make_stamp(img: np.ndarray, bbox: list[float], max_noise: float) -> dict | None:
    """Cut one target out of its frame as a 'difference from background' patch.

    Works when the target sits on a smooth background (sky): the background is modelled as a
    colour plane fitted on the ring around the box. Returns None when the ring is cluttered.
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    if w < 4 or h < 4:
        return None
    margin = int(round(max(4.0, 0.4 * max(w, h))))
    x0, y0 = int(math.floor(x)) - margin, int(math.floor(y)) - margin
    x1, y1 = int(math.ceil(x + w)) + margin, int(math.ceil(y + h)) + margin
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1)
    if (cx1 - cx0) * (cy1 - cy0) < 0.7 * (x1 - x0) * (y1 - y0):
        return None  # target too close to the frame edge
    patch = img[cy0:cy1, cx0:cx1]
    ph, pw = patch.shape[:2]
    bx, by = x - cx0, y - cy0  # box inside the patch

    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    grow = max(1.0, 0.12 * max(w, h))
    inside = (xx >= bx - grow) & (xx < bx + w + grow) & (yy >= by - grow) & (yy < by + h + grow)
    ring = ~inside
    if ring.sum() < 40:
        return None

    # background = colour plane fitted on the ring (handles sky gradients)
    a_mat = np.stack([xx[ring], yy[ring], np.ones(int(ring.sum()), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(a_mat, patch[ring], rcond=None)
    bg = (np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ coef).astype(np.float32)
    diff = patch - bg
    noise = float(np.abs(diff[ring]).max(axis=1).std() + np.abs(diff[ring]).max(axis=1).mean())
    if noise > max_noise:
        return None  # cluttered surroundings: a clean cut-out is not possible

    mag = np.abs(diff).max(axis=2)
    t0 = max(4.0, 2.5 * noise)
    alpha = np.clip((mag - t0) / 10.0, 0.0, 1.0) * blur_mask(inside.astype(np.float32), max(0.6, 0.05 * max(w, h)))
    box_px = (xx >= bx) & (xx < bx + w) & (yy >= by) & (yy < by + h)
    strong = alpha[box_px] > 0.5
    if strong.mean() < 0.08 or float(mag[box_px][strong].mean()) < 14.0:
        return None  # target does not stand out from its background
    return {"delta": diff * alpha[..., None], "box": (bx, by, w, h), "side": math.sqrt(w * h)}


def resize_delta(delta: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    chans = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(delta[..., c], dtype=np.float32)).resize((new_w, new_h), Image.Resampling.BOX),
            dtype=np.float32,
        )
        for c in range(3)
    ]
    return np.stack(chans, axis=-1)


def iou_free(box: tuple[float, float, float, float], others: list[list[float]], pad: float) -> bool:
    x, y, w, h = box
    for ox, oy, ow, oh in others:
        if x < ox + ow + pad and ox < x + w + pad and y < oy + oh + pad and oy < y + h + pad:
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True, help="challenge dataset/ (only train/ is read)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=800, help="number of images to generate")
    p.add_argument("--min-size", type=float, default=6.0, help="smallest pasted target, px (sqrt of area)")
    p.add_argument("--max-size", type=float, default=32.0, help="largest pasted target, px")
    p.add_argument("--bg-dir", type=Path, default=None, help="optional folder of extra images WITHOUT any aircraft")
    p.add_argument("--class-weights", default="0.30,0.35,0.35", help="drone,airplane,helicopter sampling weights")
    p.add_argument("--max-noise", type=float, default=7.0, help="reject source targets on busier backgrounds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="syn")
    p.add_argument("--start-id", type=int, default=100000, help="first image/annotation id (avoid clashes on merge)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_dir = args.dataset_dir / "train"
    doc = json.loads((train_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = {im["id"]: im for im in doc["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for a in doc["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # 1) collect cut-outs ("stamps") of real targets
    stamps: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    cache: dict[int, np.ndarray] = {}
    for a in doc["annotations"]:
        im = images[a["image_id"]]
        if a["image_id"] not in cache:
            cache.clear()
            cache[a["image_id"]] = load_rgb(train_dir / im["file_name"])
        st = make_stamp(cache[a["image_id"]], a["bbox"], args.max_noise)
        if st is not None:
            st["video_id"] = im.get("video_id")
            stamps[a["category_id"]].append(st)
    for c in CATEGORIES:
        vids = sorted({s["video_id"] for s in stamps[c["id"]] if s["video_id"] is not None})
        print(f"usable cut-outs {c['name']:>10}: {len(stamps[c['id']]):4d} from clips {vids}")
    weights = [float(v) if stamps[i] else 0.0 for i, v in enumerate(args.class_weights.split(","))]
    if sum(weights) == 0:
        raise SystemExit("No usable cut-outs found; raise --max-noise")

    # 2) backgrounds: every train frame (its own labels are kept) + optional extra folder
    backgrounds: list[tuple[Path, list[list[float]], list[int]]] = []
    for im in doc["images"]:
        own = anns_by_img.get(im["id"], [])
        backgrounds.append((train_dir / im["file_name"], [list(a["bbox"]) for a in own], [a["category_id"] for a in own]))
    extra: list[tuple[Path, list[list[float]], list[int]]] = []
    if args.bg_dir is not None:
        for pth in sorted(args.bg_dir.rglob("*")):
            if pth.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                extra.append((pth, [], []))
        print(f"extra backgrounds: {len(extra)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_images, out_anns = [], []
    img_id, ann_id = args.start_id, args.start_id
    pasted_sizes: dict[int, list[float]] = {0: [], 1: [], 2: []}
    order: list[tuple[Path, list[list[float]], list[int]]] = []

    while len(out_images) < args.n:
        if not order:
            order = backgrounds[:]
            rng.shuffle(order)
        use_extra = bool(extra) and rng.random() < 0.5
        path, boxes, cats = rng.choice(extra) if use_extra else order.pop()
        frame = load_rgb(path)
        boxes = [list(b) for b in boxes]
        cats = list(cats)
        if rng.random() < 0.5:  # mirror the frame so it is not a plain copy of a train image
            frame = frame[:, ::-1].copy()
            boxes = [[frame.shape[1] - b[0] - b[2], b[1], b[2], b[3]] for b in boxes]
        h_img, w_img = frame.shape[:2]
        new_flags = [False] * len(boxes)

        for _ in range(1 if rng.random() < 0.7 else 2):
            cls = rng.choices([0, 1, 2], weights=weights)[0]
            st = rng.choice(stamps[cls])
            target = math.exp(rng.uniform(math.log(args.min_size), math.log(args.max_size)))
            f = min(1.0, target / st["side"])  # only ever shrink a real target
            delta = st["delta"][:, ::-1] if rng.random() < 0.5 else st["delta"]
            bx, by, bw, bh = st["box"]
            if delta is not st["delta"]:
                bx = delta.shape[1] - bx - bw
            new_w, new_h = max(3, round(delta.shape[1] * f)), max(3, round(delta.shape[0] * f))
            if new_w >= w_img or new_h >= h_img:
                continue
            small = resize_delta(np.ascontiguousarray(delta), new_w, new_h) * rng.uniform(0.75, 1.15)
            fx, fy = new_w / delta.shape[1], new_h / delta.shape[0]

            # try a few places; prefer smooth areas (sky) but keep some clutter
            best = None
            for _try in range(10):
                px, py = rng.randrange(0, w_img - new_w), rng.randrange(0, h_img - new_h)
                box = (px + bx * fx, py + by * fy, bw * fx, bh * fy)
                if not iou_free(box, boxes, pad=8.0):
                    continue
                dst = frame[py : py + new_h, px : px + new_w]
                out = np.clip(dst + small, 0.0, 255.0)
                change = np.sort(np.abs(out - dst).max(axis=2).ravel())
                visible = float(change[int(0.9 * len(change)) :].mean())
                smooth = float(dst.std(axis=(0, 1)).mean())  # how busy the background is here
                if visible < max(12.0, 2.0 * smooth):
                    continue  # the target would not stand out here (dark on dark, or lost in foliage)
                if best is None or smooth < best[0]:
                    best = (smooth, px, py, box, out)
                if rng.random() < 0.3:
                    break
            if best is None:
                continue
            _, px, py, box, out = best
            frame[py : py + new_h, px : px + new_w] = out
            boxes.append([round(v, 2) for v in box])
            cats.append(cls)
            new_flags.append(True)
            pasted_sizes[cls].append(math.sqrt(box[2] * box[3]))

        if not any(new_flags):
            continue
        pil = Image.fromarray(frame.astype(np.uint8))
        if rng.random() < 0.5:  # soften slightly, like video compression does
            pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
        name = f"{args.prefix}_{img_id:06d}.jpg"
        pil.save(args.out_dir / name, quality=rng.randint(80, 94))
        out_images.append({"id": img_id, "file_name": name, "width": w_img, "height": h_img, "source": path.name})
        for b, c, is_new in zip(boxes, cats, new_flags):
            out_anns.append(
                {"id": ann_id, "image_id": img_id, "category_id": c, "bbox": [float(v) for v in b],
                 "area": float(b[2] * b[3]), "iscrowd": 0, "synthetic": is_new}
            )
            ann_id += 1
        img_id += 1

    (args.out_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": out_images, "annotations": out_anns, "categories": CATEGORIES}), encoding="utf-8"
    )
    print(f"wrote {len(out_images)} images, {len(out_anns)} boxes -> {args.out_dir}")
    for c in CATEGORIES:
        s = pasted_sizes[c["id"]]
        if s:
            print(f"  pasted {c['name']:>10}: {len(s):4d}  size px min/median/max = {min(s):.0f}/{float(np.median(s)):.0f}/{max(s):.0f}")


if __name__ == "__main__":
    main()
