#!/usr/bin/env python3
"""Inference du Nano (base etendue) : detection, retrecissement, ByteTrack.

Le modele ONNX, les seuils et les parametres du tracker sont a cote de ce
fichier. Une image est etiree en 384x384, sans bandes noires.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CLASSES = CFG["classes"]
SIZE = int(CFG["input_size"])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)
_STD_POS = 1.0 / 20.0
_STD_VEL = 1.0 / 160.0
_MIN_SIZE = 8.0


def preprocess(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    nchw = np.transpose(arr, (2, 0, 1))[None, ...]
    return (nchw - MEAN) / STD


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def decode(dets: np.ndarray, labels: np.ndarray, width: int, height: int) -> list[list[float]]:
    """dets (300, 4) cxcywh normalise, labels (300, 4) logits. Retour [classe, score, x, y, w, h]."""
    ncls = len(CLASSES)
    probs = _sigmoid(labels[:, :ncls]).reshape(-1)
    k = min(100, len(probs))
    chosen = np.argpartition(-probs, k - 1)[:k]
    chosen = chosen[np.argsort(-probs[chosen])]
    raw: list[tuple[int, float, list[float]]] = []
    for flat in chosen:
        score = float(probs[flat])
        if score < CFG["score_min"]:
            continue
        q, cls = divmod(int(flat), ncls)
        cx, cy, bw, bh = (float(v) for v in dets[q])
        x1 = max(0.0, min(width - 1.0, (cx - bw / 2.0) * width))
        y1 = max(0.0, min(height - 1.0, (cy - bh / 2.0) * height))
        x2 = max(x1 + 1.0, min(float(width), (cx + bw / 2.0) * width))
        y2 = max(y1 + 1.0, min(float(height), (cy + bh / 2.0) * height))
        raw.append((cls, score, [x1, y1, x2 - x1, y2 - y1]))
    kept: list[tuple[int, float, list[float]]] = []
    for cand in sorted(raw, key=lambda item: -item[1]):
        if any(_iou(np.array(cand[2]), np.array(prev[2])) >= CFG["nms_iou"] for prev in kept):
            continue
        kept.append(cand)
    out = []
    factor = float(CFG["shrink_factor"])
    cutoff = float(CFG["shrink_cutoff_px"])
    for cls, score, (x, y, w, h) in kept[: int(CFG["max_dets"])]:
        if max(w, h) < cutoff:
            x = x + w * (1.0 - factor) / 2.0
            y = y + h * (1.0 - factor) / 2.0
            w *= factor
            h *= factor
        out.append([cls, score, x, y, w, h])
    return out


class Detector:
    def __init__(self, onnx: Path) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def frame(self, bgr: np.ndarray) -> list[list[float]]:
        h, w = bgr.shape[:2]
        outs = dict(zip(
            self.out_names,
            self.sess.run(self.out_names, {self.in_name: preprocess(bgr)}),
            strict=True,
        ))
        return decode(outs["dets"][0], outs["labels"][0], w, h)


def _kf_initiate(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, h = max(c[2], _MIN_SIZE), max(c[3], _MIN_SIZE)
    std = np.array(
        [2 * _STD_POS * w, 2 * _STD_POS * h, 2 * _STD_POS * w, 2 * _STD_POS * h]
        + [10 * _STD_VEL * w, 10 * _STD_VEL * h, 10 * _STD_VEL * w, 10 * _STD_VEL * h]
    )
    return np.concatenate([c, np.zeros(4)]), np.diag(std**2)


def _kf_predict(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, h = max(mean[2], _MIN_SIZE), max(mean[3], _MIN_SIZE)
    std = np.array(
        [_STD_POS * w, _STD_POS * h, _STD_POS * w, _STD_POS * h]
        + [_STD_VEL * w, _STD_VEL * h, _STD_VEL * w, _STD_VEL * h]
    )
    return _F @ mean, _F @ cov @ _F.T + np.diag(std**2)


def _kf_update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, h = max(mean[2], _MIN_SIZE), max(mean[3], _MIN_SIZE)
    r = np.diag(np.square([_STD_POS * w, _STD_POS * h, _STD_POS * w, _STD_POS * h]))
    s = _H @ cov @ _H.T + r
    k = np.linalg.solve(s, _H @ cov).T
    return mean + k @ (z - _H @ mean), cov - k @ _H @ cov


def _center(b: np.ndarray) -> np.ndarray:
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, b[2], b[3]], dtype=np.float64)


def _inflate(boxes: np.ndarray, expand: float) -> np.ndarray:
    if expand == 0.0 or len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] * (1.0 + expand)
    out[:, 3] = boxes[:, 3] * (1.0 + expand)
    out[:, 0] = boxes[:, 0] - (out[:, 2] - boxes[:, 2]) / 2.0
    out[:, 1] = boxes[:, 1] - (out[:, 3] - boxes[:, 3]) / 2.0
    return out


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = bx1 + b[:, 2][None, :], by1 + b[:, 3][None, :]
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass(frozen=True)
class TrackerParams:
    high_thr: float
    low_thr: float
    match_iou: float
    low_match_iou: float
    track_buffer: int
    min_hits: int
    expand: float


class _Track:
    def __init__(self, tid: int, det: np.ndarray) -> None:
        self.tid = tid
        self.mean, self.cov = _kf_initiate(_center(det[2:6]))
        self.state = "tracked"
        self.hits = 1
        self.tsu = 0
        self.votes: dict[int, float] = {}
        self._observe(det)

    def _observe(self, det: np.ndarray) -> None:
        self.last_box = det[2:6].copy()
        self.last_score = float(det[1])
        cls = int(det[0])
        self.votes[cls] = self.votes.get(cls, 0.0) + float(det[1])

    def predict(self) -> None:
        self.tsu += 1
        if self.state != "tracked":
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        self.mean, self.cov = _kf_predict(self.mean, self.cov)

    def update(self, det: np.ndarray) -> None:
        self.mean, self.cov = _kf_update(self.mean, self.cov, _center(det[2:6]))
        self.state = "tracked"
        self.hits += 1
        self.tsu = 0
        self._observe(det)

    def pred_box(self) -> np.ndarray:
        w, h = max(self.mean[2], 1.0), max(self.mean[3], 1.0)
        return np.array([self.mean[0] - w / 2.0, self.mean[1] - h / 2.0, w, h])

    @property
    def cls(self) -> int:
        return max(self.votes, key=self.votes.get)


def _associate(tracks, det_boxes, thr, expand):
    n_t, n_d = len(tracks), len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))
    t_boxes = _inflate(np.stack([t.pred_box() for t in tracks]), expand)
    iou = _iou_matrix(t_boxes, _inflate(det_boxes, expand))
    cost = np.where(iou >= thr, 1.0 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= thr]
    used_t = {r for r, _ in matches}
    used_d = {c for _, c in matches}
    return (
        matches,
        [i for i in range(n_t) if i not in used_t],
        [j for j in range(n_d) if j not in used_d],
    )


class ByteTracker:
    def __init__(self, params: TrackerParams) -> None:
        self.p = params
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, dets: np.ndarray) -> list[tuple[int, np.ndarray, float, int]]:
        p = self.p
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        high = dets[dets[:, 1] >= p.high_thr]
        low = dets[(dets[:, 1] >= p.low_thr) & (dets[:, 1] < p.high_thr)]
        for t in self.tracks:
            t.predict()
        tentative = [t for t in self.tracks if t.state == "tracked" and t.hits < 2]
        pool = [t for t in self.tracks if not (t.state == "tracked" and t.hits < 2)]
        m1, u_trk, u_det = _associate(pool, high[:, 2:6], p.match_iou, p.expand)
        for i, j in m1:
            pool[i].update(high[j])
        rem = [pool[i] for i in u_trk if pool[i].state == "tracked"]
        m2, u_rem, _ = _associate(rem, low[:, 2:6], p.low_match_iou, p.expand)
        for i, j in m2:
            rem[i].update(low[j])
        for i in u_rem:
            rem[i].state = "lost"
        left = high[u_det]
        m3, u_ten, u_left = _associate(tentative, left[:, 2:6], p.match_iou, p.expand)
        for i, j in m3:
            tentative[i].update(left[j])
        for i in u_ten:
            tentative[i].state = "removed"
        for j in u_left:
            self.tracks.append(_Track(self._next_id, left[j]))
            self._next_id += 1
        self.tracks = [
            t for t in self.tracks
            if t.state != "removed" and not (t.state == "lost" and t.tsu > p.track_buffer)
        ]
        return [
            (t.tid, t.last_box, t.last_score, t.cls)
            for t in self.tracks
            if t.state == "tracked" and t.tsu == 0 and t.hits >= p.min_hits
        ]


def draw_tracks(bgr: np.ndarray, tracks: list[tuple[int, np.ndarray, float, int]]) -> np.ndarray:
    img = bgr.copy()
    for tid, box, _score, cls in tracks:
        x, y, w, h = (int(round(v)) for v in box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{CLASSES[cls]} #{tid}"
        cv2.putText(img, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def draw_dets(bgr: np.ndarray, dets: list[list[float]]) -> np.ndarray:
    img = bgr.copy()
    for cls, score, x, y, w, h in dets:
        x, y, w, h = (int(round(v)) for v in (x, y, w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{CLASSES[int(cls)]} {score:.2f}"
        cv2.putText(img, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def run_video(det: Detector, tracker: ByteTracker, cap: cv2.VideoCapture, out_path: Path | None) -> None:
    writer = None
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        tracks = tracker.update(det.frame(bgr))
        frame = draw_tracks(bgr, tracks)
        if out_path is not None and writer is None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(frame)
        n += 1
        if n == 1 or n % 50 == 0:
            print(f"frame {n}, pistes {len(tracks)}", flush=True)
    if writer is not None:
        writer.release()
    print(f"{n} frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="image ou video")
    parser.add_argument("--camera", type=int, help="indice de la camera, par exemple 0")
    parser.add_argument("-o", "--out", type=Path, help="image ou video annotee")
    args = parser.parse_args()
    if (args.source is None) == (args.camera is None):
        parser.error("donner une image, une video, ou --camera")

    params = TrackerParams(**CFG["tracker"])
    det = Detector(HERE / CFG["weights"])
    tracker = ByteTracker(params)

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"camera {args.camera} inaccessible")
        run_video(det, tracker, cap, args.out)
        cap.release()
        return

    source = Path(args.source)
    bgr = cv2.imread(str(source))
    if bgr is not None:
        dets = det.frame(bgr)
        for cls, score, x, y, w, h in dets:
            print(f"{CLASSES[int(cls)]}  score {score:.2f}  {[round(v, 1) for v in (x, y, w, h)]}")
        if not dets:
            print("aucune detection")
        if args.out is not None:
            cv2.imwrite(str(args.out), draw_dets(bgr, dets))
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"fichier illisible : {source}")
    out = args.out or source.with_name(source.stem + "_pistes.mp4")
    run_video(det, tracker, cap, out)
    cap.release()
    print(f"video ecrite : {out}")


if __name__ == "__main__":
    main()
