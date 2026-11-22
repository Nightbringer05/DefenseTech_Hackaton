"""Noise test for Challenge 5 wake models: real drones in real noise at six levels, and the same noise alone.

    python run_noise_test.py --model my_model.onnx
    python run_noise_test.py --model mine=a.onnx --model other=b.onnx
    python run_noise_test.py --predictions mine=preds.jsonl      # a starter-kit prediction file instead of a model

A model is an ONNX file with the starter-kit contract: input [1, 3, 224, 224] in [0, 1], outputs
wake_logit and sound_class_logits, wake threshold in the ONNX metadata (override: --threshold 0.6).
A prediction file is what `predict_onnx_baseline.py acoustic --challenge-dir <this folder> --split valid`
writes: one JSON per line with id, wake_triggered, sound_class.

What is reported
    drones caught   share of drone clips that wake the camera, per drone level relative to the noise
                    (+10 dB = drone clearly louder than the noise ... -15 dB = drone buried in it)
    false alarms    share of noise-only clips that wake the camera, total and per kind of noise
    score           mean of "drones caught" over the six noise levels, minus the false-alarm rate.
                    1 = perfect; 0 = a model that always wakes, never wakes, or wakes at random.
    at 5 % FA       drones caught if the threshold were set to give 5 % false alarms on this test
                    (compares models independently of their threshold; needs a model, not a prediction file)
    4-class         accuracy of the sound class, averaged over the four classes (3 clips in 4 are drones,
                    so a model that always answers drone_buzz gets 25 %, not 76 %)

The ranking at the end is sorted by score and includes the models stored in leaderboard.json.
Needs: numpy, Pillow, onnxruntime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")
KINDS = ("ambient", "speech", "vehicle", "tonal_machine", "explosion", "gunfire")


def load_rows() -> list[dict]:
    p = HERE / "dataset" / "valid" / "labels.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_onnx(path: Path, rows: list[dict], threshold: float | None) -> dict:
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    thr = threshold if threshold is not None else float(meta.get("wake_threshold", 0.5))
    if threshold is None and "wake_threshold" not in meta:
        print(f"  ! {path.name}: no wake_threshold in the ONNX metadata, using 0.5 (set one with --threshold)")
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    wake_i = next((i for i, o in enumerate(outs) if "wake" in o.name.lower()), 0)
    cls_i = next((i for i, o in enumerate(outs) if "class" in o.name.lower()), None)
    if cls_i is None and len(outs) > 1:
        cls_i = 1 - wake_i if len(outs) == 2 else None
    p, c = [], []
    for r in rows:
        with Image.open(HERE / r["spectrogram"]) as im:
            x = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255.0
        res = sess.run(None, {inp.name: x.transpose(2, 0, 1)[None]})
        p.append(1.0 / (1.0 + np.exp(-float(np.ravel(res[wake_i])[0]))))
        c.append(int(np.argmax(np.ravel(res[cls_i]))) if cls_i is not None else -1)
    p = np.array(p)
    return {"wake": p >= thr, "prob": p, "cls": np.array(c), "threshold": thr}


def read_predictions(path: Path, rows: list[dict]) -> dict:
    by_id = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            by_id[r["id"]] = r
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} test clips have no prediction (first: {missing[0]})")
    wake = np.array([bool(by_id[r["id"]]["wake_triggered"]) for r in rows])
    cls = np.array([CLASSES.index(by_id[r["id"]]["sound_class"]) if by_id[r["id"]].get("sound_class") in CLASSES else -1 for r in rows])
    return {"wake": wake, "prob": None, "cls": cls, "threshold": None}


def score(res: dict, rows: list[dict]) -> dict:
    y = np.array([r["has_drone_audio"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    db = np.array([r["drone_vs_noise_db"] if r["drone_vs_noise_db"] is not None else np.nan for r in rows], float)
    kind = np.array([r["noise_kind"] or "" for r in rows])
    ycls = np.array([CLASSES.index(r["sound_class"]) for r in rows])
    wake = res["wake"]
    levels = sorted({int(v) for v in db[~np.isnan(db)]}, reverse=True)
    noise = cond == "noise_alone"
    out = {"threshold": res["threshold"],
           "caught_drone_alone": float(wake[cond == "drone_alone"].mean()),
           "caught_by_level": {lab(l): float(wake[db == l].mean()) for l in levels},
           "false_alarms": float(wake[noise].mean()),
           "false_alarms_by_kind": {k: float(wake[noise & (kind == k)].mean()) for k in KINDS},
           "wake_F1": float(2 * (wake & y).sum() / max(1, 2 * (wake & y).sum() + (wake & ~y).sum() + (~wake & y).sum())),
           "duty": float(wake.mean()),
           "class_accuracy": (float(np.mean([(res["cls"][ycls == k] == k).mean() for k in range(len(CLASSES))]))
                              if (res["cls"] >= 0).any() else None)}
    out["caught_mean"] = float(np.mean(list(out["caught_by_level"].values())))
    out["score"] = out["caught_mean"] - out["false_alarms"]
    if res["prob"] is not None:                                     # threshold-free view
        p = res["prob"]
        neg = np.sort(p[noise])[::-1]
        t5 = neg[int(np.floor(0.05 * len(neg)))]                    # 5 % of the noise-only clips score above it
        out["caught_at_5pct_fa"] = float(np.mean([(p[db == l] > t5).mean() for l in levels]))
        out["caught_at_5pct_fa_by_level"] = {lab(l): float((p[db == l] > t5).mean()) for l in levels}
        pos = p[cond == "drone_in_noise"]
        out["auc"] = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
    return out


def lab(level: int) -> str:
    return "0" if level == 0 else f"{level:+d}"


def pct(v) -> str:
    return "  n/a" if v is None else f"{100 * v:4.0f}%"


def show(results: dict[str, dict], mine: set[str]) -> None:
    names = sorted(results, key=lambda n: -results[n]["score"])
    levels = list(next(iter(results.values()))["caught_by_level"])
    w = max(12, max(len(n) for n in names) + 2)
    print("\nDrones caught, by drone level relative to the noise (dB)                      noise alone")
    print(f"{'model':<{w}} {'alone':>6} " + " ".join(f"{l:>5}" for l in levels) + f"  {'mean':>5}   {'false alarms':>12}   {'score':>6}  {'at 5% FA':>8}  {'4-class':>7}")
    for n in names:
        m = results[n]
        print(f"{n + ('  <-' if n in mine else ''):<{w}} {pct(m['caught_drone_alone']):>6} " + " ".join(pct(m["caught_by_level"][l]) for l in levels)
              + f"  {pct(m['caught_mean'])}   {pct(m['false_alarms']):>12}   {m['score']:>6.2f}  {pct(m.get('caught_at_5pct_fa')):>8}  {pct(m['class_accuracy']):>7}")
    print("\nFalse alarms by kind of noise")
    print(f"{'model':<{w}} " + " ".join(f"{k:>13}" for k in KINDS))
    for n in names:
        print(f"{n:<{w}} " + " ".join(f"{pct(results[n]['false_alarms_by_kind'][k]):>13}" for k in KINDS))
    print("\nRanking (score = mean drones caught over the six noise levels - false alarms)")
    for i, n in enumerate(names, 1):
        m = results[n]
        thr = "" if m["threshold"] is None else f", threshold {m['threshold']:.2f}"
        print(f"  {i}. {n:<{w}} {m['score']:.2f}   catches {100 * m['caught_mean']:.0f}% of drones, wakes on {100 * m['false_alarms']:.0f}% of noise{thr}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="[NAME=]FILE.onnx")
    ap.add_argument("--predictions", action="append", default=[], metavar="[NAME=]FILE.jsonl")
    ap.add_argument("--threshold", type=float, default=None, help="wake threshold for every --model (default: the one in each ONNX)")
    ap.add_argument("--no-leaderboard", action="store_true", help="do not show the models stored in leaderboard.json")
    ap.add_argument("--save-leaderboard", action="store_true", help="store these results in leaderboard.json")
    ap.add_argument("--out-json", type=Path, default=None, help="write the full results of this run")
    args = ap.parse_args()
    if not args.model and not args.predictions and args.no_leaderboard:
        ap.error("give --model or --predictions")
    rows = load_rows()
    n_drone = sum(r["has_drone_audio"] for r in rows)
    print(f"noise test: {len(rows)} clips, {n_drone} with a drone, {len(rows) - n_drone} noise only")

    def split(spec: str) -> tuple[str, Path]:
        name, _, path = spec.rpartition("=")
        return (name or Path(path).stem), Path(path)

    results = {}
    for spec in args.model:
        name, path = split(spec)
        print(f"running {name} ...", flush=True)
        results[name] = score(run_onnx(path, rows, args.threshold), rows)
    for spec in args.predictions:
        name, path = split(spec)
        results[name] = score(read_predictions(path, rows), rows)
    mine = set(results)
    board = HERE / "leaderboard.json"
    stored = json.loads(board.read_text(encoding="utf-8")) if board.exists() else {}
    if args.save_leaderboard:
        stored.update(results)
        board.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    if args.out_json:
        args.out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    if not args.no_leaderboard:
        results = {**stored, **results}
    if not results:
        raise SystemExit("nothing to show: give --model or --predictions")
    show(results, mine if len(results) > len(mine) else set())


if __name__ == "__main__":
    main()
