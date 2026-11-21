"""Build training datasets in the challenge layout from MAD + synthetic drones.

Sources
  * MAD_dataset/  real military audio (7 labels), mapped onto the 4 challenge classes
  * synthetic     drones (and optional extra vehicles/explosions/ambient) mixed over
                  real non-drone backgrounds from the challenge train split and MAD

Output (paths in labels.jsonl are relative to the dataset root, like the challenge):
    <out>/<name>/audio/<split>/<id>.flac          16 kHz mono 16-bit, 5 s
    <out>/<name>/spectrograms/<split>/<id>.png    224x224 log-mel, challenge style
    <out>/<name>/dataset/train/labels.jsonl       MAD training + synthetic (+ real challenge train)
    <out>/<name>/dataset/valid/labels.jsonl       real challenge valid (threshold selection)
    <out>/<name>/dataset/mad_test/labels.jsonl    MAD test split, held out (optional)
    <out>/<name>/manifest.json

`train_audio_cnn_axelera.py --challenge-dir <out>/<name>` trains on it unchanged.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import drones, sources
from .propagate import PAD_S, acoular_available, propagate, sample_trajectory
from .spectro import spectrogram_image

CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")

# MAD label ids (Military Audio Dataset, Kim et al. 2024). Title keywords in training.csv agree.
MAD_LABELS = {"0": "communication", "1": "shooting", "2": "footsteps", "3": "shelling",
              "4": "vehicle", "5": "helicopter", "6": "fighter"}
DEFAULT_MAD_MAPPING = {"0": "skip",                # speech; the challenge screened speech out
                       "1": "explosion", "2": "ambient_outdoor", "3": "explosion",
                       "4": "vehicle_mechanical", "5": "vehicle_mechanical", "6": "vehicle_mechanical"}

DEFAULT_CONFIG = {
    "name": "mad_drones_v1",
    "comment": "",               # free text: purpose, changes, who made it; saved with the dataset
    "seed": 0,
    "sample_rate": 16000,
    "duration_s": 5.0,
    "engine": "numpy",            # "numpy" (fast) or "acoular"
    "audio_format": "flac",       # "flac" or "wav"
    "workers": max(1, (os.cpu_count() or 2) - 1),
    "mad": {
        "enabled": True,
        "mapping": DEFAULT_MAD_MAPPING,   # MAD label id -> challenge class or "skip"
        "max_per_label": 0,               # cap per MAD label for train (0 = all)
        "windows_per_clip": 1,            # 5 s windows cut from each clip (long clips give more)
        "short_clips": "pad",             # clips < 5 s: "pad" over a quiet real background, or "skip"
        "test_split": True,               # MAD test -> dataset/mad_test (held out)
        "as_drone_background": True,      # MAD clips as backgrounds under synthetic drones
    },
    "counts": {"drone_buzz": 2000, "vehicle_mechanical": 0, "explosion": 600, "ambient_outdoor": 0},
    "drone": {
        # airframe archetypes (see dronesynth/drones.py) and their sampling weights
        "types": {"mini": 0.2, "camera": 0.3, "fpv": 0.25, "heavy": 0.12, "agri": 0.05, "fixed_wing": 0.08},
        "payload_frac": [0.0, 0.6],    # share of the archetype's max payload
        "manoeuvres": {"steady": 0.5, "climb": 0.2, "aggressive": 0.3},
        "wind_mps": [0, 12],
        "temperature_c": [-10, 35],
        "elevation_m": [0, 1500],      # site elevation -> air density
        "flight_from_type": True,      # use each archetype's speed/altitude envelope
        "level_from_physics": True,    # louder airframes get higher SNR at the same range
        "harmonics": [6, 14],
        "harmonic_rolloff": [0.6, 1.4],
        "hiss": [0.05, 0.5],
        "motor_whine": [0.0, 0.4],
    },
    "vehicle": {"engine_rpm": [700, 3500]},
    "explosion": {                     # synthetic explosion-class clips
        "salvo_prob": 0.6,             # share that are salvos instead of a single blast
        "salvo_types": {"burst": 0.5, "volley": 0.25, "artillery": 0.25},
        "burst_rpm": [500, 950],       # automatic fire rate
        "rounds": [3, 30],             # rounds per burst
    },
    "flight": {
        "modes": {"flyby": 0.6, "hover": 0.25, "approach": 0.15},
        "altitude_m": [10, 120],
        "lateral_offset_m": [0, 80],
        "speed_mps": [3, 25],
    },
    "mix": {
        "snr_db_at_30m": [0, 25],      # drone/vehicle SNR at 30 m, scaled by 1/r with range
        "snr_clip_db": [-12, 30],
        "explosion_snr_db": [0, 30],
        "real_background_prob": 0.8,   # else procedural ambient bed
        "use_challenge_backgrounds": True,  # challenge train non-drone clips as backgrounds
        "background_dirs": [],         # any extra folders of .wav / .flac files
        "peak_dbfs": [-12, -1],
    },
    "include_real": True,              # copy the 48 real challenge train clips in
    "challenge_dir": "..",             # relative to the synth/ folder
}


def merge_config(user: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    def rec(dst, src):
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict) and k not in ("modes", "counts", "mapping", "salvo_types", "types", "manoeuvres"):
                rec(dst[k], v)
            elif isinstance(v, dict) and k == "mapping":
                dst[k] = {**dst.get(k, {}), **v}
            else:
                dst[k] = v
    rec(cfg, user)
    # backwards compatibility with the old flag
    old = (user or {}).get("real_negatives", {}).get("mad_per_class")
    if old:
        cfg["mad"]["enabled"], cfg["mad"]["max_per_label"] = True, int(old)
    return cfg


# ---------------------------------------------------------------- audio io

def read_audio(path: Path, sr: int) -> np.ndarray:
    x, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != sr:
        g = np.gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x


def audio_bytes(y: np.ndarray, sr: int, fmt: str = "flac") -> bytes:
    """16-bit PCM mono as FLAC (lossless, ~half the size) or WAV."""
    fmt = "wav" if fmt == "wav" else "flac"
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format=fmt.upper(), subtype="PCM_16")
    return buf.getvalue()


def _ext(cfg) -> str:
    return "wav" if cfg.get("audio_format") == "wav" else "flac"


def _crop(rng, x, n):
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(1, len(x)))))
    s = int(rng.integers(0, len(x) - n + 1))
    return x[s : s + n]


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ---------------------------------------------------------------- pools

def _challenge_root(cfg, base: Path) -> Path:
    return (base / cfg["challenge_dir"]).resolve()


def _challenge_rows(cfg, base: Path, split: str) -> list[dict]:
    lab = _challenge_root(cfg, base) / "dataset" / split / "labels.jsonl"
    if not lab.exists():
        return []
    return [json.loads(l) for l in lab.read_text().splitlines() if l.strip()]


def mad_rows(cfg, base: Path, split: str) -> list[dict]:
    """MAD csv rows for split ('training' or 'test') with mapped class; skipped labels removed."""
    root = _challenge_root(cfg, base) / "MAD_dataset"
    f = root / f"{split}.csv"
    if not f.exists():
        return []
    out = []
    with f.open() as fh:
        for r in csv.DictReader(fh):
            cls = cfg["mad"]["mapping"].get(str(r["label"]), "skip")
            if cls in CLASSES:
                out.append({"path": str(root / r["path"]), "rel": r["path"], "label": str(r["label"]),
                            "class": cls, "url": r.get("youtube url", "")})
    return out


def background_pools(cfg, base: Path) -> dict[str, list[str]]:
    """Real non-drone backgrounds, train splits only (no valid/test leakage).

    quiet: challenge ambient_outdoor clips (+ background_dirs) -> under ambient / vehicle / explosion
    any:   quiet + challenge vehicle/explosion + mapped MAD training clips -> under drones
    """
    root = _challenge_root(cfg, base)
    quiet, loud = [], []
    if cfg["mix"]["use_challenge_backgrounds"]:
        for r in _challenge_rows(cfg, base, "train"):
            if r["has_drone_audio"]:
                continue
            (quiet if r["sound_class"] == "ambient_outdoor" else loud).append(str(root / r["audio"]))
    for d in cfg["mix"]["background_dirs"]:
        p = Path(d) if Path(d).is_absolute() else (base / d)
        quiet += [str(f) for f in sorted(list(p.rglob("*.wav")) + list(p.rglob("*.flac")))]
    mad = [r["path"] for r in mad_rows(cfg, base, "training")] \
        if cfg["mad"]["enabled"] and cfg["mad"]["as_drone_background"] else []
    return {"quiet": quiet, "any": quiet + loud + mad}


# ---------------------------------------------------------------- one synthetic clip

def make_clip(cfg: dict, kind: str, seed: int, pools: dict[str, list[str]]):
    """Return (audio float32, params dict). kind in CLASSES."""
    rng = np.random.default_rng(seed)
    sr, n = cfg["sample_rate"], int(cfg["sample_rate"] * cfg["duration_s"])
    mix = cfg["mix"]
    params: dict = {"kind": kind, "seed": seed}

    bg_paths = pools["any"] if kind == "drone_buzz" else pools["quiet"]
    if bg_paths and rng.random() < mix["real_background_prob"]:
        bp = bg_paths[int(rng.integers(len(bg_paths)))]
        bg = _crop(rng, read_audio(Path(bp), sr), n)
        params["background"] = _short(bp)
    else:
        bg, _ = sources.ambient(rng, n, sr)
        params["background"] = "procedural"
    bg = bg / _rms(bg)

    if kind == "ambient_outdoor":
        if params["background"] != "procedural" and rng.random() < 0.5:
            extra, _ = sources.ambient(rng, n, sr)
            bg = 0.6 * bg + 0.4 * extra / _rms(extra)
        return _finish(rng, bg, mix), params

    n_src = n + int(PAD_S * sr)
    if kind == "drone_buzz":
        return _drone_clip(cfg, rng, bg, n, n_src, params)
    if kind == "vehicle_mechanical":
        src, sp = sources.vehicle(rng, n_src, sr, cfg["vehicle"])
    elif rng.random() < cfg["explosion"]["salvo_prob"]:
        src, sp = sources.salvo(rng, n, sr, cfg)
    else:
        src, sp = sources.explosion(rng, n, sr, cfg)
    params.update(sp)

    if kind == "explosion":
        sig = src
        snr = rng.uniform(*mix["explosion_snr_db"])
        sig = sig / _rms(sig[np.abs(sig) > 0.01 * np.abs(sig).max()])
    else:
        flight = copy.deepcopy(cfg["flight"])
        if kind == "vehicle_mechanical":  # ground vehicles stay on the ground
            flight["altitude_m"] = [0.5, 3]
            flight["speed_mps"] = [3, 20]
            flight["modes"] = {"flyby": 0.7, "approach": 0.2, "hover": 0.1}
        _, pos, fp = sample_trajectory(rng, cfg["duration_s"], flight)
        params.update(fp)
        sig = propagate(src, sr, n, pos, cfg["engine"])
        params["engine"] = cfg["engine"] if (cfg["engine"] != "acoular" or acoular_available()) else "numpy"
        snr = rng.uniform(*mix["snr_db_at_30m"]) - 20 * np.log10(max(fp["min_range_m"], 1) / 30.0)
        snr = float(np.clip(snr, *mix["snr_clip_db"]))
        sig = sig / _rms(sig)
    params["snr_db"] = round(float(snr), 1)
    y = bg + sig * 10 ** (snr / 20)
    return _finish(rng, y, mix), params


def _drone_clip(cfg, rng, bg, n, n_src, params):
    sr, dc, mix = cfg["sample_rate"], cfg["drone"], cfg["mix"]
    temp = float(rng.uniform(*dc["temperature_c"]))
    elev = float(rng.uniform(*dc["elevation_m"]))
    spec = drones.sample_spec(rng, dc, temp, elev)
    man_w = {k: float(v) for k, v in dc["manoeuvres"].items() if float(v) > 0} or {"steady": 1.0}
    names = list(man_w)
    pw = np.array([man_w[k] for k in names])
    manoeuvre = names[int(rng.choice(len(names), p=pw / pw.sum()))]
    if spec["type"] == "fixed_wing":
        manoeuvre = "cruise"
    flight = copy.deepcopy(cfg["flight"])
    if dc["flight_from_type"]:
        flight["speed_mps"] = spec["speed_range"]
        flight["altitude_m"] = spec["altitude_range"]
        if spec["type"] == "fixed_wing":
            flight["modes"] = {"flyby": 0.7, "approach": 0.3}
    vz = float(rng.uniform(1, 5)) if manoeuvre == "climb" else 0.0
    _, pos, fp = sample_trajectory(rng, cfg["duration_s"], flight, vz=vz)
    wind = float(rng.uniform(*dc["wind_mps"]))
    src, sp = drones.render(rng, n_src, sr, dc, spec, fp["speed_mps"], manoeuvre, wind)
    params.update(fp)
    params.update(sp)
    sig = propagate(src, sr, n, pos, cfg["engine"], c=spec["c"])
    params["engine"] = cfg["engine"] if (cfg["engine"] != "acoular" or acoular_available()) else "numpy"
    snr = rng.uniform(*mix["snr_db_at_30m"]) - 20 * np.log10(max(fp["min_range_m"], 1) / 30.0)
    if dc["level_from_physics"]:
        snr += sp["level_db"]
    snr = float(np.clip(snr, *mix["snr_clip_db"]))
    params["snr_db"] = round(snr, 1)
    y = bg + sig / _rms(sig) * 10 ** (snr / 20)
    return _finish(rng, y, mix), params


def _short(p: str) -> str:
    parts = Path(p).parts
    return "/".join(parts[-3:]) if "MAD_dataset" in parts else Path(p).name


def _finish(rng, y, mix):
    peak = 10 ** (rng.uniform(*mix["peak_dbfs"]) / 20)
    m = np.max(np.abs(y))
    return (y / m * peak if m > 0 else y).astype(np.float32)


# ---------------------------------------------------------------- one MAD clip

def make_mad_clip(cfg: dict, item: dict, seed: int, quiet: list[str]):
    """Cut window `item['window']` of a MAD recording to exactly duration_s."""
    rng = np.random.default_rng(seed)
    sr, n = cfg["sample_rate"], int(cfg["sample_rate"] * cfg["duration_s"])
    x = read_audio(Path(item["path"]), sr)
    params = {"kind": item["class"], "seed": seed, "source": "mad:" + item["rel"],
              "mad_label": item["label"], "mad_name": MAD_LABELS.get(item["label"], "?"),
              "youtube_url": item["url"]}
    if len(x) >= n:
        k, nw = item["window"], item["n_windows"]
        span = len(x) - n
        start = int(span * (k + rng.uniform(0.0, 1.0)) / nw) if nw > 1 else int(rng.integers(0, span + 1))
        start = min(max(start, 0), span)
        y = x[start : start + n]
        params["offset_s"] = round(start / sr, 2)
    else:  # short clip: place it over a quiet real background, ~20 dB down
        if quiet:
            bp = quiet[int(rng.integers(len(quiet)))]
            bg = _crop(rng, read_audio(Path(bp), sr), n)
            params["background"] = _short(bp)
        else:
            bg, _ = sources.ambient(rng, n, sr)
            params["background"] = "procedural"
        bg = bg / _rms(bg) * _rms(x) * 10 ** (-rng.uniform(15, 25) / 20)
        start = int(rng.integers(0, n - len(x) + 1))
        fade = min(160, len(x) // 4)
        win = np.ones(len(x))
        if fade:
            win[:fade] = np.linspace(0, 1, fade)
            win[-fade:] = np.linspace(1, 0, fade)
        y = bg.copy()
        y[start : start + len(x)] += x * win
        params["padded_from_s"] = round(len(x) / sr, 2)
        params["offset_s"] = round(start / sr, 2)
    return _finish(rng, y, cfg["mix"]), params


# ---------------------------------------------------------------- jobs

def _write(cfg, root, split, cid, y):
    sr, ext = cfg["sample_rate"], _ext(cfg)
    (root / "audio" / split / f"{cid}.{ext}").write_bytes(audio_bytes(y, sr, ext))
    spectrogram_image(y, sr).save(root / "spectrograms" / split / f"{cid}.png")
    return f"audio/{split}/{cid}.{ext}", f"spectrograms/{split}/{cid}.png"


def _job(args):
    cfg, job, pools, root = args
    if job["type"] == "mad":
        y, params = make_mad_clip(cfg, job["item"], job["seed"], pools["quiet"])
        kind, synthetic, source = job["item"]["class"], False, "mad"
    else:
        y, params = make_clip(cfg, job["kind"], job["seed"], pools)
        kind, synthetic, source = job["kind"], True, "synthetic"
    audio, png = _write(cfg, root, job["split"], job["id"], y)
    return {"id": job["id"], "split": job["split"], "audio": audio, "video": None, "spectrogram": png,
            "has_drone_audio": kind == "drone_buzz", "sound_class": kind,
            "synthetic": synthetic, "source": source, "params": params}


def _mad_jobs(cfg, base: Path, split_csv: str, out_split: str, seed0: int, cap: int) -> list[dict]:
    rows = mad_rows(cfg, base, split_csv)
    rng = np.random.default_rng(seed0)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    sr, n = cfg["sample_rate"], int(cfg["sample_rate"] * cfg["duration_s"])
    want = max(1, int(cfg["mad"]["windows_per_clip"]))
    jobs = []
    for label in sorted(by_label):
        items = by_label[label]
        for r in items:
            r["frames"] = sf.info(r["path"]).frames
        if cfg["mad"]["short_clips"] == "skip":  # filter before capping so the cap is met
            items = [r for r in items if r["frames"] >= n]
        if cap and len(items) > cap:
            items = [items[i] for i in sorted(rng.choice(len(items), cap, replace=False))]
        for r in items:
            frames = r["frames"]
            nw = 1 if frames < n else max(1, min(want, frames // n))
            stem = r["rel"].replace("/", "_").replace("training_", "").replace("test_", "").rsplit(".", 1)[0]
            for k in range(nw):
                jobs.append({"type": "mad", "split": out_split, "seed": seed0 * 7919 + len(jobs),
                             "id": f"mad{'' if out_split == 'train' else 'test'}_{stem}_w{k}",
                             "item": {**r, "window": k, "n_windows": nw}})
    return jobs


def plan(cfg, base: Path) -> tuple[list[dict], dict]:
    pools = background_pools(cfg, base)
    jobs: list[dict] = []
    i = 0
    for kind in CLASSES:
        for _ in range(int(cfg["counts"].get(kind, 0))):
            i += 1
            jobs.append({"type": "syn", "kind": kind, "split": "train",
                         "seed": cfg["seed"] * 1_000_003 + i, "id": f"syn_{i:06d}"})
    if cfg["mad"]["enabled"]:
        jobs += _mad_jobs(cfg, base, "training", "train", cfg["seed"] + 11, int(cfg["mad"]["max_per_label"]))
        if cfg["mad"]["test_split"]:
            jobs += _mad_jobs(cfg, base, "test", "mad_test", cfg["seed"] + 13, 0)
    return jobs, pools


def _copy_real(cfg, base: Path, root: Path, split: str) -> list[dict]:
    src_root = _challenge_root(cfg, base)
    rows = []
    for r in _challenge_rows(cfg, base, split):
        src_audio = src_root / r["audio"]
        if _ext(cfg) == "flac" and src_audio.suffix.lower() != ".flac":
            r["audio"] = str(Path(r["audio"]).with_suffix(".flac"))
            x, rate = sf.read(str(src_audio), dtype="float32")
            (root / r["audio"]).parent.mkdir(parents=True, exist_ok=True)
            (root / r["audio"]).write_bytes(audio_bytes(x, rate, "flac"))
        else:
            (root / r["audio"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_audio, root / r["audio"])
        dst = root / r["spectrogram"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / r["spectrogram"], dst)
        r["video"] = None
        r["synthetic"] = False
        r["source"] = "challenge"
        rows.append(r)
    return rows


LABEL_FIELDS = ("id", "split", "audio", "video", "spectrogram", "has_drone_audio", "sound_class")


def write_labels(split_dir: Path, rows: list[dict]) -> None:
    """labels.jsonl (challenge schema), labels.csv (same columns), metadata.jsonl (everything)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "labels.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in LABEL_FIELDS}) + "\n")
    with (split_dir / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(LABEL_FIELDS)
        for r in rows:
            w.writerow(["" if r.get(k) is None else str(r[k]).lower() if isinstance(r[k], bool) else r[k]
                        for k in LABEL_FIELDS])
    with (split_dir / "metadata.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "source": r.get("source"), "synthetic": r.get("synthetic"),
                                 "params": r.get("params", {})}) + "\n")


def estimate(cfg: dict, base: Path) -> dict:
    """Clip counts per split/class/source without generating anything (for the UI)."""
    jobs, pools = plan(cfg, base)
    out: dict = {"total": len(jobs), "by_split": {}, "backgrounds": {k: len(v) for k, v in pools.items()}}
    for j in jobs:
        cls = j["item"]["class"] if j["type"] == "mad" else j["kind"]
        s = out["by_split"].setdefault(j["split"], {})
        key = f"{cls}"
        s.setdefault(key, {"mad": 0, "synthetic": 0})
        s[key]["mad" if j["type"] == "mad" else "synthetic"] += 1
    out["real_train"] = len(_challenge_rows(cfg, base, "train")) if cfg["include_real"] else 0
    out["valid"] = len(_challenge_rows(cfg, base, "valid"))
    return out


def build(cfg: dict, out_dir: Path, base: Path, progress=None) -> dict:
    """Generate the dataset. progress(done, total) is called as clips finish."""
    t0 = time.time()
    name = "".join(ch for ch in cfg["name"] if ch.isalnum() or ch in "-_.") or "dataset"
    root = out_dir / name
    if root.exists():
        shutil.rmtree(root)
    jobs, pools = plan(cfg, base)
    splits = sorted({j["split"] for j in jobs} | {"train", "valid"})
    for s in splits:
        for sub in ("audio", "spectrograms", "dataset"):
            (root / sub / s).mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    args = [(cfg, j, pools, root) for j in jobs]
    rows: list[dict] = []
    if progress:
        progress(0, total)
    workers = max(1, int(cfg.get("workers", 1)))
    if workers == 1 or total < 8:
        for a in args:
            rows.append(_job(a))
            if progress:
                progress(len(rows), total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job, a) for a in args]
            for f in as_completed(futs):
                rows.append(f.result())
                if progress:
                    progress(len(rows), total)
    rows.sort(key=lambda r: (r["split"], r["source"], r["id"]))

    real_train = _copy_real(cfg, base, root, "train") if cfg["include_real"] else []
    real_valid = _copy_real(cfg, base, root, "valid")
    by_split: dict[str, list[dict]] = {s: [] for s in splits}
    by_split["train"] += real_train
    by_split["valid"] += real_valid
    for r in rows:
        by_split[r["split"]].append(r)
    for s, rs in by_split.items():
        write_labels(root / "dataset" / s, rs)

    def tally(rs):
        t = {c: 0 for c in CLASSES}
        for r in rs:
            t[r["sound_class"]] += 1
        return t

    manifest = {
        "name": name, "comment": str(cfg.get("comment", "")).strip(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "synthetic_clips": sum(r["synthetic"] for r in rows),
        "mad_clips": sum(r["source"] == "mad" for r in rows),
        "real_train_clips": len(real_train), "valid_clips": len(real_valid),
        "counts": tally(by_split["train"]),
        "splits": {s: {"total": len(rs), "classes": tally(rs)} for s, rs in by_split.items()},
        "backgrounds_available": {k: len(v) for k, v in pools.items()},
        "audio_format": _ext(cfg), "config": cfg,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_readme(root, manifest)
    return manifest


def _write_readme(root: Path, m: dict) -> None:
    """Human-readable summary next to the data, so the comment travels with the zip."""
    lines = [f"# {m['name']}", ""]
    if m["comment"]:
        lines += [m["comment"], ""]
    lines += [f"Created {m['created']} in {m['seconds']} s. Audio: {m['audio_format'].upper()}, 16 kHz mono, 5 s.",
              f"Sources: {m['mad_clips']} MAD, {m['synthetic_clips']} synthetic, {m['real_train_clips']} challenge train.",
              "", "| split | total | drone_buzz | vehicle_mechanical | explosion | ambient_outdoor |",
              "|---|---|---|---|---|---|"]
    for s, v in m["splits"].items():
        c = v["classes"]
        lines.append(f"| {s} | {v['total']} | {c['drone_buzz']} | {c['vehicle_mechanical']} | {c['explosion']} | {c['ambient_outdoor']} |")
    lines += ["", "Labels: dataset/<split>/labels.jsonl (challenge schema), labels.csv, metadata.jsonl.",
              "Full generator settings: manifest.json -> config.", ""]
    (root / "README.md").write_text("\n".join(lines))


def preview(cfg: dict, kind: str, base: Path, seed: int | None = None, source: str = "synthetic"):
    """One clip for the web preview: (audio bytes in cfg's format, png bytes, params).

    source="mad" draws a random MAD training clip mapped to `kind` instead of synthesizing one.
    """
    seed = int(seed if seed is not None else np.random.default_rng().integers(1 << 31))
    pools = background_pools(cfg, base)
    if source == "mad":
        rows = [r for r in mad_rows(cfg, base, "training") if r["class"] == kind]
        if not rows:
            raise ValueError(f"No MAD labels are mapped to {kind}. Change the mapping to preview it.")
        r = rows[int(np.random.default_rng(seed).integers(len(rows)))]
        y, params = make_mad_clip(cfg, {**r, "window": 0, "n_windows": 1}, seed, pools["quiet"])
    else:
        y, params = make_clip(cfg, kind, seed, pools)
    sr = cfg["sample_rate"]
    buf = io.BytesIO()
    spectrogram_image(y, sr).save(buf, format="PNG")
    return audio_bytes(y, sr, _ext(cfg)), buf.getvalue(), params
