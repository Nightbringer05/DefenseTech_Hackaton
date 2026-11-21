"""Generate a shortcut-free synthetic training set for Challenge 5, in the exact challenge format.

    python generate_synthetic_dataset.py --download                      # MAD + ESC-50 backgrounds, synthetic drones
    python generate_synthetic_dataset.py --download --real-drone-dir D   # + real drone recordings from folder D

Output (default WorkSpace/datasets/synth_c5/) has the same layout as InputData/, so the
starter kit trains on it directly (--challenge-dir points to it):
    dataset/{train,valid}/labels.jsonl   spectrograms/{train,valid}/*.png   audit.json   preview.png

How a clip is made
  1. Background: a real 5 s window from MAD / ESC-50 / challenge-train ambient / your own folders,
     sorted into categories: ambient, speech, vehicle, tonal_neg (helicopter, jets, saws, sirens), explosion.
  2. Negatives: N per class (ambient_outdoor, vehicle_mechanical, explosion).
  3. Positives (drone_buzz): N clips, made by adding a drone to a *copy* of N/3 negatives from each
     class. Every background category therefore has the same P(drone) = 25 %, and the drone is
     the only thing that separates a drone clip from its paired negative (no background shortcut).
  4. Drone = synthetic rotor model (2-6 detuned rotors, 80-400 Hz blade-pass, harmonics, small RPM drift or manoeuvres,
     rotor whoosh, optional Doppler fly-by, partial presence, distance low-pass) or a real recording.
  5. Distractor tones (pure sine, mains hum, low engine, helicopter thump, siren, whine) are added
     with the same probability to every clip, drone or not: "a line in the spectrogram" is not drone.
  6. Channel matched to the challenge: low-pass at challenge-train cutoffs, loudness from challenge
     train, then the exact challenge log-mel PNG recipe (c5_audio.spectrogram_image).

Train/valid are split by source recording before mixing. Challenge valid/test are never read.
Needs only numpy + Pillow (ffmpeg optional, used to decode non-WAV real-drone files).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_audio as A  # noqa: E402
from build_mad_pretrain import c5_train_channel_stats  # noqa: E402

SR, N = A.SR, A.CLIP_SAMPLES
CLASSES = ["drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor"]
NEG_CLASSES = ["vehicle_mechanical", "explosion", "ambient_outdoor"]
CATEGORY_TO_CLASS = {"ambient": "ambient_outdoor", "speech": "ambient_outdoor", "vehicle": "vehicle_mechanical",
                     "tonal_neg": "vehicle_mechanical", "explosion": "explosion"}
CLASS_MIX = {"ambient_outdoor": {"ambient": 0.95, "speech": 0.05},
             "vehicle_mechanical": {"vehicle": 0.6, "tonal_neg": 0.4},
             "explosion": {"explosion": 1.0}}
# MAD labels: 0 communication, 1 gunshot, 2 footsteps, 3 shelling, 4 vehicle, 5 helicopter, 6 fighter
MAD_CATEGORY = {0: "speech", 3: "explosion", 4: "vehicle", 5: "tonal_neg", 6: "tonal_neg"}
ESC50_CATEGORY = {
    **{c: "ambient" for c in ["wind", "rain", "sea_waves", "thunderstorm", "crickets", "insects", "chirping_birds",
                              "crow", "frog", "rooster", "hen", "cow", "sheep", "dog", "crackling_fire",
                              "water_drops", "church_bells"]},
    **{c: "vehicle" for c in ["engine", "train", "car_horn"]},
    **{c: "tonal_neg" for c in ["helicopter", "airplane", "chainsaw", "hand_saw", "vacuum_cleaner", "siren"]},
    "fireworks": "explosion",
}
MAD_URL = "https://www.kaggle.com/api/v1/datasets/download/junewookim/mad-dataset-military-audio-dataset"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ============================================================================ audio loading
def resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x
    n_out = int(round(len(x) * SR / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, complex)
    m = min(len(Y), len(X))
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)


def load_audio(path: str) -> np.ndarray:
    """Any WAV (8/16/24/32-bit PCM, any rate/channels) natively; other formats through ffmpeg."""
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as w:
                ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
            if sw == 3:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
                x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(np.float32) / (1 << 23)
            else:
                dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
                x = np.frombuffer(raw, dt).astype(np.float32)
                x = (x - 128) / 128 if sw == 1 else x / float(2 ** (8 * sw - 1))
            if ch > 1:
                x = x.reshape(-1, ch).mean(1)
            return resample(x, sr)
        except (wave.Error, EOFError, KeyError):
            pass  # e.g. float WAV -> ffmpeg
    if shutil.which("ffmpeg"):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    raise RuntimeError(f"cannot decode {path} (install ffmpeg for non-PCM-WAV files)")


def window_quality_ok(w: np.ndarray, check_flatness: bool = True) -> str | None:
    if np.mean(np.abs(w) > 0.999) > 0.001:
        return "clipped"
    frames = w[: len(w) // 1024 * 1024].reshape(-1, 1024)
    if np.mean(np.sqrt(np.mean(frames ** 2, 1)) < 1e-4) > 0.2:
        return "silent"
    if check_flatness and A.spectral_flatness(w) < 1e-4:
        return "flatness"
    return None


def tile_to_5s(x: np.ndarray, rng: np.random.Generator, xfade: int = 800) -> np.ndarray:
    """Loop a short clip to 5 s with crossfades and a random start (for short real-drone files)."""
    x = np.roll(x, rng.integers(len(x)))
    out = x.copy()
    ramp = np.linspace(0, 1, xfade, dtype=np.float32)
    while len(out) < N + xfade:
        out = np.concatenate([out[:-xfade], out[-xfade:] * (1 - ramp) + x[:xfade] * ramp, x[xfade:]])
    return out[:N]


# ============================================================================ synthesis
def smooth_noise(rng, rate_hz: float) -> np.ndarray:
    """Smooth random curve in [-1, 1] with about `rate_hz` changes per second."""
    k = max(3, int(5 * rate_hz) + 2)
    pts = rng.uniform(-1, 1, k)
    xs = np.linspace(0, N - 1, k)
    t = np.arange(N)
    i = np.clip(np.searchsorted(xs, t) - 1, 0, k - 2)
    u = (t - xs[i]) / (xs[i + 1] - xs[i])
    u = (1 - np.cos(np.pi * u)) / 2
    return (pts[i] * (1 - u) + pts[i + 1] * u).astype(np.float32)


def bandpass(x, lo, hi, width=150.0):
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.clip((f - lo) / width + 1, 0, 1) * np.clip((hi - f) / width + 1, 0, 1)
    return np.fft.irfft(np.fft.rfft(x) * g, n=len(x)).astype(np.float32)


def harmonic_stack(rng, finst, amps):
    phase = 2 * np.pi * np.cumsum(finst, dtype=np.float64) / SR
    k = np.arange(1, len(amps) + 1)[:, None]
    return (amps[:, None] * np.sin(k * phase[None, :] + rng.uniform(0, 2 * np.pi, (len(amps), 1)))).sum(0)


def synth_drone(rng) -> tuple[np.ndarray, dict]:
    t = np.arange(N) / SR
    n_rot = int(rng.choice([2, 3, 4, 4, 4, 6]))
    f0 = float(np.exp(rng.uniform(np.log(80), np.log(400))))
    manoeuvre = bool(rng.random() < 0.15)
    drift = float(rng.uniform(0.08, 0.2) if manoeuvre else rng.uniform(0.01, 0.05))
    throttle = 1 + drift * smooth_noise(rng, rng.uniform(0.1, 0.8))
    doppler, dist_env, flyby = np.ones(N, np.float32), np.ones(N, np.float32), bool(rng.random() < 0.3)
    if flyby:
        v, d, t0 = rng.uniform(5, 25), rng.uniform(10, 80), rng.uniform(0.5, 4.5)
        r = np.sqrt((v * (t - t0)) ** 2 + d ** 2)
        doppler = (343.0 / (343.0 + v * v * (t - t0) / r)).astype(np.float32)
        dist_env = (d / r).astype(np.float32)
    n_h = int(rng.integers(8, 31))
    alpha = float(rng.uniform(0.8, 2.0))
    shape_db = 6 * np.interp(np.arange(n_h), np.linspace(0, n_h - 1, 5), rng.uniform(-1, 1, 5))
    sig = np.zeros(N, np.float32)
    for _ in range(n_rot):
        finst = f0 * (1 + rng.uniform(-0.04, 0.04)) * throttle * doppler
        finst = finst * (1 + rng.uniform(0.002, 0.01) * smooth_noise(rng, rng.uniform(2, 8)))
        kmax = max(1, min(n_h, int(7600 / finst.max())))
        k = np.arange(1, kmax + 1)
        amps = k ** (-alpha) * 10 ** (shape_db[:kmax] / 20)
        rotor = harmonic_stack(rng, finst, amps)
        rotor *= 1 + rng.uniform(0.02, 0.1) * smooth_noise(rng, rng.uniform(1, 6))
        sig += rng.uniform(0.6, 1.0) * rotor
    # broadband rotor "whoosh", amplitude-modulated at the blade-pass rate
    bpf_phase = 2 * np.pi * np.cumsum(f0 * throttle * doppler) / SR
    whoosh = bandpass(rng.standard_normal(N).astype(np.float32), rng.uniform(200, 800), rng.uniform(2000, 6000))
    whoosh *= 1 + 0.5 * np.sin(bpf_phase)
    w_db = float(rng.uniform(-15, 0))
    whoosh *= np.sqrt(np.mean(sig ** 2) / (np.mean(whoosh ** 2) + 1e-12)) * 10 ** (w_db / 20)
    sig = (sig + whoosh) * dist_env
    return sig.astype(np.float32), {"f0": round(f0, 1), "rotors": n_rot, "harmonics": n_h,
                                    "drift": round(drift, 3), "manoeuvre": manoeuvre, "flyby": flyby, "whoosh_db": round(w_db, 1)}


def synth_distractor(rng) -> tuple[np.ndarray, str]:
    """Tonal sounds that are NOT drones (labels stay unchanged)."""
    t = np.arange(N) / SR
    kind = str(rng.choice(["sine", "hum", "engine", "heli", "siren", "whine"]))
    if kind == "sine":
        f = rng.uniform(100, 3000) * (1 + 0.003 * smooth_noise(rng, 1))
        x = harmonic_stack(rng, f, np.array([1.0, rng.uniform(0, 0.3)]))
    elif kind == "hum":
        k = np.arange(1, 21)
        amps = np.where(k % 2 == 1, 1.0, 0.3) * k ** -0.8
        x = harmonic_stack(rng, np.full(N, float(rng.choice([50.0, 60.0]))), amps)
    elif kind == "engine":
        f = rng.uniform(15, 50) * (1 + 0.05 * smooth_noise(rng, 0.5))
        kmax = int(1500 / f.max())
        x = harmonic_stack(rng, f, np.arange(1, kmax + 1) ** -0.7)
        x *= 1 + 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    elif kind == "heli":
        bp = rng.uniform(8, 25)
        ph = (t * bp) % 1.0
        x = np.exp(-ph * rng.uniform(8, 20)) * rng.standard_normal(N)
        x = bandpass(x.astype(np.float32), 30, rng.uniform(600, 1500))
    elif kind == "siren":
        fc, depth, rate = rng.uniform(600, 1200), rng.uniform(0.15, 0.35), rng.uniform(0.2, 1.5)
        x = harmonic_stack(rng, fc * (1 + depth * np.sin(2 * np.pi * rate * t)), np.array([1.0, 0.4, 0.2]))
    else:  # whine
        x = harmonic_stack(rng, rng.uniform(2000, 6000) * (1 + 0.01 * smooth_noise(rng, 0.5)), np.array([1.0]))
    return x.astype(np.float32), kind


def presence_envelope(rng) -> tuple[np.ndarray, str]:
    if rng.random() >= CFG["p_partial"]:
        return np.ones(N, np.float32), "full"
    dur = int(rng.uniform(2.5, 4.5) * SR)
    start = int(rng.integers(0, N - dur + 1))
    fade = int(rng.uniform(0.3, 1.0) * SR)
    env = np.zeros(N, np.float32)
    env[start:start + dur] = 1
    r = np.linspace(0, 1, fade, dtype=np.float32)
    env[start:start + fade] = r[: min(fade, dur)]
    env[start + dur - fade:start + dur] = np.minimum(env[start + dur - fade:start + dur], r[::-1])
    return env, f"partial_{start / SR:.1f}-{(start + dur) / SR:.1f}s"


# ============================================================================ worker side
CFG: dict = {}


def _init(cfg):
    CFG.update(cfg)


def load_window(w: dict, rng=None) -> np.ndarray:
    x = load_audio(w["path"])
    if w["start"] < 0:  # short file -> tile
        return tile_to_5s(x, rng or np.random.default_rng(0))
    return x[w["start"]:w["start"] + N].astype(np.float32)


def index_file(task: dict) -> tuple[list[dict], Counter]:
    stats = Counter()
    try:
        x = load_audio(task["path"])
    except Exception:
        stats["decode_error"] += 1
        return [], stats
    out = []
    if len(x) >= N:
        n = min(len(x) // N, task["max_windows"])
        s0 = (len(x) - n * N) // 2
        starts = [s0 + i * N for i in range(n)]
    elif task["kind"] == "drone" and len(x) >= SR // 2:
        starts = [-1]
    else:
        stats["shorter_than_5s"] += 1
        return [], stats
    for s in starts:
        w = tile_to_5s(x, np.random.default_rng(0)) if s < 0 else x[s:s + N]
        bad = window_quality_ok(w, check_flatness=task["kind"] == "bg" and task["source"] != "c5train")
        if bad:
            stats[bad] += 1
            continue
        out.append({k: task[k] for k in ("path", "category", "source", "group", "kind")} | {"start": int(s)})
    return out, stats


def render(job: dict) -> dict:
    rng = np.random.default_rng(job["seed"])
    mix = load_window(job["bg"])
    row = {"id": job["id"], "split": job["split"], "spectrogram": f"spectrograms/{job['split']}/{job['id']}.png",
           "has_drone_audio": job["drone"] is not None, "sound_class": job["cls"],
           "bg_category": job["bg"]["category"], "bg_source": job["bg"]["source"], "bg_group": job["bg"]["group"],
           "bg_path": job["bg"]["path"], "bg_start": job["bg"]["start"], "pair_of": job.get("pair_of")}
    if job["distractor"]:
        d, kind = synth_distractor(rng)
        lvl = float(rng.uniform(*CFG["snr_db"]))  # same level range as drones: loudness is not a cue
        mix = A.mix_background(mix, d, -lvl)
        row["distractor"] = {"kind": kind, "distractor_to_bg_db": round(lvl, 1)}
    else:
        row["distractor"] = None
    if job["drone"] is not None:
        if job["drone"]["source"] == "synth":
            dr, params = synth_drone(rng)
        else:
            dr, params = load_window(job["drone"]["window"], rng), {"path": job["drone"]["window"]["path"]}
        env, presence = presence_envelope(rng)
        dr = dr * env
        snr = float(rng.uniform(*CFG["snr_db"]))
        if snr < 0 and rng.random() < 0.5:
            fc = float(rng.uniform(3000, 6000))
            dr = A.lowpass(dr, fc)
            params["distance_lowpass_hz"] = int(fc)
        on = env > 0.5
        pd = np.mean(dr[on] ** 2) + 1e-12
        pb = np.mean(mix ** 2) + 1e-12
        mix = mix + dr * np.sqrt(pb * 10 ** (snr / 10) / pd)
        row["drone"] = {"source": job["drone"]["source"], "snr_db": round(snr, 1), "presence": presence, **params}
    else:
        row["drone"] = None
    aug = []
    if rng.random() < CFG["p_lowpass"]:
        fc = float(rng.choice(CFG["cutoffs"]))
        mix = A.lowpass(mix, fc)
        aug.append(f"lp{int(fc)}")
    if rng.random() < 0.3:
        tilt = float(rng.uniform(-2, 2))
        mix = A.spectral_tilt(mix, tilt)
        aug.append(f"tilt{tilt:+.1f}")
    mix = A.match_rms(mix, float(rng.choice(CFG["rms_pool"])))
    row["channel"] = aug
    out = Path(CFG["out_dir"])
    (out / row["spectrogram"]).parent.mkdir(parents=True, exist_ok=True)
    A.spectrogram_image(mix).save(out / row["spectrogram"])
    if CFG["save_audio"]:
        row["audio"] = f"audio/{job['split']}/{job['id']}.wav"
        A.write_wav(out / row["audio"], mix)
    return row


# ============================================================================ main side
def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    print(f"  downloading {url}")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        total, done, last = int(r.headers.get("Content-Length") or 0), 0, time.time()
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                print(f"    {done / 1e6:,.0f} MB" + (f" / {total / 1e6:,.0f} MB" if total else ""))
                last = time.time()
    tmp.replace(dest)


def ensure_sources(args) -> None:
    if not args.download:
        return
    if not (args.mad_dir / "training.csv").exists():
        z = args.sources_dir / "mad.zip"
        if not z.exists():
            download(MAD_URL, z)
        print("  extracting MAD")
        zipfile.ZipFile(z).extractall(args.mad_dir.parent)
    if not (args.esc50_dir / "meta" / "esc50.csv").exists():
        z = args.sources_dir / "esc50.zip"
        if not z.exists():
            download(ESC50_URL, z)
        print("  extracting ESC-50")
        zipfile.ZipFile(z).extractall(args.esc50_dir.parent)


def collect_tasks(args) -> list[dict]:
    tasks = []
    mw = args.max_windows_per_file

    def add(path, category, source, group, kind="bg"):
        tasks.append(dict(path=str(path), category=category, source=source, group=f"{source}:{group}",
                          kind=kind, max_windows=mw))

    # challenge train ambient (train split only, never valid/test)
    for l in (args.c5_dir / "dataset/train/labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["split"] == "train" and r["sound_class"] == "ambient_outdoor":
            add(args.c5_dir / r["audio"], "ambient", "c5train", r["id"])
    if (args.mad_dir / "training.csv").exists():
        for name in ("training.csv", "test.csv"):
            with open(args.mad_dir / name, encoding="utf-8", errors="replace", newline="") as fh:
                for r in csv.DictReader(fh):
                    cat = MAD_CATEGORY.get(int(r["label"]))
                    if cat:
                        vid = r["youtube url"].split("v=")[-1].split("&")[0]
                        add(args.mad_dir / r["path"], cat, "mad", vid)
    else:
        print(f"  ! MAD not found at {args.mad_dir} (use --download)")
    if (args.esc50_dir / "meta" / "esc50.csv").exists():
        with open(args.esc50_dir / "meta" / "esc50.csv", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                cat = ESC50_CATEGORY.get(r["category"])
                if cat:
                    add(args.esc50_dir / "audio" / r["filename"], cat, "esc50", r["src_file"])
    else:
        print(f"  ! ESC-50 not found at {args.esc50_dir} (use --download)")
    for spec in args.extra_bg:
        d, cat = spec.rsplit("=", 1)
        if cat not in CATEGORY_TO_CLASS:
            sys.exit(f"--extra-bg category must be one of {sorted(CATEGORY_TO_CLASS)}")
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                add(p, cat, f"extra_{Path(d).name}", p.relative_to(d).parent / p.stem)
    for d in args.real_drone_dir:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                g = p.relative_to(d).parent if args.real_drone_group == "dir" else p.relative_to(d).with_suffix("")
                add(p, "drone", f"drone_{Path(d).name}", g, kind="drone")
    return tasks


def split_of(group: str, args) -> str:
    if group.startswith("c5train:"):
        return "train"
    h = int(hashlib.sha1(f"{args.seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if h < args.valid_frac else "train"


def cap_per_group(windows: list[dict], cap: int, rng) -> list[dict]:
    by = defaultdict(list)
    for w in windows:
        by[(w["group"], w["category"])].append(w)
    out = []
    for ws in by.values():
        idx = rng.permutation(len(ws))[:cap]
        out += [ws[i] for i in idx]
    return out


def plan_split(split, bg, drones, n, rng, args) -> list[dict]:
    by_cat = defaultdict(list)
    for w in bg:
        by_cat[w["category"]].append(w)
    jobs, negs_by_class = [], {}
    for cls in NEG_CLASSES:
        mix = {c: p for c, p in CLASS_MIX[cls].items() if by_cat.get(c)}
        if not mix:
            sys.exit(f"[{split}] no background audio for class {cls}: add sources (--download / --extra-bg)")
        cats = list(mix)
        counts = rng.multinomial(n, np.array([mix[c] for c in cats]) / sum(mix.values()))
        chosen = []
        for c, k in zip(cats, counts):
            pool = by_cat[c]
            idx = rng.choice(len(pool), k, replace=k > len(pool))
            chosen += [pool[i] for i in idx]
        rng.shuffle(chosen)
        negs_by_class[cls] = chosen
    seq = 0
    for cls, ws in negs_by_class.items():
        for w in ws:
            jobs.append(dict(id=f"syn_{split}_{seq:06d}", split=split, cls=cls, bg=w, drone=None))
            seq += 1
    # drone positives: copies of n/3 negatives from each class -> equal P(drone) per background category
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    neg_jobs = defaultdict(list)
    for j in jobs:
        neg_jobs[j["cls"]].append(j)
    picks = []
    for cls, k in zip(NEG_CLASSES, per):
        by_cat = defaultdict(list)
        for j in neg_jobs[cls]:
            by_cat[j["bg"]["category"]].append(j)
        for c, q in largest_remainder({c: len(v) for c, v in by_cat.items()}, k).items():
            picks += [(cls, by_cat[c][i]) for i in rng.choice(len(by_cat[c]), q, replace=False)]
    for cls, src in picks:
        use_real = bool(drones) and rng.random() < args.p_real_drone
        drone = ({"source": "real", "window": drones[rng.integers(len(drones))]} if use_real
                 else {"source": "synth"})
        jobs.append(dict(id=f"syn_{split}_{seq:06d}", split=split, cls="drone_buzz", bg=src["bg"],
                         drone=drone, pair_of=src["id"]))
        seq += 1
    # distractors: same share inside every (class, drone) cell -> carry no label information
    cells = defaultdict(list)
    for j in jobs:
        cells[(j["cls"], j["drone"] is None)].append(j)
    for cell in cells.values():
        on = set(rng.choice(len(cell), int(round(args.p_distractor * len(cell))), replace=False).tolist())
        for i, j in enumerate(cell):
            j["distractor"] = i in on
    for j in jobs:
        j["seed"] = int(rng.integers(1 << 62))
    return jobs


def largest_remainder(sizes: dict, k: int) -> dict:
    """Split k proportionally to sizes, exactly (Hamilton apportionment)."""
    total = sum(sizes.values())
    exact = {c: k * v / total for c, v in sizes.items()}
    out = {c: int(e) for c, e in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - out[c], reverse=True)[: k - sum(out.values())]:
        out[c] += 1
    return out


def audit(rows: list[dict]) -> dict:
    rep = {}
    for split in ("train", "valid"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        cats = sorted({r["bg_category"] for r in rs})
        table = {}
        for c in cats:
            sub = [r for r in rs if r["bg_category"] == c]
            pos = sum(r["has_drone_audio"] for r in sub)
            table[c] = {"drone": pos, "no_drone": len(sub) - pos, "p_drone": round(pos / len(sub), 3)}
        dis = [r for r in rs if r["distractor"]]
        rep[split] = {
            "clips": len(rs), "classes": dict(Counter(r["sound_class"] for r in rs)),
            "by_background_category": table,
            "p_drone_with_distractor": round(sum(r["has_drone_audio"] for r in dis) / max(1, len(dis)), 3),
            "p_drone_without_distractor": round(sum(r["has_drone_audio"] for r in rs if not r["distractor"])
                                                / max(1, len(rs) - len(dis)), 3),
            "background_sources": dict(Counter(r["bg_source"] for r in rs)),
            "drone_sources": dict(Counter(r["drone"]["source"] for r in rs if r["drone"])),
            "unique_background_windows": len({(r["bg_path"], r["bg_start"]) for r in rs if not r["has_drone_audio"]}),
        }
        p_all = sum(r["has_drone_audio"] for r in rs) / len(rs)
        bad = [c for c, v in table.items()
               if abs(v["drone"] - p_all * (v["drone"] + v["no_drone"])) > max(1.5, 0.03 * (v["drone"] + v["no_drone"]))]
        rep[split]["shortcut_check"] = "OK" if not bad else f"WARNING: P(drone) differs for {bad}"
    return rep


def preview(rows, out: Path, c5_dir: Path) -> None:
    tr = [r for r in rows if r["split"] == "train"]
    groups = [("synthetic drone", [r for r in tr if r["drone"] and r["drone"]["source"] == "synth"][:6]),
              ("real drone", [r for r in tr if r["drone"] and r["drone"]["source"] == "real"][:6]),
              ("negative + distractor", [r for r in tr if not r["drone"] and r["distractor"]][:6]),
              ("negative", [r for r in tr if not r["drone"] and not r["distractor"]][:6])]
    c5 = [json.loads(l) for l in (c5_dir / "dataset/train/labels.jsonl").read_text(encoding="utf-8").splitlines()]
    c5_dr = [c5_dir / r["spectrogram"] for r in c5 if r["has_drone_audio"]][:6]
    lines = [("CHALLENGE real drone", [Image.open(p) for p in c5_dr])]
    lines += [(name, [Image.open(out / r["spectrogram"]) for r in rs]) for name, rs in groups if rs]
    cell = 160
    canvas = Image.new("RGB", (150 + 6 * (cell + 6), len(lines) * (cell + 6) + 6), "white")
    d = ImageDraw.Draw(canvas)
    for i, (name, ims) in enumerate(lines):
        y = 6 + i * (cell + 6)
        d.text((6, y + cell // 2 - 6), name, fill="black")
        for j, im in enumerate(ims):
            canvas.paste(im.convert("RGB").resize((cell, cell)), (150 + j * (cell + 6), y))
    canvas.save(out / "preview.png")


def parse_args() -> argparse.Namespace:
    ws = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="download MAD + ESC-50 if missing (~1.7 GB)")
    p.add_argument("--c5-dir", type=Path, default=ws.parent / "InputData", help="only train/ is read")
    p.add_argument("--mad-dir", type=Path, default=ws / "mad-dataset" / "MAD_dataset")
    p.add_argument("--esc50-dir", type=Path, default=ws / "sources" / "ESC-50-master")
    p.add_argument("--sources-dir", type=Path, default=ws / "sources", help="where downloads are stored")
    p.add_argument("--real-drone-dir", type=Path, action="append", default=[],
                   help="folder of real drone recordings (any length/format); repeatable")
    p.add_argument("--real-drone-group", choices=("file", "dir"), default="file",
                   help="train/valid grouping unit for real drones (use 'dir' if one folder = one recording)")
    p.add_argument("--extra-bg", action="append", default=[], metavar="DIR=CATEGORY",
                   help=f"extra background folder; CATEGORY in {sorted(CATEGORY_TO_CLASS)}; repeatable")
    p.add_argument("--out-dir", type=Path, default=ws / "datasets" / "synth_c5")
    p.add_argument("--n-per-class", type=int, default=1500, help="train clips per class (4 classes)")
    p.add_argument("--n-valid-per-class", type=int, default=250)
    p.add_argument("--valid-frac", type=float, default=0.15, help="fraction of source recordings held out")
    p.add_argument("--p-real-drone", type=float, default=0.5, help="share of positives using real drone audio")
    p.add_argument("--snr-db", type=float, nargs=2, default=(-10.0, 10.0),
                   help="drone-to-background SNR range (also used for distractor tones)")
    p.add_argument("--p-partial", type=float, default=0.3, help="drone present for only 2.5-4.5 s")
    p.add_argument("--p-distractor", type=float, default=0.25, help="non-drone tone added (any clip)")
    p.add_argument("--p-lowpass", type=float, default=0.25, help="challenge-style band-limiting")
    p.add_argument("--max-windows-per-file", type=int, default=2)
    p.add_argument("--max-per-group", type=int, default=8, help="windows per source recording and category")
    p.add_argument("--save-audio", action="store_true", help="also write the 5 s WAVs")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("[1/5] sources")
    ensure_sources(args)
    tasks = collect_tasks(args)
    print(f"      {len(tasks)} files: {dict(Counter((t['source'].split('_')[0], t['category']) for t in tasks))}")

    print("[2/5] indexing + quality screening")
    windows, stats = [], Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for ws, st in ex.map(index_file, tasks, chunksize=32):
            windows += ws
            stats += st
    print(f"      {len(windows)} usable 5 s windows; rejected: {dict(stats)}")
    windows = cap_per_group(windows, args.max_per_group, rng)
    bg = {"train": [], "valid": []}
    dr = {"train": [], "valid": []}
    for w in windows:
        (dr if w["kind"] == "drone" else bg)[split_of(w["group"], args)].append(w)
    for s in ("train", "valid"):
        print(f"      {s}: {len(bg[s])} background windows {dict(Counter(w['category'] for w in bg[s]))}, "
              f"{len(dr[s])} real-drone windows")
    if args.real_drone_dir and not dr["valid"]:
        print("      ! no real-drone recordings landed in valid (too few groups); valid uses synthetic drones only")

    print("[3/5] channel statistics from challenge train/")
    _, rms_pool, cutoffs, n_c5 = c5_train_channel_stats(args.c5_dir)
    print(f"      {n_c5} clips, low-pass cutoffs {sorted(int(c) for c in cutoffs)}")

    print("[4/5] planning + rendering")
    jobs = (plan_split("train", bg["train"], dr["train"], args.n_per_class, rng, args)
            + plan_split("valid", bg["valid"], dr["valid"], args.n_valid_per_class, rng, args))
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    cfg = dict(out_dir=str(args.out_dir), snr_db=tuple(args.snr_db), p_partial=args.p_partial,
               p_lowpass=args.p_lowpass, cutoffs=cutoffs, rms_pool=rms_pool, save_audio=args.save_audio)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg,)) as ex:
        for i, row in enumerate(ex.map(render, jobs, chunksize=16), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(jobs):
                print(f"      {i}/{len(jobs)} clips ({time.time() - t0:.0f} s)")

    print("[5/5] writing labels, audit, preview")
    for split in ("train", "valid"):
        p = args.out_dir / "dataset" / split / "labels.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == split), encoding="utf-8")
    rep = audit(rows)
    rep["args"] = {k: str(v) for k, v in vars(args).items()}
    (args.out_dir / "audit.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    preview(rows, args.out_dir, args.c5_dir)
    for split in ("train", "valid"):
        if split in rep:
            r = rep[split]
            print(f"\n  {split}: {r['clips']} clips {r['classes']}")
            print(f"    {'background':<11} {'drone':>6} {'no drone':>9} {'P(drone)':>9}")
            for c, v in r["by_background_category"].items():
                print(f"    {c:<11} {v['drone']:>6} {v['no_drone']:>9} {v['p_drone']:>9.3f}")
            print(f"    P(drone | distractor) {r['p_drone_with_distractor']:.3f} vs "
                  f"{r['p_drone_without_distractor']:.3f} without -> {r['shortcut_check']}")
            print(f"    drone sources {r['drone_sources']}, unique negative backgrounds {r['unique_background_windows']}")
    print(f"\nDone in {time.time() - t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
