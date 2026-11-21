"""Generate Challenge-5-identical 224x224 log-mel PNGs from audio (verified MAE=0 on 62/62 clips).

Pipeline: load mono 16 kHz -> keep first 80 000 samples (zero-pad if shorter)
-> mel (n_fft=1024, hop=256, n_mels=128, 50-8000 Hz, power=2)
-> power_to_db(ref=max, top_db=80) -> (dB+80)/80*255 -> uint8 -> flipud
-> PIL bilinear resize to 224x224 -> grayscale replicated on 3 RGB channels.

Usage (from starter-kit/):
  # 1. one file (optionally compare against an existing PNG)
  uv run --with librosa python make_spectrogram.py single --wav ../audio/train/c5_0002.wav \
      --out /tmp/c5_0002.png --compare ../spectrograms/train/c5_0002.png

  # 2. a whole folder (e.g. MAD), long files cut into 5 s windows, parallel
  uv run --with librosa python make_spectrogram.py dir --in-dir /data/MAD_dataset/training \
      --out-dir /data/mad_spec/training --workers 8 --min-rms-db -60

  # 3. one mix at a given SNR (drone over background)
  uv run --with librosa python make_spectrogram.py mix --signal drone.wav --background ../audio/train/c5_0001.wav \
      --snr 0 --out /tmp/mix.png --out-wav /tmp/mix.wav

  # 4. N random mixes (signals x backgrounds, SNR ~ U[snr_min, snr_max]) + manifest
  uv run --with librosa python make_spectrogram.py mix-random --signals-dir /data/drones \
      --backgrounds-dir /data/mad_wav_bg --n 2000 --snr-min -10 --snr-max 20 --out-dir /data/mix_spec --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from PIL import Image

SR = 16_000
N_SAMPLES = 80_000  # 5.00 s
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 256, 128, 50.0, 8000.0
TOP_DB = 80.0
IMG = 224
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ---------------------------------------------------------------- core pipeline
def load_audio(path: Path, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Mono float32 at 16 kHz (resampled if needed)."""
    y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    return y.astype(np.float32)


def fix_length(y: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Keep the first n samples (as the organisers did), zero-pad at the end if shorter."""
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def logmel_gray(y: np.ndarray) -> np.ndarray:
    """5 s mono 16 kHz -> uint8 (224, 224), bit-identical to the supplied PNGs."""
    y = fix_length(y)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    db = librosa.power_to_db(m, ref=np.max, amin=1e-10, top_db=TOP_DB)
    x = np.clip((db + TOP_DB) / TOP_DB, 0.0, 1.0) * 255  # truncation below, as in the original
    img = Image.fromarray(np.flipud(x).astype(np.uint8))
    return np.asarray(img.resize((IMG, IMG), Image.Resampling.BILINEAR), dtype=np.uint8)


def save_png(gray: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).convert("RGB").save(out)


def rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12))


def mix_at_snr(signal: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + background with SNR = 20 log10(rms_s / rms_b); both cut/padded to 5 s, peak-safe."""
    s, b = fix_length(signal), fix_length(background)
    rs = np.sqrt(np.mean(s.astype(np.float64) ** 2)) + 1e-12
    rb = np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12
    mix = s * (rb / rs) * 10 ** (snr_db / 20) + b
    peak = float(np.abs(mix).max())
    return (mix / peak * 0.99 if peak > 0.99 else mix).astype(np.float32)  # gain is irrelevant (ref=max)


# ---------------------------------------------------------------- CLI modes
def cmd_single(a: argparse.Namespace) -> None:
    gray = logmel_gray(load_audio(a.wav))
    save_png(gray, a.out)
    print(f"saved {a.out}")
    if a.compare:
        ref = np.asarray(Image.open(a.compare).convert("L"), dtype=np.float32)
        diff = np.abs(gray.astype(np.float32) - ref)
        print(f"vs {a.compare}: MAE={diff.mean():.3f} max={diff.max():.1f}")


def _segments(path: Path, hop_s: float, min_rms: float | None) -> list[tuple[float, np.ndarray]]:
    y = load_audio(path)
    win, hop = N_SAMPLES, int(hop_s * SR)
    starts = list(range(0, max(1, len(y) - win + 1), hop)) if len(y) > win else [0]
    if len(y) > win and starts[-1] + hop < len(y):  # tail window (kept if >= 2.5 s, zero-padded)
        starts.append(starts[-1] + hop)
    out = []
    for st in starts:
        seg = y[st: st + win]
        if len(seg) < win // 2:  # drop tails shorter than 2.5 s
            continue
        if min_rms is not None and rms_db(seg) < min_rms:  # skip near-silence (ref=max would blow up noise)
            continue
        out.append((st / SR, seg))
    return out


def _process_file(job: tuple[str, str, str, float, float | None]) -> list[dict]:
    src, in_dir, out_dir, hop_s, min_rms = job
    p, rel = Path(src), Path(src).relative_to(in_dir)
    rows = []
    try:
        for off, seg in _segments(p, hop_s, min_rms):
            out = Path(out_dir) / rel.parent / f"{rel.stem}__{int(round(off * 1000)):07d}ms.png"
            save_png(logmel_gray(seg), out)
            rows.append({"png": str(out), "source": str(p), "offset_s": round(off, 3), "rms_db": round(rms_db(seg), 1)})
    except Exception as exc:  # corrupt / unreadable file: log and continue
        rows.append({"source": str(p), "error": repr(exc)})
    return rows


def cmd_dir(a: argparse.Namespace) -> None:
    files = sorted(p for p in a.in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    jobs = [(str(f), str(a.in_dir), str(a.out_dir), a.hop, a.min_rms_db) for f in files]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_process_file, jobs, chunksize=8), 1):
            rows += r
            if i % 200 == 0:
                print(f"{i}/{len(files)} files")
    (a.out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    n_err = sum("error" in r for r in rows)
    print(f"{len(files)} files -> {len(rows) - n_err} PNGs, {n_err} errors | manifest: {a.out_dir / 'manifest.jsonl'}")


def cmd_mix(a: argparse.Namespace) -> None:
    mix = mix_at_snr(load_audio(a.signal), load_audio(a.background), a.snr)
    save_png(logmel_gray(mix), a.out)
    if a.out_wav:
        import soundfile as sf
        sf.write(a.out_wav, mix, SR, subtype="PCM_16")
    print(f"saved {a.out} (SNR={a.snr} dB)")


def _random_5s(path: Path, rng: random.Random) -> tuple[np.ndarray, float]:
    dur = librosa.get_duration(path=path)
    off = rng.uniform(0, max(0.0, dur - 5.0))
    return load_audio(path, offset=off, duration=5.0), off


def cmd_mix_random(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    sigs = sorted(p for p in a.signals_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    bgs = sorted(p for p in a.backgrounds_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not sigs or not bgs:
        raise SystemExit("empty signals or backgrounds folder")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        sp, bp = rng.choice(sigs), rng.choice(bgs)
        s, s_off = _random_5s(sp, rng)
        b, b_off = _random_5s(bp, rng)
        snr = rng.uniform(a.snr_min, a.snr_max)
        out = a.out_dir / f"mix_{i:06d}.png"
        save_png(logmel_gray(mix_at_snr(s, b, snr)), out)
        rows.append({"png": str(out), "signal": str(sp), "signal_offset_s": round(s_off, 3),
                     "background": str(bp), "background_offset_s": round(b_off, 3),
                     "snr_db": round(snr, 2), "has_drone_audio": True})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{a.n}")
    (a.out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    print(f"{a.n} mixes -> {a.out_dir} | manifest.jsonl written")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single")
    s.add_argument("--wav", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--compare", type=Path, default=None)

    d = sub.add_parser("dir")
    d.add_argument("--in-dir", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--hop", type=float, default=5.0, help="window hop in s (5 = no overlap)")
    d.add_argument("--min-rms-db", type=float, default=None, help="skip windows quieter than this (dBFS)")
    d.add_argument("--workers", type=int, default=4)

    m = sub.add_parser("mix")
    m.add_argument("--signal", type=Path, required=True)
    m.add_argument("--background", type=Path, required=True)
    m.add_argument("--snr", type=float, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--out-wav", type=Path, default=None)

    r = sub.add_parser("mix-random")
    r.add_argument("--signals-dir", type=Path, required=True)
    r.add_argument("--backgrounds-dir", type=Path, required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--snr-min", type=float, default=-10.0)
    r.add_argument("--snr-max", type=float, default=20.0)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    {"single": cmd_single, "dir": cmd_dir, "mix": cmd_mix, "mix-random": cmd_mix_random}[a.mode](a)


if __name__ == "__main__":
    main()
