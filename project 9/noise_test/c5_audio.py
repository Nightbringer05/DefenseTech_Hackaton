"""Numpy re-implementation of the Challenge 5 audio -> 224x224 log-mel PNG pipeline, plus
domain-harmonization transforms for external audio (e.g. MAD).

The recipe is the organizers' one (same as starter-kit/make_spectrogram.py, which needs librosa):

    16 kHz mono, first 80 000 samples (5.0 s)
    STFT: n_fft=1024, hop=256, periodic Hann, center=True with ZERO padding
    Mel: 128 bands, fmin=50 Hz, fmax=8000 Hz, Slaney scale + Slaney norm (librosa defaults)
    power_to_db(ref=max, top_db=80) -> fixed scale (dB + 80) / 80 * 255, truncated to uint8
    flip so low frequencies are at the bottom -> PIL bilinear resize to 224x224
    grayscale replicated to RGB

Checked against the 62 shipped PNGs: max |pixel error| 1, most clips bit-exact.
An earlier version used reflect padding and per-image min-max scaling: nearly identical on
challenge audio, but it stretched the contrast of external clips with < 80 dB dynamic range.

Only numpy + Pillow are required (no librosa / scipy).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from PIL import Image

SR = 16000
CLIP_SAMPLES = 5 * SR
N_FFT, HOP, N_MELS, FMIN, FMAX, TOP_DB = 1024, 256, 128, 50.0, 8000.0, 80.0


# ----------------------------------------------------------------------------- I/O
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * sw}-bit")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------- mel
def _hz_to_mel(f):
    f = np.asarray(f, float)
    logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / logstep, f / (200 / 3))


def _mel_to_hz(m):
    m = np.asarray(m, float)
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15, 1000 * np.exp(logstep * (m - 15)), m * 200 / 3)


def _mel_filterbank() -> np.ndarray:
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    hz = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fd = np.diff(hz)
    ramps = hz[:, None] - freqs[None, :]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fd[:-1, None], ramps[2:] / fd[1:, None]))
    return w * (2.0 / (hz[2:] - hz[:-2]))[:, None]


MEL_FB = _mel_filterbank()
MEL_CENTERS_HZ = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))[1:-1]
_WIN = np.hanning(N_FFT + 1)[:-1]


def power_spectrogram(x: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    xp = np.pad(x, N_FFT // 2, mode=pad_mode)
    n = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return (np.abs(np.fft.rfft(xp[idx] * _WIN, axis=1)) ** 2).T


def log_mel_db(x: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """Absolute log-mel in dB (no ref/top_db), shape [128, frames]."""
    return 10 * np.log10(np.maximum(MEL_FB @ power_spectrogram(x, pad_mode), 1e-10))


def fix_length(x: np.ndarray) -> np.ndarray:
    """First 80 000 samples (as the organizers did); zero-pad at the end if shorter."""
    return x[:CLIP_SAMPLES] if len(x) >= CLIP_SAMPLES else np.pad(x, (0, CLIP_SAMPLES - len(x)))


def spectrogram_image(x: np.ndarray) -> Image.Image:
    """Waveform (5 s @16 kHz) -> 224x224 RGB PIL image, the organizers' exact recipe."""
    D = log_mel_db(fix_length(x), pad_mode="constant")
    D = np.maximum(D - D.max(), -TOP_DB)
    g = np.clip((D + TOP_DB) / TOP_DB, 0.0, 1.0) * 255  # fixed scale; truncation below
    img = Image.fromarray(np.flipud(g).astype(np.uint8))
    return img.resize((224, 224), Image.BILINEAR).convert("RGB")


# ----------------------------------------------------------------------------- harmonization
def _fft_gain(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    X = np.fft.rfft(x)
    return np.fft.irfft(X * 10 ** (gain_db / 20.0), n=len(x)).astype(np.float32)


def _freqs(n: int) -> np.ndarray:
    return np.fft.rfftfreq(n, 1 / SR)


def lowpass(x: np.ndarray, cutoff_hz: float, width_hz: float = 250.0) -> np.ndarray:
    """Zero-phase low-pass with a raised-cosine transition (mimics C5's band-limited clips)."""
    f = _freqs(len(x))
    t = np.clip((f - cutoff_hz) / width_hz, 0, 1)
    g = 0.5 * (1 + np.cos(np.pi * t))  # 1 -> 0
    return _fft_gain(x, 20 * np.log10(np.maximum(g, 10 ** (-90 / 20))))


def spectral_tilt(x: np.ndarray, db_per_octave: float, pivot_hz: float = 500.0) -> np.ndarray:
    """Apply a log-frequency tilt above `pivot_hz` (negative = darker / steeper roll-off)."""
    f = np.maximum(_freqs(len(x)), 1.0)
    g = db_per_octave * np.log2(np.maximum(f, pivot_hz) / pivot_hz)
    return _fft_gain(x, g)


def mix_background(x: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    px, pb = np.mean(x ** 2) + 1e-12, np.mean(bg ** 2) + 1e-12
    return x + bg * np.sqrt(px / (pb * 10 ** (snr_db / 10)))


def match_rms(x: np.ndarray, target_dbfs: float, peak_ceiling: float = 0.98) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    y = x * (10 ** (target_dbfs / 20) / rms)
    pk = np.max(np.abs(y))
    return y * (peak_ceiling / pk) if pk > peak_ceiling else y


def spectral_flatness(x: np.ndarray) -> float:
    P = power_spectrogram(x) + 1e-12
    return float(np.mean(np.exp(np.mean(np.log(P), 0)) / np.mean(P, 0)))


def windows_5s(x: np.ndarray, max_windows: int = 2) -> list[np.ndarray]:
    """Non-overlapping 5 s windows, centred; clips shorter than 5 s return []."""
    n = len(x) // CLIP_SAMPLES
    if n == 0:
        return []
    n = min(n, max_windows)
    start = (len(x) - n * CLIP_SAMPLES) // 2
    return [x[start + i * CLIP_SAMPLES: start + (i + 1) * CLIP_SAMPLES] for i in range(n)]


# ----------------------------------------------------------------------------- folder defaults
def default_dirs() -> dict:
    """Defaults that work in the repo layout (`project 9/harmonize/`) and in a local
    `InputData/` + `WorkSpace/harmonize/` layout. Everything can be overridden on the CLI."""
    here = Path(__file__).resolve().parent
    c5 = next((c for c in (here.parent, here.parent.parent / "InputData")
               if (c / "dataset" / "train" / "labels.jsonl").exists()), here.parent)
    data = here / "data"  # downloads + generated datasets, git-ignored
    mad_candidates = (data / "mad-dataset" / "MAD_dataset", here.parent / "MAD_dataset",
                      here.parent / "synth" / "MAD_dataset", here.parent.parent / "MAD_dataset",
                      here.parent / "mad-dataset" / "MAD_dataset")
    mad = next((m for m in mad_candidates if (m / "training.csv").exists()), mad_candidates[0])
    return {"c5": c5, "data": data, "mad": mad}
