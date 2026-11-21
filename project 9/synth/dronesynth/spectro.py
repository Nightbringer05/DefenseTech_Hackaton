"""Log-mel spectrogram PNGs that match the challenge's `spectrograms/<split>/*.png`.

Parameters were fitted against the provided train PNGs (pixel correlation ~0.98):
n_fft=1024, hop=160, 64 Slaney mel bands, power dB with top_db=80 relative to the
clip max, mapped to [0, 255], low frequencies at the bottom, bilinear resize to 224x224
grayscale. No librosa needed.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

N_FFT = 1024
HOP = 160
N_MELS = 64
TOP_DB = 80.0
IMG = 224


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=float)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=float)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-normalised mel filterbank, same as librosa.filters.mel defaults."""
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights


def log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """Return dB log-mel (n_mels, frames), max = 0 dB, floor = -TOP_DB."""
    y = np.asarray(y, dtype=np.float64)
    pad = N_FFT // 2
    y = np.pad(y, pad, mode="constant")
    n_frames = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    win = np.hanning(N_FFT + 1)[:-1]  # periodic Hann
    spec = np.abs(np.fft.rfft(y[idx] * win, axis=1)) ** 2  # (frames, bins)
    mel = mel_filterbank(sr) @ spec.T
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= db.max()
    return np.maximum(db, -TOP_DB)


def spectrogram_image(y: np.ndarray, sr: int) -> Image.Image:
    db = log_mel(y, sr)
    px = ((db + TOP_DB) / TOP_DB * 255.0).astype(np.uint8)
    im = Image.fromarray(np.flipud(px), mode="L")
    return im.resize((IMG, IMG), Image.BILINEAR)


def spectrogram_png_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    spectrogram_image(y, sr).save(buf, format="PNG")
    return buf.getvalue()
