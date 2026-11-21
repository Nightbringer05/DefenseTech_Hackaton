"""224x224 log-mel PNGs identical to the challenge's `spectrograms/<split>/*.png`.

Same pipeline as Adrien's `starter-kit/make_spectrogram.py` (verified MAE = 0 on the 62
challenge clips): mono 16 kHz -> first 80 000 samples (zero-pad shorter) -> mel
(n_fft 1024, hop 256, 128 bands, 50-8000 Hz, power 2) -> power_to_db(ref=max, top_db=80)
-> (dB + 80) / 80 * 255, truncated to uint8 -> flip -> bilinear 224x224 -> 3-channel RGB.
"""
from __future__ import annotations

import io

import librosa
import numpy as np
from PIL import Image

SR = 16_000
N_SAMPLES = 80_000
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 256, 128, 50.0, 8000.0
TOP_DB = 80.0
IMG = 224


def fix_length(y: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def logmel_gray(y: np.ndarray, sr: int = SR) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = fix_length(y)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                       fmin=FMIN, fmax=FMAX, power=2.0)
    db = librosa.power_to_db(m, ref=np.max, amin=1e-10, top_db=TOP_DB)
    x = np.clip((db + TOP_DB) / TOP_DB, 0.0, 1.0) * 255
    img = Image.fromarray(np.flipud(x).astype(np.uint8))
    return np.asarray(img.resize((IMG, IMG), Image.Resampling.BILINEAR), dtype=np.uint8)


def spectrogram_image(y: np.ndarray, sr: int = SR) -> Image.Image:
    return Image.fromarray(logmel_gray(y, sr)).convert("RGB")


def spectrogram_png_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    spectrogram_image(y, sr).save(buf, format="PNG")
    return buf.getvalue()
