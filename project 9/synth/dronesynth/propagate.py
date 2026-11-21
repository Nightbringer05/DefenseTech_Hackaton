"""Move a source along a trajectory past one microphone at the origin.

Two engines give the same physics (retarded-time Doppler + 1/r spreading):
  * "numpy"   fast, no extra dependency
  * "acoular" acoular.MovingPointSource, the reference implementation
Air absorption is approximated with a distance-dependent low-pass in both.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sps

C = 343.0
PAD_S = 1.0  # source pre-roll so the output starts with sound already in flight


def acoular_available() -> bool:
    try:
        import acoular  # noqa: F401
        return True
    except Exception:
        return False


def sample_trajectory(rng, dur, cfg):
    """Return (mode, fn(t)->(N,3) positions, params). t covers [0, dur + PAD_S]."""
    mode = str(rng.choice(list(cfg["modes"].keys()), p=_probs(cfg["modes"])))
    alt = rng.uniform(*cfg["altitude_m"])
    speed = rng.uniform(*cfg["speed_mps"])
    head = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(head), np.sin(head), 0.0])
    perp = np.array([-d[1], d[0], 0.0])
    total = dur + PAD_S
    if mode == "flyby":
        off = rng.uniform(*cfg["lateral_offset_m"])
        t_cpa = rng.uniform(PAD_S, total)  # closest approach somewhere inside the clip
        p0 = perp * off + np.array([0, 0, alt]) - d * speed * t_cpa
        vel = d * speed
    elif mode == "approach":
        start = rng.uniform(80, 250)
        p0 = -d * start + np.array([0, 0, alt])
        vel = d * speed * rng.uniform(0.5, 1.0)
    else:  # hover
        off = rng.uniform(*cfg["lateral_offset_m"])
        p0 = perp * off + d * rng.uniform(0, 60) + np.array([0, 0, alt])
        vel = np.zeros(3)
        speed = 0.0
    wob = rng.uniform(0.2, 1.5) if mode == "hover" else 0.0
    ph = rng.uniform(0, 6.28, 3)

    def pos(t):
        t = np.asarray(t, float)[:, None]
        p = p0[None, :] + vel[None, :] * t
        if wob:
            p = p + wob * np.sin(2 * np.pi * np.array([0.3, 0.23, 0.17]) * t + ph)
        return p

    r = np.linalg.norm(pos(np.linspace(PAD_S, total, 50)), axis=1)
    return mode, pos, dict(mode=mode, altitude_m=round(alt, 1), speed_mps=round(speed, 1),
                           min_range_m=round(float(r.min()), 1), max_range_m=round(float(r.max()), 1))


def _probs(d):
    v = np.array(list(d.values()), float)
    return v / v.sum()


def _air_absorption(y, sr, r_mean):
    fc = float(np.clip(9000 * np.exp(-r_mean / 400.0), 1500, sr / 2 * 0.95))
    sos = sps.butter(2, fc, fs=sr, output="sos")
    return sps.sosfilt(sos, y)


def propagate(src, sr, n_out, pos, engine="numpy"):
    """src: source signal of length n_out + PAD_S*sr. Returns mic signal of length n_out."""
    pad = int(PAD_S * sr)
    if engine == "acoular" and acoular_available():
        y = _acoular(src, sr, pos)
    else:
        y = _numpy(src, sr, pos)
    y = y[pad : pad + n_out]
    if len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)))
    r = np.linalg.norm(pos(np.linspace(PAD_S, PAD_S + n_out / sr, 20)), axis=1)
    return _air_absorption(y, sr, float(r.mean()))


def _numpy(src, sr, pos):
    n = len(src)
    t = np.arange(n) / sr
    te = t - np.linalg.norm(pos(t), axis=1) / C
    for _ in range(3):  # fixed-point solve t = te + r(te)/c
        te = t - np.linalg.norm(pos(te), axis=1) / C
    r = np.linalg.norm(pos(te), axis=1)
    y = np.interp(te * sr, np.arange(n), src, left=0.0, right=0.0)
    return y / np.maximum(r, 1.0)


def _acoular(src, sr, pos):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import acoular as ac
    n = len(src)
    T = n / sr
    knots = np.linspace(0, T, 12)
    traj = ac.Trajectory(points={float(k): tuple(map(float, p)) for k, p in zip(knots, pos(knots))})
    mics = ac.MicGeom(pos_total=np.zeros((3, 1)))
    gen = ac.GenericSignalGenerator(source=ac.TimeSamples(data=src[:, None].astype(np.float64), sample_freq=sr))
    mps = ac.MovingPointSource(signal=gen, mics=mics, trajectory=traj, conv_amp=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.concatenate(list(mps.result(4096)))[:, 0]
    return y
