"""Procedural sound sources. Every function takes an rng and returns (signal, params)."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _ui(rng, lo_hi):
    lo, hi = lo_hi
    return int(rng.integers(int(lo), int(hi) + 1))


def _norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def _bandnoise(rng, n, sr, lo, hi, order=2):
    x = rng.normal(size=n)
    hi = min(hi, sr / 2 * 0.98)
    lo = max(lo, 1.0)
    if lo >= hi:
        return x
    sos = sps.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sps.sosfilt(sos, x)


def _random_walk(rng, n, sr, scale, smooth_hz=2.0):
    """Smooth zero-mean wander with roughly `scale` standard deviation."""
    w = rng.normal(size=n)
    sos = sps.butter(1, smooth_hz, fs=sr, output="sos")
    w = sps.sosfilt(sos, w)
    s = np.std(w)
    return w / s * scale if s > 0 else w


def drone(rng, n, sr, cfg):
    """Multirotor: per-rotor blade-pass harmonics, RPM wander, motor whine, broadband hiss."""
    rotors = int(rng.choice(cfg["rotors"]))
    blades = int(rng.choice(cfg["blades"]))
    rpm = _u(rng, cfg["rpm"])
    spread = _u(rng, cfg["rpm_spread_pct"]) / 100
    jitter = _u(rng, cfg["rpm_jitter_pct"]) / 100
    n_harm = _ui(rng, cfg["harmonics"])
    rolloff = _u(rng, cfg["harmonic_rolloff"])
    hiss = _u(rng, cfg["hiss"])
    whine = _u(rng, cfg["motor_whine"])
    t = np.arange(n) / sr
    out = np.zeros(n)
    common = _random_walk(rng, n, sr, jitter, 1.0)  # shared throttle changes
    for _ in range(rotors):
        r_rpm = rpm * (1 + rng.uniform(-spread, spread))
        inst = r_rpm * (1 + common + _random_walk(rng, n, sr, jitter * 0.5, 3.0))
        f_bpf = blades * inst / 60
        phase = 2 * np.pi * np.cumsum(f_bpf) / sr + rng.uniform(0, 2 * np.pi)
        for h in range(1, n_harm + 1):
            if h * f_bpf.mean() > sr / 2 * 0.95:
                break
            amp = h ** (-rolloff) * rng.uniform(0.6, 1.0)
            out += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
        if whine > 0:
            poles = int(rng.choice([7, 12, 14]))  # electrical frequency = pole pairs * rev/s
            f_e = poles * inst / 60
            out += whine * 0.3 * np.sin(2 * np.pi * np.cumsum(f_e) / sr)
    out = _norm(out)
    # blade-rate amplitude modulation gives the characteristic "buzz"
    am = 1 + 0.15 * np.sin(2 * np.pi * (rpm / 60) * t)
    out = out * am + hiss * _norm(_bandnoise(rng, n, sr, 300, 6000))
    params = dict(rotors=rotors, blades=blades, rpm=round(rpm), bpf_hz=round(blades * rpm / 60, 1),
                  rpm_spread_pct=round(spread * 100, 2), rpm_jitter_pct=round(jitter * 100, 2),
                  harmonics=n_harm, hiss=round(hiss, 3))
    return _norm(out), params


def vehicle(rng, n, sr, cfg):
    """Combustion engine: firing-frequency harmonics, low-passed, plus tyre/road noise."""
    rpm = _u(rng, cfg["engine_rpm"])
    cyl = int(rng.choice([4, 6, 8]))
    f_fire = rpm / 60 * cyl / 2
    drift = _random_walk(rng, n, sr, 0.08, 0.5)
    inst = f_fire * (1 + drift)
    phase = 2 * np.pi * np.cumsum(inst) / sr
    out = np.zeros(n)
    for h in range(1, 30):
        if h * f_fire > 2500:
            break
        out += (h ** -0.8) * rng.uniform(0.3, 1.0) * np.sin(h * phase + rng.uniform(0, 6.28))
    out = _norm(out) * (1 + 0.3 * _random_walk(rng, n, sr, 1.0, 8.0))
    out += _u(rng, (0.2, 0.8)) * _norm(_bandnoise(rng, n, sr, 80, 1500))
    sos = sps.butter(2, _u(rng, (800, 3000)), fs=sr, output="sos")
    return _norm(sps.sosfilt(sos, out)), dict(engine_rpm=round(rpm), cylinders=cyl, firing_hz=round(f_fire, 1))


def explosion(rng, n, sr, cfg):
    """Impulsive blast: sharp onset, low-passed noise with exponential decay, optional echo."""
    out = np.zeros(n)
    onset = int(rng.uniform(0.05, 0.6) * n)
    tau = _u(rng, (0.15, 1.2))
    k = np.arange(n - onset) / sr
    burst = _bandnoise(rng, n - onset, sr, 20, _u(rng, (300, 2500)), order=2) * np.exp(-k / tau)
    burst[: int(0.004 * sr)] *= 3.0  # crack
    out[onset:] = burst
    if rng.random() < 0.6:
        d = int(_u(rng, (0.1, 0.8)) * sr)
        out[d:] += _u(rng, (0.2, 0.5)) * out[:-d]
    return _norm(out), dict(onset_s=round(onset / sr, 2), decay_s=round(tau, 2))


def ambient(rng, n, sr, cfg=None):
    """Outdoor bed: pink-ish noise, wind gusts, occasional bird chirps."""
    white = rng.normal(size=n)
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    pink = _norm(sps.lfilter(b, a, white))
    gust = np.clip(1 + _random_walk(rng, n, sr, 0.8, 0.4), 0.05, None)
    wind = _norm(_bandnoise(rng, n, sr, 20, 400)) * gust
    out = 0.5 * pink + _u(rng, (0.0, 1.0)) * wind
    for _ in range(int(rng.integers(0, 4))):
        s = int(rng.uniform(0, n - 0.3 * sr))
        L = int(_u(rng, (0.05, 0.25)) * sr)
        f0, f1 = _u(rng, (2000, 5000)), _u(rng, (2000, 7000))
        tt = np.arange(L) / sr
        chirp = np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * tt[-1])))
        out[s : s + L] += _u(rng, (0.05, 0.3)) * chirp * np.hanning(L)
    return _norm(out), {}


# ---------------------------------------------------------------- salvos

def _shot(rng, sr, heavy=False, distance=None):
    """One report: optional supersonic crack (N-wave) + muzzle blast + short tail."""
    dist = distance if distance is not None else rng.uniform(30, 600)
    L = int(sr * (rng.uniform(0.6, 1.8) if heavy else rng.uniform(0.15, 0.5)))
    out = np.zeros(L)
    t = np.arange(L) / sr
    # muzzle blast: band-limited noise with a fast attack and exponential decay
    hi = rng.uniform(150, 600) if heavy else rng.uniform(1500, 6000)
    tau = rng.uniform(0.12, 0.45) if heavy else rng.uniform(0.015, 0.06)
    blast = _bandnoise(rng, L, sr, 20 if heavy else 200, hi, order=2) * np.exp(-t / tau)
    blast[: int(0.002 * sr)] *= 2.5
    if heavy:  # low-frequency thump
        f = rng.uniform(25, 60)
        blast += 1.5 * np.std(blast) * np.sin(2 * np.pi * f * t) * np.exp(-t / (tau * 0.7))
    out += blast
    # supersonic bullets: a sharp N-wave arriving before the blast
    if not heavy and rng.random() < 0.5:
        nw = int(sr * rng.uniform(0.0003, 0.0006))
        lead = int(sr * min(0.2, dist / 343.0 * 0.4))
        pre = np.zeros(lead + L)
        pre[:nw] = np.linspace(1, -1, nw) * np.max(np.abs(blast)) * rng.uniform(0.5, 1.2)
        pre[lead:] += out
        out = pre
    # distance: air absorption + level
    fc = float(np.clip(12000 * np.exp(-dist / 500.0), 800, sr / 2 * 0.95))
    out = sps.sosfilt(sps.butter(2, fc, fs=sr, output="sos"), out)
    return out / (max(dist, 10) / 30.0)


def salvo(rng, n, sr, cfg):
    """Burst of automatic fire, a rifle volley, or an artillery salvo, with echoes."""
    c = cfg.get("explosion", {})
    kinds = c.get("salvo_types", {"burst": 0.5, "volley": 0.25, "artillery": 0.25})
    names = list(kinds)
    p = np.array([kinds[k] for k in names], float)
    kind = str(rng.choice(names, p=p / p.sum()))
    out = np.zeros(n)
    t0 = rng.uniform(0.05, 0.4) * n / sr
    times, heavy = [], kind == "artillery"
    if kind == "burst":
        rpm = rng.uniform(*c.get("burst_rpm", [500, 950]))
        rounds = int(rng.integers(*np.add(c.get("rounds", [3, 30]), [0, 1])))
        dt = 60.0 / rpm
        times = [t0 + i * dt * rng.uniform(0.93, 1.07) for i in range(rounds)]
        if rng.random() < 0.4:  # a second burst
            gap = times[-1] + rng.uniform(0.3, 1.2)
            times += [gap + i * dt * rng.uniform(0.93, 1.07) for i in range(int(rng.integers(3, 15)))]
        dist = rng.uniform(30, 600)
    elif kind == "volley":
        shooters = int(rng.integers(3, 12))
        volleys = int(rng.integers(1, 4))
        for v in range(volleys):
            base = t0 + v * rng.uniform(0.8, 1.6)
            times += [base + abs(rng.normal(0, 0.04)) for _ in range(shooters)]
        dist = rng.uniform(50, 800)
    else:
        guns = int(rng.integers(2, 7))
        times = sorted(t0 + rng.uniform(0, 3.0) for _ in range(guns))
        dist = rng.uniform(300, 4000)
    for ti in times:
        s = _shot(rng, sr, heavy, dist * rng.uniform(0.9, 1.1)) * rng.uniform(0.7, 1.0)
        i = int(ti * sr)
        if i >= n:
            break
        out[i : i + len(s)] += s[: n - i]
    # terrain echoes
    for _ in range(int(rng.integers(0, 3))):
        d = int(sr * rng.uniform(0.15, 1.2))
        if d < n:
            echo = sps.sosfilt(sps.butter(2, 1500, fs=sr, output="sos"), out[:-d])
            out[d:] += rng.uniform(0.15, 0.4) * echo
    params = dict(salvo=kind, shots=int(sum(1 for ti in times if ti * sr < n)), salvo_range_m=round(float(dist)))
    if kind == "burst":
        params["rate_rpm"] = round(float(rpm))
    return _norm(out), params
