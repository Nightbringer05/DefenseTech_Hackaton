"""Physics-based drone sound source.

A drone is sampled from an archetype (mass, payload, rotors, prop size, blades, drag area,
motor poles, speed and altitude envelope). From that and the air state we derive:

  air density     rho = p(h) / (R * T)                         (ISA pressure at site elevation)
  speed of sound  c   = 331.3 * sqrt(1 + T_C / 273.15)
  forward drag    D   = 0.5 * rho * CdA * v^2,  tilt = atan(D / W)
  total thrust    T   = sqrt(W^2 + D^2)  (+ climb / manoeuvre throttle)
  rotor speed     n   = sqrt(T_rotor / (C_T * rho * D_prop^4))  [rev/s]   (static thrust law)
  blade passing   BPF = blades * n
  tip Mach        M   = pi * D_prop * n / c   (capped at 0.75)
  source level    L   = 10 log10(T / T_ref) + 30 log10(v_tip / v_tip_ref)   dB re. camera drone

Per rotor, thrust varies with forward-flight tilt (rear rotors work harder), manoeuvre
throttle and gust corrections (stronger for light drones in wind); RPM follows sqrt(thrust)
and harmonic amplitude follows thrust, so all of this is audible in the spectrum.

The fixed-wing archetype is a piston pusher (one 2-blade prop + 2-stroke engine firing
harmonics), the sound of long-range one-way attack drones.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

G = 9.81
R_AIR = 287.05

# Ranges are [min, max]; lists of discrete values are sampled uniformly.
ARCHETYPES = {
    "mini": dict(label="Mini quad (<250 g)", mass_kg=[0.18, 0.25], max_payload_kg=0.05, rotors=[4],
                 prop_in=[3.5, 4.7], blades=[2, 3], ct=[0.09, 0.11], cda_m2=[0.006, 0.012], pole_pairs=[6, 7],
                 speed_mps=[0, 16], altitude_m=[5, 120]),
    "camera": dict(label="Camera quad (Mavic class)", mass_kg=[0.7, 1.4], max_payload_kg=0.3, rotors=[4],
                   prop_in=[7.5, 9.5], blades=[2], ct=[0.08, 0.10], cda_m2=[0.02, 0.04], pole_pairs=[7],
                   speed_mps=[0, 21], altitude_m=[10, 150]),
    "fpv": dict(label="FPV racer / strike quad", mass_kg=[0.5, 1.2], max_payload_kg=1.5, rotors=[4],
                prop_in=[5.0, 7.0], blades=[3], ct=[0.11, 0.13], cda_m2=[0.012, 0.025], pole_pairs=[7],
                speed_mps=[5, 45], altitude_m=[2, 60]),
    "heavy": dict(label="Heavy lift (Matrice class)", mass_kg=[4.0, 9.0], max_payload_kg=3.0, rotors=[4, 6],
                  prop_in=[17, 22], blades=[2], ct=[0.09, 0.11], cda_m2=[0.15, 0.3], pole_pairs=[12, 14],
                  speed_mps=[0, 23], altitude_m=[15, 200]),
    "agri": dict(label="Agricultural hexa/octo", mass_kg=[18, 35], max_payload_kg=30.0, rotors=[6, 8],
                 prop_in=[28, 36], blades=[2], ct=[0.09, 0.11], cda_m2=[0.5, 0.9], pole_pairs=[14],
                 speed_mps=[0, 10], altitude_m=[3, 30]),
    "fixed_wing": dict(label="Fixed-wing piston (one-way attack)", mass_kg=[150, 220], max_payload_kg=50.0,
                       rotors=[1], prop_in=[26, 32], blades=[2], engine_rpm=[5000, 7200], cylinders=[2, 4],
                       speed_mps=[40, 55], altitude_m=[60, 1000]),
}

# Reference drone for source level: Mavic class in hover.
_T_REF = 0.9 * G
_VTIP_REF = 60.0


def air(temp_c: float, elevation_m: float) -> tuple[float, float]:
    """(density kg/m^3, speed of sound m/s)."""
    p = 101325.0 * (1 - 2.25577e-5 * elevation_m) ** 5.25588
    t = temp_c + 273.15
    return p / (R_AIR * t), 331.3 * np.sqrt(1 + temp_c / 273.15)


def _u(rng, v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return float(rng.uniform(*v))
    if isinstance(v, (list, tuple)):
        return v[int(rng.integers(len(v)))]
    return v


def sample_spec(rng, dcfg: dict, temp_c: float, elevation_m: float) -> dict:
    """Pick an archetype and draw a concrete airframe."""
    types = {k: float(w) for k, w in dcfg["types"].items() if float(w) > 0 and k in ARCHETYPES}
    if not types:
        types = {"camera": 1.0}
    names = list(types)
    p = np.array([types[k] for k in names])
    kind = names[int(rng.choice(len(names), p=p / p.sum()))]
    a = ARCHETYPES[kind]
    rho, c = air(temp_c, elevation_m)
    payload = a["max_payload_kg"] * float(rng.uniform(*dcfg["payload_frac"]))
    spec = dict(type=kind, mass_kg=round(float(rng.uniform(*a["mass_kg"])), 3), payload_kg=round(payload, 3),
                rotors=int(_u(rng, a["rotors"])), prop_in=round(float(rng.uniform(*a["prop_in"])), 1),
                blades=int(_u(rng, a["blades"])), rho=rho, c=c, temp_c=temp_c, elevation_m=elevation_m,
                speed_range=list(a["speed_mps"]), altitude_range=list(a["altitude_m"]))
    if kind == "fixed_wing":
        spec.update(engine_rpm=float(rng.uniform(*a["engine_rpm"])), cylinders=int(_u(rng, a["cylinders"])))
    else:
        spec.update(ct=float(rng.uniform(*a["ct"])), cda=float(rng.uniform(*a["cda_m2"])),
                    pole_pairs=int(_u(rng, a["pole_pairs"])))
    return spec


def hover_rpm(spec: dict, speed: float = 0.0) -> dict:
    """Steady-flight thrust, tilt, rotor RPM, BPF, tip Mach and relative level."""
    W = (spec["mass_kg"] + spec["payload_kg"]) * G
    D_m = spec["prop_in"] * 0.0254
    if spec["type"] == "fixed_wing":
        n = spec["engine_rpm"] / 60
        vtip = np.hypot(np.pi * D_m * n, speed)
        return dict(thrust_n=None, tilt_deg=0.0, rpm=spec["engine_rpm"], bpf_hz=spec["blades"] * n,
                    tip_mach=vtip / spec["c"], level_db=18.0 + 10 * np.log10(spec["cylinders"] / 2))
    drag = 0.5 * spec["rho"] * spec["cda"] * speed ** 2
    T = float(np.hypot(W, drag))
    n = np.sqrt(T / spec["rotors"] / (spec["ct"] * spec["rho"] * D_m ** 4))
    n = min(n, 0.75 * spec["c"] / (np.pi * D_m))  # keep tips subsonic
    vtip = np.pi * D_m * n
    level = 10 * np.log10(T / _T_REF) + 30 * np.log10(vtip / _VTIP_REF)
    return dict(thrust_n=T, tilt_deg=float(np.degrees(np.arctan2(drag, W))), rpm=n * 60,
                bpf_hz=spec["blades"] * n, tip_mach=vtip / spec["c"], level_db=float(level))


def _walk(rng, n, sr, scale, smooth_hz):
    w = sps.sosfilt(sps.butter(1, smooth_hz, fs=sr, output="sos"), rng.normal(size=n))
    s = np.std(w)
    return w / s * scale if s > 0 else w


def _bandnoise(rng, n, sr, lo, hi):
    return sps.sosfilt(sps.butter(2, [lo, min(hi, sr / 2 * 0.95)], btype="band", fs=sr, output="sos"),
                       rng.normal(size=n))


def _norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def throttle_profile(rng, n, sr, manoeuvre: str) -> np.ndarray:
    """Fractional change of total thrust over time."""
    base = _walk(rng, n, sr, 0.02, 1.0)
    if manoeuvre == "climb":
        t = np.arange(n) / sr
        t0, dur = rng.uniform(0, n / sr * 0.6), rng.uniform(1.0, 3.0)
        amt = rng.uniform(0.1, 0.35)
        base += amt * np.clip((t - t0) / 0.4, 0, 1) * np.clip((t0 + dur - t) / 0.4, 0, 1)
    elif manoeuvre == "aggressive":
        out = np.zeros(n)
        i = 0
        while i < n:
            L = int(sr * rng.uniform(0.15, 0.9))
            out[i : i + L] = rng.uniform(-0.35, 0.6)
            i += L + int(sr * rng.uniform(0.05, 0.6))
        base += sps.sosfilt(sps.butter(2, 6.0, fs=sr, output="sos"), out)  # motor/ESC response
    return np.clip(base, -0.6, 1.2)


def render(rng, n, sr, dcfg: dict, spec: dict, speed: float, manoeuvre: str, wind: float):
    """Return (signal, params) for one airframe in steady forward flight at `speed`."""
    st = hover_rpm(spec, speed)
    t = np.arange(n) / sr
    n_harm = int(rng.integers(dcfg["harmonics"][0], dcfg["harmonics"][1] + 1))
    # big slow rotors still radiate blade harmonics up to a few kHz
    n_harm = int(min(80, max(n_harm, rng.uniform(1800, 3500) / max(st["bpf_hz"], 1.0))))
    rolloff = float(rng.uniform(*dcfg["harmonic_rolloff"]))
    hiss = float(rng.uniform(*dcfg["hiss"]))
    whine = float(rng.uniform(*dcfg["motor_whine"]))
    out = np.zeros(n)

    if spec["type"] == "fixed_wing":
        f_rev = st["rpm"] / 60 * (1 + _walk(rng, n, sr, 0.01, 0.5))
        ph = 2 * np.pi * np.cumsum(f_rev) / sr
        cyl = spec["cylinders"]
        for h in range(1, 60):
            if h * f_rev.mean() > sr / 2 * 0.9:
                break
            amp = h ** -0.7 * (2.5 if h % cyl == 0 else 0.6) * (1.8 if h % spec["blades"] == 0 else 1.0)
            out += amp * rng.uniform(0.6, 1.0) * np.sin(h * ph + rng.uniform(0, 6.28))
        out = _norm(out) + 0.35 * _norm(_bandnoise(rng, n, sr, 80, 3000))  # exhaust
        params = dict(manoeuvre="cruise", wind_mps=round(wind, 1))
    else:
        thr = throttle_profile(rng, n, sr, manoeuvre)
        W = (spec["mass_kg"] + spec["payload_kg"]) * G
        # gust corrections: stronger when wind is high and the airframe light
        gust_pct = 0.004 + 0.012 * wind / np.sqrt(W)
        tilt = np.radians(st["tilt_deg"])
        D_m = spec["prop_in"] * 0.0254
        k = spec["ct"] * spec["rho"] * D_m ** 4
        n_max = 0.75 * spec["c"] / (np.pi * D_m)
        heading = rng.uniform(0, 2 * np.pi)
        for i in range(spec["rotors"]):
            phi = 2 * np.pi * i / spec["rotors"] + np.pi / spec["rotors"]
            diff = 0.35 * np.tan(tilt) * -np.cos(phi - heading)  # rear rotors carry more
            T_i = st["thrust_n"] / spec["rotors"] * (1 + thr) * (1 + diff + _walk(rng, n, sr, gust_pct * 4, 3.0))
            T_i = np.maximum(T_i, 0.05 * st["thrust_n"] / spec["rotors"])
            n_i = np.minimum(np.sqrt(T_i / k), n_max)
            amp_env = T_i / (st["thrust_n"] / spec["rotors"])
            ph = 2 * np.pi * np.cumsum(spec["blades"] * n_i) / sr + rng.uniform(0, 6.28)
            for h in range(1, n_harm + 1):
                if h * spec["blades"] * n_i.mean() > sr / 2 * 0.95:
                    break
                out += amp_env * h ** (-rolloff) * rng.uniform(0.6, 1.0) * np.sin(h * ph + rng.uniform(0, 6.28))
            if whine > 0:
                out += whine * 0.3 * amp_env * np.sin(2 * np.pi * np.cumsum(spec["pole_pairs"] * n_i) / sr)
        out = _norm(out)
        out = out * (1 + 0.12 * np.sin(2 * np.pi * st["rpm"] / 60 * t))  # shaft-rate AM
        out += hiss * (0.6 + 0.4 * np.clip(st["tip_mach"] / 0.5, 0, 1.5)) * _norm(_bandnoise(rng, n, sr, 300, 7000))
        params = dict(manoeuvre=manoeuvre, wind_mps=round(wind, 1), tilt_deg=round(st["tilt_deg"], 1),
                      thrust_n=round(st["thrust_n"], 2), gust_pct=round(gust_pct * 100, 2))

    params.update(drone_type=spec["type"], mass_kg=spec["mass_kg"], payload_kg=spec["payload_kg"],
                  rotors=spec["rotors"], prop_in=spec["prop_in"], blades=spec["blades"],
                  rpm=round(st["rpm"]), bpf_hz=round(float(st["bpf_hz"]), 1), tip_mach=round(float(st["tip_mach"]), 3),
                  level_db=round(float(st["level_db"]), 1), temp_c=round(spec["temp_c"], 1),
                  elevation_m=round(spec["elevation_m"]), air_density=round(spec["rho"], 3),
                  speed_of_sound=round(spec["c"], 1), harmonics=n_harm)
    return _norm(out), params


def archetype_table(temp_c: float = 15.0, elevation_m: float = 0.0) -> list[dict]:
    """Hover RPM / BPF / level at the archetype's mean mass, for the UI."""
    rho, c = air(temp_c, elevation_m)
    rows = []
    for k, a in ARCHETYPES.items():
        mid = lambda r: (r[0] + r[-1]) / 2  # noqa: E731
        spec = dict(type=k, mass_kg=mid(a["mass_kg"]), payload_kg=0.0, rotors=a["rotors"][0],
                    prop_in=mid(a["prop_in"]), blades=a["blades"][0], rho=rho, c=c,
                    ct=mid(a.get("ct", [0.1, 0.1])), cda=mid(a.get("cda_m2", [0.02, 0.02])),
                    engine_rpm=mid(a.get("engine_rpm", [6000, 6000])), cylinders=a.get("cylinders", [2])[0])
        st = hover_rpm(spec)
        rows.append(dict(type=k, label=a["label"], mass_kg=a["mass_kg"], max_payload_kg=a["max_payload_kg"],
                         rotors=a["rotors"], prop_in=a["prop_in"], blades=a["blades"],
                         speed_mps=a["speed_mps"], altitude_m=a["altitude_m"],
                         rpm=round(st["rpm"]), bpf_hz=round(st["bpf_hz"]), tip_mach=round(st["tip_mach"], 2),
                         level_db=round(st["level_db"], 1)))
    return rows
