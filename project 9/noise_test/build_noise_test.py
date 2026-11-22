"""Build the noise test: real drone recordings in real noise at fixed levels, and the same noise alone.

You do not need this script to run the test (see run_noise_test.py); it documents how the clips
were made and rebuilds them from the source audio.

Step 1, --select (done once, needs the harmonize tools of branch project9-cpu-train and the audio):
    picks source recordings that none of our training sets used and writes manifest.json
Step 2 (default): renders every clip listed in manifest.json
    spectrograms/valid/*.png     the organizers' spectrogram recipe (c5_audio.py), saved as grayscale
    dataset/valid/labels.jsonl   challenge format + condition, drone_vs_noise_db, noise_kind, unit
    examples/                    two units as WAV files, to hear what every level sounds like
    preview.png                  four units, every level side by side

One "unit" = one noise recording + one drone recording:
    noise alone | drone at +10, +5, 0, -5, -10, -15 dB relative to the noise | (half the units) drone alone
The drone and the noise are the same in every clip of a unit, only their ratio changes.

    python build_noise_test.py --mad-dir MAD_dataset --esc50-dir ESC-50-master --real-drone-dir real_drones

Rendering needs numpy + Pillow only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import c5_audio as A

HERE = Path(__file__).resolve().parent
LEVELS = (10, 5, 0, -5, -10, -15)        # drone power relative to the noise power, in dB
LOUDNESS_DBFS = -25.0                    # every clip is brought to the same loudness
KINDS = {                                # noise kind -> (challenge sound class, what it contains)
    "ambient": ("ambient_outdoor", "wind, rain, birds, animals"),
    "speech": ("ambient_outdoor", "voice and radio communication"),
    "vehicle": ("vehicle_mechanical", "ground vehicles, engines, trains"),
    "tonal_machine": ("vehicle_mechanical", "helicopter, jet, chainsaw, siren"),
    "explosion": ("explosion", "shelling, fireworks"),
    "gunfire": ("explosion", "gunshots"),
}


def tag(db: int) -> str:
    return f"{'p' if db >= 0 else 'm'}{abs(db):02d}"


def resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == A.SR:
        return x
    n_out = int(round(len(x) * A.SR / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, complex)
    m = min(len(Y), len(X))
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)


def load_window(root: Path, src: dict) -> np.ndarray:
    x, sr = A.read_wav(root / src["path"])
    x = resample(x, sr)
    return A.fix_length(x[src["start"]:src["start"] + A.CLIP_SAMPLES]).astype(np.float32)


# ----------------------------------------------------------------------------------------- select
def select(args) -> None:
    sys.path.insert(0, str(args.harmonize_dir))
    import generate_synthetic_dataset as G

    G.MAD_CATEGORY = {**G.MAD_CATEGORY, 1: "gunfire"}          # gunshots: used by none of our training sets
    ns = SimpleNamespace(c5_dir=args.c5_dir, mad_dir=args.mad_dir, esc50_dir=args.esc50_dir, extra_bg=[],
                         real_drone_dir=[args.real_drone_dir], real_drone_group="dir", max_windows_per_file=1,
                         seed=args.split_seed, valid_frac=args.valid_frac)
    # recordings on the held-out side of the split our training sets were built with
    held = [t for t in G.collect_tasks(ns) if G.split_of(t["group"], ns) == "valid"]
    windows = []
    for i, t in enumerate(held):
        ws, _ = G.index_file(t)                                  # 5 s windows that pass the quality checks
        windows += [w for w in ws if w["start"] >= 0]
        if i % 300 == 0:
            print(f"  checked {i}/{len(held)} held-out files", flush=True)

    def norm(p: str) -> str:
        return str(Path(p).resolve()).lower()

    used = set()                                                 # every source a training set really used
    for f in args.trained_on:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            used.add(norm(r["bg_path"]))
            if r.get("drone") and r["drone"].get("path"):
                used.add(norm(r["drone"]["path"]))
    overlap = [w["path"] for w in windows if norm(w["path"]) in used]
    assert not overlap, f"{len(overlap)} selected sources were used for training, e.g. {overlap[:3]}"
    print(f"  {len(windows)} held-out windows, none among the {len(used)} sources used for training")

    roots = {"mad": args.mad_dir.resolve(), "esc50": args.esc50_dir.resolve(), "drones": args.real_drone_dir.resolve()}

    def src(w: dict) -> dict:
        root = "drones" if w["kind"] == "drone" else w["source"]
        return {"root": root, "path": Path(w["path"]).resolve().relative_to(roots[root]).as_posix(),
                "start": int(w["start"]), "recording": w["group"]}

    rng = np.random.default_rng(args.seed)

    def spread(ws: list[dict], k: int) -> list[dict]:
        """k windows from as many different recordings as possible."""
        by = defaultdict(list)
        for w in sorted(ws, key=lambda w: (w["group"], w["path"], w["start"])):
            by[w["group"]].append(w)
        groups = sorted(by)
        rng.shuffle(groups)
        for g in groups:
            rng.shuffle(by[g])
        out = []
        while len(out) < k and any(by[g] for g in groups):
            for g in groups:
                if by[g] and len(out) < k:
                    out.append(by[g].pop())
        return out

    rename = {"tonal_neg": "tonal_machine"}
    by_kind = defaultdict(list)
    for w in windows:
        by_kind["drone" if w["kind"] == "drone" else rename.get(w["category"], w["category"])].append(w)
    n = args.units_per_kind
    noise = {}
    for kind in KINDS:
        got = spread(by_kind[kind], 2 * n)
        print(f"  {kind:<14} {len(by_kind[kind]):>4} windows from {len({w['group'] for w in by_kind[kind]}):>3} recordings"
              f" -> {len(got)} used ({len({w['group'] for w in got})} recordings)")
        if len(got) < 2 * n:
            sys.exit(f"not enough held-out audio for noise kind {kind}")
        noise[kind] = got
    drones = spread(by_kind["drone"], n * len(KINDS))
    print(f"  {'drone':<14} {len(by_kind['drone']):>4} windows from {len({w['group'] for w in by_kind['drone']}):>3} recordings"
          f" -> {len(drones)} used ({len({w['group'] for w in drones})} recordings)")
    if len(drones) < n * len(KINDS):
        sys.exit("not enough held-out drone recordings")

    units, u = [], 0
    for i in range(n):                                            # units with a drone, noise kinds interleaved
        for kind in KINDS:
            u += 1
            units.append({"unit": u, "noise_kind": kind, "noise": src(noise[kind][i]), "drone": src(drones[u - 1]),
                          "drone_alone": u % 2 == 1})
    for i in range(n, 2 * n):                                     # extra noise-only units
        for kind in KINDS:
            u += 1
            units.append({"unit": u, "noise_kind": kind, "noise": src(noise[kind][i]), "drone": None, "drone_alone": False})
    first = {k: next(x["unit"] for x in units if x["noise_kind"] == k) for k in KINDS}
    man = {"levels_db": list(LEVELS), "seed": args.seed, "units": units,
           "example_units": [first["vehicle"], first["gunfire"]],
           "preview_units": [first["ambient"], first["vehicle"], first["tonal_machine"], first["gunfire"]]}
    (HERE / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"  wrote manifest.json: {n * len(KINDS)} units with a drone, {n * len(KINDS)} noise-only units")


# ----------------------------------------------------------------------------------------- render
def render(args) -> None:
    man = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    roots = {"mad": args.mad_dir, "esc50": args.esc50_dir, "drones": args.real_drone_dir}
    for name, r in roots.items():
        if r is None or not Path(r).exists():
            sys.exit(f"source folder for '{name}' not found: {r}")
    spec_dir = HERE / "spectrograms" / "valid"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for old in spec_dir.glob("nt_*.png"):
        old.unlink()
    rows, tiles = [], {}

    def emit(id_: str, x: np.ndarray, unit: dict, **meta) -> np.ndarray:
        x = A.match_rms(x, LOUDNESS_DBFS)
        rel = f"spectrograms/valid/{id_}.png"
        img = A.spectrogram_image(x).convert("L")                # .convert("RGB") gives back the organizers' 3 channels
        img.save(HERE / rel, optimize=True)
        tiles[id_] = img
        rows.append({"id": id_, "split": "valid", "spectrogram": rel, **meta, "unit": unit["unit"],
                     "noise_source": (f"{unit['noise']['root']}:{unit['noise']['path']}@{unit['noise']['start']}"
                                      if meta["condition"] != "drone_alone" else None),
                     "drone_source": unit["drone"]["path"] if meta["has_drone_audio"] else None})
        return x

    for unit in man["units"]:
        uid = f"nt_{unit['unit']:03d}"
        kind = unit["noise_kind"]
        noise = load_window(Path(roots[unit["noise"]["root"]]), unit["noise"])
        audio = {"0_noise_alone": emit(f"{uid}_noise", noise, unit, has_drone_audio=False, sound_class=KINDS[kind][0],
                                       condition="noise_alone", drone_vs_noise_db=None, noise_kind=kind)}
        if unit["drone"]:
            drone = load_window(Path(roots["drones"]), unit["drone"])
            pn, pd = float(np.mean(noise ** 2)) + 1e-12, float(np.mean(drone ** 2)) + 1e-12
            for i, db in enumerate(man["levels_db"]):
                mix = noise + drone * np.sqrt(pn * 10 ** (db / 10) / pd)
                word = "same_level" if db == 0 else f"{abs(db)}dB_{'louder' if db > 0 else 'quieter'}"
                audio[f"{i + 1}_drone_{word}"] = emit(f"{uid}_{tag(db)}", mix, unit, has_drone_audio=True, sound_class="drone_buzz",
                                                      condition="drone_in_noise", drone_vs_noise_db=db, noise_kind=kind)
            if unit["drone_alone"] or unit["unit"] in man["preview_units"] + man["example_units"]:
                x = A.match_rms(drone, LOUDNESS_DBFS)
                tiles[f"{uid}_drone"] = A.spectrogram_image(x).convert("L")
                audio["7_drone_alone"] = x
            if unit["drone_alone"]:
                emit(f"{uid}_drone", drone, unit, has_drone_audio=True, sound_class="drone_buzz",
                     condition="drone_alone", drone_vs_noise_db=None, noise_kind=None)
        if unit["unit"] in man["example_units"]:
            for name, x in audio.items():
                A.write_wav(HERE / "examples" / f"unit_{unit['unit']:03d}_{kind}" / f"{name}.wav", x)

    lab = HERE / "dataset" / "valid" / "labels.jsonl"
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # preview: one row per unit, one column per condition
    cols = ["noise"] + [tag(db) for db in man["levels_db"]] + ["drone"]
    heads = ["noise alone"] + [f"drone {db:+d} dB" if db else "drone 0 dB" for db in man["levels_db"]] + ["drone alone"]
    T, pad, top, left = 168, 6, 22, 96
    sheet = Image.new("RGB", (left + len(cols) * (T + pad), top + len(man["preview_units"]) * (T + pad)), "white")
    d = ImageDraw.Draw(sheet)
    for j, h in enumerate(heads):
        d.text((left + j * (T + pad) + 4, 5), h, fill="black")
    for i, un in enumerate(man["preview_units"]):
        kind = next(x["noise_kind"] for x in man["units"] if x["unit"] == un)
        d.text((4, top + i * (T + pad) + 4), f"unit {un}\n{kind}", fill="black")
        for j, c in enumerate(cols):
            sheet.paste(tiles[f"nt_{un:03d}_{c}"].resize((T, T)), (left + j * (T + pad), top + i * (T + pad)))
    sheet.save(HERE / "preview.png", optimize=True)

    c = Counter(r["condition"] for r in rows)
    mb = sum(p.stat().st_size for p in spec_dir.glob("nt_*.png")) / 1e6
    print(f"rendered {len(rows)} clips ({dict(c)}), {mb:.1f} MB of spectrograms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mad-dir", type=Path, help="MAD_dataset folder (contains training.csv)")
    ap.add_argument("--esc50-dir", type=Path, help="ESC-50-master folder")
    ap.add_argument("--real-drone-dir", type=Path, help="5 s real drone clips, one folder per recording")
    ap.add_argument("--select", action="store_true", help="choose the sources again and rewrite manifest.json")
    ap.add_argument("--harmonize-dir", type=Path, help="[select] folder with generate_synthetic_dataset.py")
    ap.add_argument("--c5-dir", type=Path, help="[select] challenge folder (only to list sources kept for training)")
    ap.add_argument("--trained-on", type=Path, nargs="*", default=[],
                    help="[select] labels.jsonl of the datasets models were trained on; their sources are refused")
    ap.add_argument("--units-per-kind", type=int, default=15, help="[select] units with a drone per noise kind")
    ap.add_argument("--split-seed", type=int, default=0, help="[select] seed of the training sets' source split")
    ap.add_argument("--valid-frac", type=float, default=0.15, help="[select] held-out share of that split")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.select:
        select(args)
    render(args)


if __name__ == "__main__":
    main()
