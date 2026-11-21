#!/usr/bin/env python3
"""Generate a synthetic dataset from the command line.

    python generate.py                                   # all MAD (mapped) + 2000 synthetic drones, FLAC
    python generate.py --name small --mad-per-class 100 --drone 300
    python generate.py --name synth_only --no-mad --drone 1000 --vehicle 500 --explosion 250 --ambient 250
    python generate.py --name drones_wav --format wav --engine acoular
    python generate.py --print-config > my.json      # dump defaults, edit, pass back with --config

Then train:
    cd ../starter-kit && python train_audio_cnn_axelera.py --challenge-dir ../synth/out/<name> ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from dronesynth.dataset import DEFAULT_CONFIG, build, merge_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="JSON config (merged over defaults)")
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--engine", choices=("numpy", "acoular"))
    ap.add_argument("--format", dest="audio_format", choices=("flac", "wav"), help="audio file format (default flac)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--drone", type=int, help="number of drone_buzz clips")
    ap.add_argument("--vehicle", type=int, help="number of vehicle_mechanical clips")
    ap.add_argument("--explosion", type=int, help="number of explosion clips")
    ap.add_argument("--ambient", type=int, help="number of ambient_outdoor clips")
    ap.add_argument("--mad-per-class", type=int, help="cap MAD training clips per MAD label (0 = all)")
    ap.add_argument("--windows", type=int, help="5 s windows cut from each long MAD clip")
    ap.add_argument("--no-mad", action="store_true", help="synthetic only, don't use MAD_dataset")
    ap.add_argument("--no-real", action="store_true", help="don't copy the real train clips in")
    ap.add_argument("--out", type=Path, default=BASE / "out")
    ap.add_argument("--print-config", action="store_true")
    a = ap.parse_args()

    user = json.loads(a.config.read_text()) if a.config else {}
    cfg = merge_config(user)
    for key in ("name", "seed", "engine", "workers", "audio_format"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    for flag, cls in (("drone", "drone_buzz"), ("vehicle", "vehicle_mechanical"),
                      ("explosion", "explosion"), ("ambient", "ambient_outdoor")):
        if getattr(a, flag) is not None:
            cfg["counts"][cls] = getattr(a, flag)
    if a.mad_per_class is not None:
        cfg["mad"]["max_per_label"] = a.mad_per_class
    if a.no_mad:
        cfg["mad"]["enabled"] = False
    if a.windows is not None:
        cfg["mad"]["windows_per_clip"] = a.windows
    if a.no_real:
        cfg["include_real"] = False
    if a.print_config:
        print(json.dumps(cfg if user else DEFAULT_CONFIG, indent=2))
        return

    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 0.5 or done == total:
            last[0] = time.time()
            print(f"\r  {done}/{total} clips", end="", flush=True)

    print(f"Generating '{cfg['name']}' with engine={cfg['engine']}, workers={cfg['workers']}")
    m = build(cfg, a.out, BASE, progress)
    print(f"\nDone in {m['seconds']} s -> {a.out / m['name']}")
    for split, info in m["splits"].items():
        print(f"  {split:9s} {info['total']:6d}  {info['classes']}")
    print(f"  sources: MAD {m['mad_clips']} | synthetic {m['synthetic_clips']} | challenge train {m['real_train_clips']}")


if __name__ == "__main__":
    main()
