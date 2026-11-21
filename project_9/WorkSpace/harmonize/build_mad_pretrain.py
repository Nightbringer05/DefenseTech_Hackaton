"""Build the stage-1 (pre-training) dataset from MAD, in the exact Challenge-5 spectrogram format.

Two-stage plan:
  stage 1  pre-train the CNN on this dataset           (WorkSpace/datasets/mad_pretrain)
  stage 2  fine-tune the same CNN on the challenge data (InputData, untouched)

Separation rules:
  * Nothing from challenge valid/ or test/ is read. The only challenge input is C5 *train* audio,
    used to match the recording channel (low-pass cutoffs, background ambience, loudness).
    Pass --p-background 0 --p-lowpass 0 to use no challenge audio at all.
  * MAD gets its own train/valid split, grouped by YouTube video, so stage-1 model selection
    never looks at challenge data.

Per MAD clip: keep >= 5 s clips, cut centred non-overlapping 5.0 s windows, drop clipped audio
and spectral flatness < 1e-4 (C5 screening rule), harmonize the channel, then render the
224x224 PNG with c5_audio.spectrogram_image (reproduces the challenge PNGs).

Each row keeps the MAD class (`label`, 7 classes) and, when one exists, the matching challenge
class (`c5_class`: vehicle -> vehicle_mechanical, shelling -> explosion), so stage 1 can train
on either label set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import c5_audio as A

MAD_LABELS = ["communication", "gunshot", "footsteps", "shelling", "vehicle", "helicopter", "fighter"]
C5_MAP = {"vehicle": "vehicle_mechanical", "shelling": "explosion"}


def parse_args() -> argparse.Namespace:
    ws = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--c5-dir", type=Path, default=ws.parent / "InputData", help="only train/ is read")
    p.add_argument("--mad-dir", type=Path, default=ws / "mad-dataset" / "MAD_dataset")
    p.add_argument("--out-dir", type=Path, default=ws / "datasets" / "mad_pretrain")
    p.add_argument("--classes", nargs="+", default=MAD_LABELS, choices=MAD_LABELS)
    p.add_argument("--max-per-class", type=int, default=600)
    p.add_argument("--max-per-video", type=int, default=3, help="5 s windows per YouTube video and class")
    p.add_argument("--valid-frac", type=float, default=0.15, help="fraction of videos held out for stage-1 valid")
    p.add_argument("--p-lowpass", type=float, default=0.25)
    p.add_argument("--p-background", type=float, default=0.7)
    p.add_argument("--snr-db", type=float, nargs=2, default=(10.0, 30.0))
    p.add_argument("--save-audio", action="store_true", help="also write the harmonized 5 s WAVs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def video_id(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    return q.get("v", [url])[0]


def split_of(group: str, valid_frac: float, seed: int) -> str:
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if h < valid_frac else "train"


def c5_train_channel_stats(c5_dir: Path):
    rows = [json.loads(l) for l in (c5_dir / "dataset/train/labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ambient, rms, cuts = [], [], []
    for r in rows:
        assert r["split"] == "train"
        x = A.read_wav(c5_dir / r["audio"])[0][:A.CLIP_SAMPLES]
        rms.append(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
        if r["sound_class"] == "ambient_outdoor":
            ambient.append(x)
        d = np.diff(A.log_mel_db(x).mean(1))
        i = int(np.argmin(d))
        if d[i] < -6 and A.MEL_CENTERS_HZ[i] > 1500:
            cuts.append(float(A.MEL_CENTERS_HZ[i]))
    return ambient, rms, cuts or [4000.0, 6000.0, 7600.0], len(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)

    ambient, rms_pool, cutoffs, n_c5 = c5_train_channel_stats(args.c5_dir)
    print(f"channel stats from {n_c5} C5 train clips: {len(ambient)} ambient beds, "
          f"low-pass cutoffs {sorted(int(c) for c in cutoffs)}")

    clips = []
    for csv_name in ("training.csv", "test.csv"):
        with open(args.mad_dir / csv_name, encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                name = MAD_LABELS[int(r["label"])]
                if name in args.classes:
                    clips.append(dict(path=r["path"], label=name, group=video_id(r["youtube url"])))

    per_class, per_video, skips = Counter(), Counter(), Counter()
    rows = {"train": [], "valid": []}
    for i in rng.permutation(len(clips)):
        m = clips[i]
        if per_class[m["label"]] >= args.max_per_class:
            continue
        vkey = (m["group"], m["label"])
        room = args.max_per_video - per_video[vkey]
        if room <= 0:
            skips["video_cap"] += 1
            continue
        x, sr = A.read_wav(args.mad_dir / m["path"])
        if sr != A.SR:
            skips["sample_rate"] += 1
            continue
        if np.mean(np.abs(x) > 0.999) > 0.001:
            skips["clipped"] += 1
            continue
        wins = A.windows_5s(x, max_windows=room)
        if not wins:
            skips["shorter_than_5s"] += 1
            continue
        split = split_of(m["group"], args.valid_frac, args.seed)
        for k, w in enumerate(wins):
            if per_class[m["label"]] >= args.max_per_class:
                break
            if A.spectral_flatness(w) < 1e-4:
                skips["flatness"] += 1
                continue
            y, aug = w, []
            if rng.random() < args.p_lowpass:
                fc = float(rng.choice(cutoffs)); y = A.lowpass(y, fc); aug.append(f"lp{int(fc)}")
            if ambient and rng.random() < args.p_background:
                snr = float(rng.uniform(*args.snr_db))
                y = A.mix_background(y, ambient[rng.integers(len(ambient))], snr); aug.append(f"bg{snr:.0f}dB")
            y = A.match_rms(y, float(rng.choice(rms_pool)))

            cid = f"mad_{m['label']}_{Path(m['path']).parent.parent.name[:2]}{Path(m['path']).parent.name}_{Path(m['path']).stem}_w{k}"
            spec_rel = f"spectrograms/{split}/{cid}.png"
            (out / spec_rel).parent.mkdir(parents=True, exist_ok=True)
            A.spectrogram_image(y).save(out / spec_rel)
            row = {"id": cid, "split": split, "spectrogram": spec_rel,
                   "label": m["label"], "label_idx": MAD_LABELS.index(m["label"]),
                   "c5_class": C5_MAP.get(m["label"]), "has_drone_audio": False,
                   "source": "mad", "group": m["group"], "mad_path": m["path"], "harmonization": aug}
            if args.save_audio:
                row["audio"] = f"audio/{split}/{cid}.wav"
                A.write_wav(out / row["audio"], y)
            rows[split].append(row)
            per_class[m["label"]] += 1
            per_video[vkey] += 1

    for split, rs in rows.items():
        p = out / "dataset" / split / "labels.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
    tr_groups = {r["group"] for r in rows["train"]}
    va_groups = {r["group"] for r in rows["valid"]}
    summary = {
        "classes": args.classes,
        "counts": {s: dict(sorted(Counter(r["label"] for r in rs).items())) for s, rs in rows.items()},
        "videos": {"train": len(tr_groups), "valid": len(va_groups), "overlap": len(tr_groups & va_groups)},
        "challenge_data_used": "train/ only (channel stats)" if (args.p_background or args.p_lowpass) else "none",
        "skips": dict(skips), "args": {k: str(v) for k, v in vars(args).items()},
    }
    (out / "dataset" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
