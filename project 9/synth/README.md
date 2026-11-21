# Drone Sound Toolkit

Builds Challenge 5 training sets as **FLAC**, in the **challenge's own folder layout**, from:

- **MAD_dataset/**: real military recordings (7 labels) mapped onto the 4 challenge classes
- **synthetic drones**: synthesized multirotor audio flown past the mic and mixed over real
  non-drone recordings (MAD has no drones)

Spectrogram PNGs match the provided ones, so the starter-kit trainer reads the result unchanged.

## Run

```bash
cd synth
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
python server.py                          # http://127.0.0.1:8770
python server.py --host 0.0.0.0           # reachable from your LAN
```

In the page: pick a preset, adjust the MAD → class mapping and drone ranges, check the
**Dataset contents** table (live counts per split and class), preview single MAD or synthetic
clips, click **Generate dataset**, then **Download dataset (.zip)** or **Audio + labels only (.zip)**.
The default (all mapped MAD clips + 2000 synthetic drones + 600 synthetic explosions/salvos, ~9.1k clips) renders in about
45 s on 11 cores and is ~870 MB of FLAC.

`.venv/` and `out/` are gitignored.

### CLI

```bash
python generate.py                                   # defaults: MAD full + 2000 drones + 600 explosions/salvos, FLAC
python generate.py --name small --mad-per-class 100 --drone 300
python generate.py --name x2 --windows 2 --drone 4000  # two 5 s windows from long MAD clips
python generate.py --name synth_only --no-mad --drone 800 --vehicle 600 --explosion 300 --ambient 300
python generate.py --print-config > my.json           # edit mapping/ranges, then --config my.json
```

### Train on it

```bash
cd starter-kit
python train_audio_cnn_axelera.py --challenge-dir ../synth/out/mad_full --epochs 12 \
  --calibration-dir ../calibration_spectrograms --output-dir runs/mad_full
```

## Drone / no-drone set from the browser

`datasets/drone_vs_nodrone_200/` (committed, 31 MB): 100 synthetic drones over MAD noise + 100
MAD noise clips (shooting, shelling, vehicle, fighter, footsteps; no helicopters, no speech),
plus the 8 real challenge valid clips. Each split has `labels.jsonl` (challenge schema, read by
the trainer), `labels.csv` (same columns) and `metadata.jsonl` (all generator parameters).

To make it (or a variant) in the page:

1. `python server.py` and open http://127.0.0.1:8770
2. Click the **Drone vs no-drone 200** preset (loaded from `presets/drone_vs_nodrone_200.json`),
   or use **Drone / no-drone set**: enter total clips and drone share, click **Apply**. It sets
   the drone count and MAD clips per label, leaves out helicopters and speech, and uses MAD-only
   backgrounds.
3. Adjust drone types, payload, wind, SNR and so on, check **Dataset contents**, click
   **Generate dataset**, then **Download dataset (.zip)**.

Each dataset has a free-text **comment** (Comment box in the page, `--comment` on the CLI,
`comment` in a preset). It is saved in `manifest.json` and in a generated `README.md` inside the
dataset folder (so it travels with the zip), and shown in the dataset list.

The browser run and `python generate.py --config presets/drone_vs_nodrone_200.json` produce
byte-identical files. Any `presets/*.json` (with optional `_label` / `_description`) appears as a
preset button.

## Sample dataset (committed)

`sample/sample_20mb/` (19.5 MB, FLAC) is a small ready-made dataset: 130 train clips (60 MAD noise
without helicopters or speech, 50 synthetic drones, 20 synthetic explosions/salvos), all
backgrounds from MAD, plus the 8 real valid clips. Regenerate it with:

```bash
echo '{"mad":{"test_split":false,"mapping":{"0":"skip","5":"skip"}}}' > /tmp/s.json
python generate.py --name sample_20mb --mad-per-class 12 --drone 50 --explosion 20 \
  --no-real --out sample --config /tmp/s.json
```

Full-size datasets go to `out/`, which is gitignored.

## MAD → challenge classes

MAD ships `training.csv` / `test.csv` (`path,label,youtube title,youtube url`) and
`training/<video>/<n>.wav`, `test/<video>/<n>.wav`: 16 kHz mono, 2–10 s. Label ids follow the
MAD paper; title keywords in `training.csv` agree (e.g. label 6 titles: 1329 jet/fighter hits).

| MAD id | MAD label | train / test clips | default challenge class |
|---|---|---|---|
| 0 | communication | 774 / 207 | skip (speech; the challenge screened it out) |
| 1 | shooting | 1293 / 280 | explosion |
| 2 | footsteps | 773 / 104 | ambient_outdoor |
| 3 | shelling | 883 / 104 | explosion |
| 4 | vehicle | 910 / 122 | vehicle_mechanical |
| 5 | helicopter | 934 / 91 | vehicle_mechanical (rotor harmonics: hard negative) |
| 6 | fighter | 862 / 129 | vehicle_mechanical |

Every mapping is editable in the page or via `mad.mapping` in the config.

- **5 s clips**: long recordings give `windows_per_clip` evenly spread random 5 s windows;
  shorter ones are either padded (placed at a random offset over a real challenge
  `ambient_outdoor` clip ~20 dB down, with fades) or skipped.
- **Splits**: MAD `training` → `train/`; MAD `test` → `mad_test/` (held out, never used as
  background); real challenge `valid/` → `valid/` (threshold selection); the 48 real challenge
  train clips are added to `train/` (losslessly converted to FLAC).

## Synthetic drones

N rotors × blade-pass harmonics (BPF = blades × RPM / 60) with shared + per-rotor RPM wander,
motor whine, blade-rate AM and hiss → flyby / hover / approach trajectory with retarded-time
Doppler, 1/r and air absorption (`numpy` engine, or `acoular` `MovingPointSource`) → mixed over a
real background (see **SNR** below).

Two drone models: `drone.model = "simple"` (default; the RPM, spread and jitter ranges in the
page) or `"physics"` (airframe types with mass/payload → rotor RPM, `dronesynth/drones.py`; used
by the `drone_vs_nodrone_200` preset).

### SNR

RMS over the whole 5 s clip, drone vs. background, both normalised first:

```
snr = U(snr_db_at_30m)                                    # default 5 … 25 dB
    − distance_db_per_decade · log10(max(r_min, 30) / 30)  # default 10: 100 m −5 dB, 300 m −10 dB
    + physics_level_weight · level_db                      # physics model only, default 0.5
snr = clip(snr, snr_clip_db)                               # default −5 … 25 dB
```

Free-field spreading would be 20 dB per decade, but the MAD backgrounds are recorded at an
unknown distance and gain, ground reflection adds energy, and narrowband rotor tones stay audible
below broadband noise, so the full 20 dB made most distant drones inaudible. Before this change
(20 dB/decade, full loudness term, −12 … 30 dB) the 200 set had a median of 9.5 dB and 14 % of
drones below 0 dB; now the median is 14 dB and 2 % are below 0 dB.

Backgrounds come only from MAD training recordings; there is no procedural noise and no
challenge audio in the mix. Drones go over any mapped MAD clip; synthetic vehicle / explosion /
ambient clips and the padding of short MAD clips use MAD clips mapped to `ambient_outdoor`
(footsteps), so their labels stay clean.

## Synthetic salvos

Synthetic `explosion` clips (600 by default) are 60% salvos, 40% single blasts. Salvo kinds,
weighted by `explosion.salvo_types`:

- **burst**: automatic fire, 500–950 rounds/min, 3–30 rounds, sometimes a second burst
- **volley**: 3–11 rifles firing within ~40 ms, 1–3 volleys
- **artillery**: 2–6 guns within 3 s, heavy low-frequency booms, 300–4000 m

Each shot has a muzzle blast, a supersonic crack (N-wave) for half the small-arms shots,
distance low-pass and level, and 0–2 terrain echoes. `params` records `salvo`, `shots`,
`salvo_range_m` and `rate_rpm`. MAD `shooting` clips already contain real bursts; the synthetic
ones add controlled rate/range variety.

## Output

```
out/<name>/
  audio/{train,valid,mad_test}/<id>.flac       16 kHz mono 16-bit, exactly 5 s (challenge originals kept as-is)
  spectrograms/{train,valid,mad_test}/<id>.png 224x224 log-mel, challenge style
  dataset/{train,valid,mad_test}/labels.jsonl
  manifest.json                                config, per-split class counts, timing
```

Each row has the challenge fields plus `source` (`mad` / `synthetic` / `challenge`) and
`params`: for MAD the source file, MAD label, YouTube URL, window offset and padding; for drones
rotors, blades, RPM, BPF, flight mode, altitude, min range, SNR, background and seed. IDs:
`mad_<video>_<n>_w<k>`, `madtest_…`, `syn_000001`.

Spectrograms are made by Adrien's `starter-kit/make_spectrogram.py`, called directly from
`dronesynth/spectro.py` (which falls back to an identical port if that file is missing): first 80 000 samples, n_fft 1024, hop 256, 128 mel bands,
50–8000 Hz, power_to_db(ref=max, top_db=80), flipped, bilinear 224×224, RGB. Identical to the
provided challenge PNGs (max difference 1 grey level on one clip). Needs `librosa`.

### Downloads

```
GET /download/<name>.zip              everything
GET /download/<name>.zip?parts=audio  FLAC + labels + manifest (no PNGs)
GET /files/<name>/audio/train/<id>.flac
```

Zips are cached as `out/.<name>.<parts>.zip` and rebuilt when the dataset changes.

## Caveats

- MAD audio comes from YouTube: expect music, narration and compression artefacts that the
  challenge clips were screened for.
- Many MAD videos contain several labels and long clips share a source video, so per-clip
  splits within MAD `training` would leak; use `mad_test/` or the real `valid/` for evaluation.
- Synthetic drones are a model, not recordings. Check results on real `valid/`.

## Files

```
synth/
  server.py               local website + JSON API (stdlib http.server)
  generate.py             CLI
  dronesynth/             dataset.py (MAD + mixing + splits), sources.py, propagate.py, spectro.py
  web/index.html          the page (works with or without the server)
  web/clips/              demo FLACs from the Acoular flyby
  examples/acoular_flyby.py
```
