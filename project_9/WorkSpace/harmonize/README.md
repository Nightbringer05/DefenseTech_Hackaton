# C5 + MAD dataset harmonization

Goal: train on the challenge data (C5, 62 clips) and the MAD Military Audio Dataset
(Kaggle `junewookim/mad-dataset-military-audio-dataset`, 7 466 clips) without the model
learning "which dataset is this" instead of "what sound is this".

## Quick start: generate the synthetic training set

```bash
git pull
cd project_9/WorkSpace/harmonize
pip install numpy pillow          # ffmpeg optional (only for MP3/FLAC drone files)
python generate_synthetic_dataset.py --download \
    --real-drone-dir /path/to/your/drone/recordings \
    --extra-bg /path/to/your/ambience=ambient
```

- `--download` fetches MAD (~1.1 GB, Kaggle) and ESC-50 (~600 MB, GitHub) once, into `WorkSpace/`.
- `--real-drone-dir`: any length or format. Short clips are looped to 5 s. Repeatable.
  If each sub-folder is one recording session, add `--real-drone-group dir` so a session never
  ends up in both train and valid.
- `--extra-bg DIR=CATEGORY`: your own backgrounds. CATEGORY is one of `ambient`, `vehicle`,
  `tonal_neg`, `explosion`, `speech`. Repeatable.
- Default size: 1,500 train + 250 valid per class (7,000 images, ~290 MB). About 1 min on 10 cores.

Output: `WorkSpace/datasets/synth_c5/`, laid out like `InputData/`. Train on it with the starter kit:

```bash
cd ../../InputData/starter-kit
uv run --with onnx --with onnxscript --with onnxruntime --with scipy python train_audio_cnn_axelera.py \
    --challenge-dir ../../WorkSpace/datasets/synth_c5 --calibration-dir ../calibration_spectrograms \
    --epochs 12 --output-dir runs/synth
```

**Before training, check the two files the script writes:**
- `audit.json` (also printed at the end): `P(drone)` must be ~0.25 for every background
  category and the same with or without distractor. The script prints `OK` or `WARNING`.
- `preview.png`: synthetic drones next to the challenge's real drones. They should look alike
  (mostly faint lines in noise), not bright clean ladders.

### Why this dataset has no shortcuts

| Rule | How |
|---|---|
| The drone is the only difference between classes | Each drone clip is a copy of a negative clip from the same background category, with a drone added; every category gets exactly P(drone) = 0.25 |
| A tonal line alone is not "drone" | Non-drone tones (sine, mains hum, low engine, helicopter thump, siren, whine) added to 25 % of *all* clips, at the same loudness range as drones |
| Real-world difficulty | Drone SNR −10…+10 dB, 30 % only partly present, fly-by Doppler, distance low-pass, small RPM drift, rotor noise |
| Same channel as the challenge | Challenge-train low-pass cutoffs and loudness, exact challenge spectrogram recipe |
| No leakage | Train/valid split by source recording before mixing; challenge valid/test never read |

## Two-stage training plan

| Stage | Data | Folder | Used for model selection |
|---|---|---|---|
| 1. Pre-train | MAD (+ drone and ambient sources later), converted to the C5 spectrogram format | `WorkSpace/datasets/mad_pretrain/` (generated, not in git) | MAD's own `valid/` (split by YouTube video) |
| 2. Fine-tune | Challenge data, untouched | `InputData/` | Challenge `valid/` (threshold only) |
| Final check | Challenge `test/` | `InputData/` | Never; evaluation only |

The two datasets are never mixed. **Challenge `valid/` and `test/` are not read by any script
here.** The only challenge input to stage 1 is `train/` audio, used to match the recording
channel. Pass `--p-background 0 --p-lowpass 0` for a stage-1 set with zero challenge audio.

## Recreate the data (every teammate, after cloning)

```bash
cd project_9/WorkSpace
curl -L -o mad-dataset-military-audio-dataset.zip https://www.kaggle.com/api/v1/datasets/download/junewookim/mad-dataset-military-audio-dataset
python -c "import zipfile; zipfile.ZipFile('mad-dataset-military-audio-dataset.zip').extractall('mad-dataset')"
cd harmonize
python build_mad_pretrain.py            # all 7 MAD classes, up to 600 windows each
```

Needs only `numpy` and `Pillow`. Output: `WorkSpace/datasets/mad_pretrain/{dataset,spectrograms}/{train,valid}`.
Each row has `label` (7 MAD classes), `c5_class` (challenge name when one exists, else null),
`group` (YouTube video id) and `harmonization` (what was applied).

## What was compared

| Property | C5 (challenge) | MAD (raw) | Verdict |
|---|---|---|---|
| Container | 16 kHz mono 16-bit PCM WAV | 16 kHz mono 16-bit PCM WAV | ✅ identical |
| Duration | exactly 5.0 s | 1–10 s (only 55 % ≥ 5 s; shelling 29 %) | ❌ cut to 5 s windows, drop < 5 s (no zero-padding: it would draw a black band in the PNG) |
| Loudness | median −20.7 dBFS RMS | median −23.5 dBFS RMS | ✅ close; PNGs are per-image min-max normalized anyway; RMS matched |
| Clipping | none | a few clips up to 16 % clipped | ⚠️ drop clips with > 0.1 % clipped samples |
| Spectral tilt (low vs high mel bands) | median 38.6 dB | median 27.6 dB | ⚠️ depends on class (vehicle −2.3 dB/oct, explosion +1.6 dB/oct); **no global EQ applied** |
| Band-limiting | 14/62 clips low-passed at 1.9–7.6 kHz | rare | ⚠️ random low-pass at C5's own cutoffs (p = 0.25) |
| Floor / "speckle holes" | 1–15 % of mel bins at the −80 dB floor | ≈ 0 % | ⚠️ partly fixed by mixing C5 ambient background in. AAC re-encoding tested and rejected (see below) |
| Labels | drone_buzz, vehicle_mechanical, explosion, ambient_outdoor | communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter | ❌ partial overlap only, **no drone class** |
| Speech / music | screened out with PANNs | `communication` = radio speech; YouTube audio may contain narration | ❌ drop `communication` |
| Leakage | hash-permuted IDs | consecutive segments from the same YouTube video; 1 URL appears in both MAD train and test | ⚠️ cap 2 windows per video, keep `group` = URL |

### Spectrogram recipe (recovered by reverse-engineering, not shipped by organizers)

`c5_audio.spectrogram_image()` reproduces all 62 challenge PNGs with mean |error| 0.84/255
(median 0.55). External audio **must** go through this function. A different librosa default
(e.g. 64 mels, fmin = 0) would make a visibly different image, and the CNN would learn to
spot the dataset.

`n_fft=1024, hop=256, Hann, center=True · 128 Slaney mels, 50–8000 Hz · power_to_db(ref=max, top_db=80) · min-max → uint8 · low freq at bottom · bilinear resize to 224×224 · gray → RGB`

### Label mapping

Stage 1 keeps all 7 MAD classes: more classes means richer pre-trained features, and the
classifier head is replaced in stage 2. `c5_class` records the direct match where one exists.
Use `--classes` to restrict the set.

| MAD label | `c5_class` | Notes |
|---|---|---|
| 4 vehicle (tanks, APCs) | vehicle_mechanical | Direct match |
| 3 shelling (tank / howitzer fire) | explosion | Direct match |
| 5 helicopter | — | Rotor harmonics but not a drone: a useful hard negative for the wake head |
| 6 fighter | — | Engine tonal noise |
| 1 gunshot | — | Impulsive; not a C5 class |
| 2 footsteps | — | Close-mic body noise, not outdoor ambience |
| 0 communication | — | Radio speech. C5 screened speech out; fine for pre-training, drop with `--classes` if unwanted |

### Measured domain gap (does harmonization help?)

A linear classifier on log-mel band statistics tries to tell C5 from MAD within the *same*
class. The score is balanced accuracy (0.5 = indistinguishable). Small n (9 C5 explosions,
22 vehicles), so ±0.05 is noise.

| Pair | Raw MAD | Low-pass + C5 background (adopted) |
|---|---|---|
| vehicle_mechanical ↔ vehicle | 0.81 | 0.74–0.75 |
| explosion ↔ shelling | 0.75 | 0.70–0.77 |
| drone_buzz ↔ helicopter | 0.90 | 0.84–0.87 |

Rejected: a global −3 dB/oct tilt (made explosions *more* separable, 0.81), and AAC
re-encoding at 24–64 kbps (no holes at 32–64 kbps; at 24 kbps the holes appear but
separability rises to 0.87–0.94).

The rest of the gap (~0.75) is **content**, not channel: MAD is YouTube footage of tanks and
howitzers, while C5 is field-microphone recordings. This is acceptable as long as model
fine-tuning (stage 2) happens on the challenge data itself.

## Remaining limitations / next steps

1. **No drone audio in MAD.** Wake F1 is 40 % of the score, and MAD adds zero positives. It
   only helps the negative classes and the 4-class head. A drone-specific source is still
   needed, for example open drone-audio datasets on GitHub/Kaggle (DroneAudioDataset, Al-Emadi
   et al., is 16 kHz but uses 1 s clips, so tile or concatenate to 5 s). Run it through the same
   `c5_audio` harmonization.
2. **No ambient_outdoor in MAD.** Candidates: wind / rain / birds / insects from ESC-50 or
   FSD50K, resampled to 16 kHz.
3. **Training code.** The starter trainer only knows the 4 challenge classes and reads one
   dataset. Stage 1 needs a trainer that reads `mad_pretrain` with the 7 MAD labels (plus
   drone later). Stage 2 loads those backbone weights and fine-tunes on `InputData` with the
   original wake + 4-class heads.
4. **INT8 calibration.** Keep the challenge `calibration_spectrograms/` (train-only rule).
5. **8 valid clips (2 positives).** Comparing "with vs without MAD pre-training" on challenge
   valid is very noisy. Use several seeds, and look at the final `test/` result only once.
