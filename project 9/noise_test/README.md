# Noise test: drones in noise, and noise alone

A test set to compare wake models. Does the model hear a real drone through noise, and does it stay
quiet when there is only noise? 765 spectrograms in the challenge format, from real recordings.

| Clips | What |
|---|---|
| 180 | noise alone: 30 each of calm ambient, voice and radio, ground vehicles, helicopter / jet / saw / siren, explosions, gunfire |
| 540 | 90 of those noises with a real drone recording mixed in, at 6 levels: drone +10, +5, 0, -5, -10, -15 dB relative to the noise |
| 45 | drone alone |

The 7 clips of one unit share the same noise and the same drone; only the level changes. At +10 dB
the drone clearly dominates, at 0 dB both are equally strong, at -15 dB the noise carries about
30 times the power of the drone. `preview.png` shows four units, and `examples/` holds two units as
WAV files to listen to.

## Run it on your model

```bash
pip install numpy pillow onnxruntime
cd "project 9/noise_test"
python run_noise_test.py --model path/to/audio_wake_cnn.onnx
```

The model is an ONNX file as the starter trainer exports it (input `[1, 3, 224, 224]` in `[0, 1]`,
outputs `wake_logit` and `sound_class_logits`, wake threshold in the metadata). `--threshold 0.6`
overrides the threshold. For several models, repeat `--model name=file.onnx`.

Without an ONNX file, score a prediction file (one JSON per line with `id`, `wake_triggered`,
`sound_class`), for example the one the starter kit writes:

```bash
cd "project 9/starter-kit"
python predict_onnx_baseline.py acoustic --onnx MODEL.onnx --challenge-dir ../noise_test --split valid --out ac_preds_noise.jsonl
python ../noise_test/run_noise_test.py --predictions mine=ac_preds_noise.jsonl
```

## Reading the output

- **Drones caught**: share of drone clips that wake the camera, for each drone level.
- **False alarms**: share of noise-only clips that wake the camera, in total and per kind of noise.
- **Score** = mean of "drones caught" over the six levels, minus the false-alarm rate. 1 is perfect.
  A model that always wakes, never wakes, or wakes at random gets 0. The ranking is sorted by it.
- **At 5 % FA**: drones caught if the threshold were moved to give 5 % false alarms on this test. It
  compares models independently of the threshold they ship with.
- **4-class**: accuracy of the sound class, averaged over the four classes.

The ranking printed at the end includes the models stored in `leaderboard.json`.

## Current ranking

| | Model | Drones caught at +10 / 0 / -10 dB | False alarms | Score |
|---|---|---|---|---|
| 1 | `real_drone_detector`: starter model trained on real drone recordings over real noise | 92 % / 81 % / 56 % | 4 % | 0.65 |
| 2 | `real_drone_from_zero`: same data, weights started from zero instead of ImageNet | 92 % / 82 % / 33 % | 6 % | 0.57 |
| 3 | `challenge_head_on_real_drone`: backbone of model 1, head fitted on the 48 challenge clips | 94 % / 90 % / 71 % | 40 % | 0.41 |
| 4 | `three_view_mid`: challenge data only, frozen ImageNet backbone | 56 % / 33 % / 19 % | 19 % | 0.14 |
| 5 | `reference`: starter two-stage recipe, challenge data only | 81 % / 82 % / 84 % | 71 % | 0.12 |

The reference wakes on about 80 % of the clips at every level and on 71 % of the noise-only clips:
it does not react to the drone.

## How it was built

- **Drones**: 90 real clips from 43 recordings of DADS (Hugging Face
  `geronimobasso/drone-audio-detection-samples`).
- **Noise**: 180 five-second windows from 134 recordings of MAD (military audio) and ESC-50.
- **Never used for training**: every source is on the held-out side of the split our training sets
  were built with, and none is among the 2,561 sources those sets used. Gunfire is in none of them.
- **Spectrograms**: the organizers' recipe (`c5_audio.py`). The PNGs are stored as grayscale to save
  space; `.convert("RGB")`, which the starter kit already does, gives back the three channels.
- `manifest.json` lists every source and `build_noise_test.py` rebuilds the set from the audio.

## Limits

- "Never used for training" is true for our training sets. A model trained directly on MAD, ESC-50
  or DADS may have seen these recordings, and its score is then too high.
- Our own models chose their epoch and threshold on other mixes of the same held-out recordings,
  which favours them slightly.
- The drones are small multirotors. There is no fixed-wing or piston-engine drone.
- Here a drone clip always contains an audible real drone. The challenge's own "drone" label is
  different, so this ranking does not predict the challenge's valid score: the reference gets wake
  F1 0.80 there and ranks last here.
- Precision: about 5 points on the mean of drones caught and on the false-alarm rate, 10 points on
  a single level (90 clips), 15 points on one kind of noise (30 clips).
- Only the drone-to-noise ratio matters: the spectrogram recipe scales every clip to its loudest
  point, so the same clip played louder gives the same image.
- For the 4-class score, voice counts as `ambient_outdoor`, helicopter / jet / saw / siren as
  `vehicle_mechanical` and gunfire as `explosion`.
