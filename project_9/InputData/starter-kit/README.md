# Axelera Embedded 110m Starter Kit — challenge-5-acoustic-visual-cuas

RF-DETR / CNN starter kit for **Axelera Embedded 110m** (M.2 2280 Key-M PCIe Gen3 x4, 1x Metis AIPU, 214 INT8 TOPS, 1 GB LPDDR4x DRAM, 3.5W-9W).

## Environment Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Use Python 3.11 or newer. MOT prediction also requires `ffmpeg` on PATH. Training downloads pretrained weights unless disabled where supported.

## Deployment status

The supplied reference runs on CPU ONNX Runtime. Metis compilation, INT8 accuracy, operator placement, memory, and power are not verified. RF-DETR backbone export does not include a runnable host decoder partition.

Follow the [Voyager SDK compiler guide](https://docs.axelera.ai/sdk/v1.8/reference/compiler/compiler-cli/) and check `axcompile --help` for your installed version. Its model input flag is `-i`. Use the train-only calibration folder and supply `--transform preprocess.py`, implementing the SDK's `get_preprocess_transform(image)` callback for your model. It must resize RGB, produce float32 CHW data, scale pixels to [0, 1], and apply ImageNet normalization only for RF-DETR; CNN graphs already include it. The callback is deployment integration work, not a supplied file.

`baseline_metrics.json` contains historical CPU smoke results, not scores guaranteed by a fresh training run or hardware benchmarks.

## Visual Data Explorer

From the challenge root directory (`..`), run the zero-dependency viewer:

```bash
python3 viewer.py
```

## Preprocessing & Export Rules

- `AcousticWakeCNN` bakes ImageNet normalization inside the exported `opset=17` ONNX graph, so inference and INT8 `../calibration_spectrograms/` PNGs expect raw `[0, 1]` scaled `224x224` RGB NCHW tensors.

The trainer embeds the validation-selected `wake_threshold` in ONNX. Prediction reads it automatically and reports the effective `wake_thresh`. Use `--wake-thresh` to override it; older exports without metadata require an explicit value (the historical smoke model used `--wake-thresh 0.55`). Tune on `valid/` only. `energy_savings_pct` is an optical inactivity proxy, not measured system energy.

## Quickstart

```bash
python train_audio_cnn_axelera.py --challenge-dir .. --epochs 12 \
  --calibration-dir ../calibration_spectrograms --output-dir runs/c5_audio
python predict_onnx_baseline.py acoustic --onnx runs/c5_audio/audio_wake_cnn.onnx \
  --challenge-dir .. --split valid --out ac_preds.jsonl
python evaluate_submissions.py acoustic --gt ../dataset/valid/labels.jsonl --preds ac_preds.jsonl
```

## Submission Format

- **Acoustic C-UAS (`ac_preds.jsonl`)**: One JSON line per ground-truth item: `{"id": str, "wake_triggered": bool, "sound_class": "drone_buzz"|"vehicle_mechanical"|"explosion"|"ambient_outdoor"}`.
