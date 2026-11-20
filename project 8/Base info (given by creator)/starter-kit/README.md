# Axelera Embedded 110m Starter Kit — challenge-4-edge-triage

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

- `EdgeTriageNet` bakes ImageNet normalization inside the exported `opset=17` ONNX graph, so inference and INT8 `../calibration/` inputs expect raw `[0, 1]` scaled `224x224` RGB NCHW tensors.

Checkpoint selection ranks by the priority head alone, matching inference and the scorer's pessimistic ties. Legacy `recall_at_*` keys measure budget-normalized capture, not conventional recall; see `../CHALLENGE.md`.

## Quickstart

```bash
python train_triage_distill_axelera.py --dataset-dir ../dataset --epochs 8 \
  --calibration-dir ../calibration --output-dir runs/c4_triage
python predict_onnx_baseline.py triage --onnx runs/c4_triage/triage_distill.onnx \
  --dataset-dir ../dataset --split valid --out triage_preds.jsonl
python evaluate_submissions.py triage --gt ../dataset/valid/labels.jsonl --preds triage_preds.jsonl
```

## Submission Format

- **Edge Triage (`triage_preds.jsonl`)**: One JSON line per ground-truth item: `{"id": str, "triage_keep": bool, "priority_score": float}`. All IDs must match cleanly with finite scores; ties break pessimistically.
