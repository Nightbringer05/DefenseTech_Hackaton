# Challenge 5: Acoustic Cue-to-Optical Slew C-UAS

**Presentation:** [Two-slide PDF](presentation/challenge-slides.pdf) · [Editable PowerPoint](presentation/challenge-slides.pptx). Includes a challenge overview and real validation examples from this data pack.

> **Target Platform**: **Axelera Embedded 110m** — M.2 2280 Key-M (PCIe Gen3 x4), 1x Metis AIPU (214 TOPS INT8, 3.5–9 W, **1 GB LPDDR4x**).
> **Architecture**: Continuous low-duty acoustic CNN/CRNN (`224x224` 2D log-mel spectrogram -> Metis INT8) gating expensive optical PTZ / RF-DETR wakeups, optionally fused with visual confirmation on paired video.

---

## 1. Mission & Problem Statement

Keeping a gimbaled EO/IR camera and full visual object detector powered continuously drains batteries and advertises RF/thermal signatures. Yet small UAV rotors emit distinct narrowband harmonic signatures when a drone is not visibly identified (`9` `train` clips are tagged audio-positive without a visual drone detection). These clip-level tags do not establish time-to-arrival or wake lead time.

Build a lightweight **acoustic wake & classification model** running on Metis INT8 that detects `drone_buzz` from 5-second field-microphone segments, separates true rotor harmonics from `vehicle_mechanical`, `explosion`, and `ambient_outdoor`, and optionally fuses cross-modal optical verification when `has_drone_audio` fires.

---

## 2. Dataset Specification

- **Synchronized Multimodal Triples (`62` 5-second clips, hash-permuted `c5_XXXX` IDs)**:
  - **Audio (`audio/<split>/*.wav`)**: Metadata-stripped `5.0s` 16 kHz mono 16-bit PCM WAV (screened using PANNs AudioSet music/speech thresholds, which do not guarantee absence, plus spectral flatness >= `0.0001`).
  - **Video (`video/<split>/*.mp4`)**: Metadata-stripped synchronized `5.0s` `30 fps` CFR H.264 visual stream (overlay-filtered with sampled visual audits; `test/` ground truth held out in public archives).
  - **Precomputed 2D Log-Mel Spectrograms (`spectrograms/<split>/*.png`)**: Ready-to-train `224x224` PNG spectrograms for standard Metis ConvNet pipelines.
- **4 Sound Classes**: `drone_buzz`, `vehicle_mechanical`, `explosion`, `ambient_outdoor` (every class appears in both `train/` and `valid/`).

| Split | Samples | `drone_buzz` (`wake=True`) | `vehicle_mechanical` | `explosion` | `ambient_outdoor` | Cross-modal `(audio+, vis-)` |
|---|---:|---:|---:|---:|---:|---:|
| `train/` | 48 | 15 | 18 | 8 | 7 | 9 |
| `valid/` | 8 | 2 | 4 | 1 | 1 | 0 |
| `test/` | 6 | — (held-out) | — (held-out) | — (held-out) | — (held-out) | — (held-out) |

- **INT8 Calibration Set (`calibration_spectrograms/`)**: `48` `224x224` spectrograms sampled **strictly from `train/`**.

### Interactive Data Explorer (`viewer.py`)

Play synchronized 5-second `16 kHz` `.wav` audio, paired `30 fps` optical `.mp4` video, and live-playhead `224x224` log-mel spectrograms offline, with False Positive / False Negative / Wake F1 inspection:

```bash
python3 viewer.py
# Optional prediction overlay (shows Wake F1, duty cycle, and estimated optical inactivity):
python3 viewer.py --preds starter-kit/ac_preds.jsonl --port 8768
```

---

## 3. Baseline & Axelera Compilation

1. **Train & Export 2D Spectrogram Wake CNN (`224x224`, `opset=17`)**:
   ```bash
   cd starter-kit
   uv run --with onnx --with onnxscript --with onnxruntime --with scipy \
     python train_audio_cnn_axelera.py \
       --challenge-dir .. \
       --epochs 12 --calibration-dir ../calibration_spectrograms \
       --output-dir runs/c5_audio
   ```
   - Exported `audio_wake_cnn.onnx`: `3.56 MB` FP32; input `spectrogram [1, 3, 224, 224]` scaled `[0, 1]` (ImageNet normalization baked in-graph); outputs `wake_logit [1]` and `sound_class_logits [1, 4]`.

The trainer stores the validation-selected `wake_threshold` inside the ONNX model. `predict_onnx_baseline.py acoustic` reads it automatically and reports the effective `wake_thresh`. `--wake-thresh` explicitly overrides it. Older exports without metadata require an explicit threshold (the historical smoke model used `--wake-thresh 0.55`). Select thresholds on `valid/`, never on `test/`. Compilation and on-device measurements remain a separate exercise; see `starter-kit/README.md`.

### Smoke-Test Reference (2-Epoch MobileNetV3-Small on `valid/` Static ONNX Runtime CPU)

| Metric | 2-Epoch Smoke Result | Notes |
|---|---:|---|
| Wake `F1` (`valid/`, `@thr=0.55`) | **0.8000** | `TP=2, FP=1, FN=0` |
| 4-Class `sound_class_accuracy` (`valid/`) | **0.5000** | `2.14 ms` per 5s window on CPU ORT |
| Optical Duty Cycle / Inactive-Window Estimate | `37.50%` / `62.50%` | Fraction of windows with / without a wake; not measured power savings |

### Ideas to Win
- **Wake Loss + Threshold Calibration**: Compare false positives and missed wakes. With only 8 validation clips and 2 positives, repeated threshold tuning can overfit; treat these scores as smoke checks, not robust field estimates.
- **Streaming Extension**: Experiment with shorter audio windows, but collect event-onset labels and measure end-to-end delay separately. The supplied clip labels and scorer cannot validate sub-second wake latency.
- **Two-Stage Cascade**: Wake via a compiled acoustic model, then verify/localize on `video/<split>/*.mp4` using Challenge 1 `RFDETRNano`!

---

## 4. Metrics and Proposed Judging Weights

The scorer returns individual metrics, not a combined leaderboard score. The percentages below are a proposed judging rubric. Organizers must publish the aggregation, normalization, tie-break rules, and hardware measurement protocol before using this dataset for a ranked event. Local scoring uses `valid/`; final scoring requires organizer-held `test/` labels.

Evaluated on `valid/` locally via `starter-kit/evaluate_submissions.py acoustic`:
- **Acoustic Wake `F1` (`40%`)**
- **4-Class Sound Classification Accuracy (`25%`)**
- **Optical inactivity proxy (`20%`)**: legacy key `energy_savings_pct = (1 - optical_duty_cycle) * 100`. This assumes equal-length windows and ignores acoustic compute, camera startup, and hold time; it is not measured system energy savings. The scorer does not enforce a recall floor, so organizers must define one before awarding this component.
- **Metis Wake Latency & Footprint (`15%`)**

The acoustic head uses whole 5-second windows. CPU ORT timing measures model inference only, excluding audio capture, spectrogram construction, camera wake, and optical confirmation. The released labels do not include bearing, slew commands, event onset, or visual boxes; optical localization is an optional extension, not a scored output. C5 is especially small (48 train / 8 valid / 6 test), so a robust acoustic benchmark needs more independent recordings.
