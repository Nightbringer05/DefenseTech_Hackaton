# Challenge 4: Edge Triage Distillation (Cloud VLM -> Metis INT8)

**Presentation:** [Two-slide PDF](presentation/challenge-slides.pdf) · [Editable PowerPoint](presentation/challenge-slides.pptx). Includes a challenge overview and real validation examples from this data pack.

> **Target Platform**: **Axelera Embedded 110m** — M.2 2280 Key-M (PCIe Gen3 x4), 1x Metis AIPU (214 TOPS INT8, 3.5–9 W, **1 GB LPDDR4x** on-card DRAM).
> **Architecture**: Compact multi-head model (`MobileNetV3-Small` or `ResNet-18` at `224x224`). Full Metis placement is a deployment objective to verify with the SDK and hardware.

---

## 1. Mission & Problem Statement

Tactical edge nodes record tens of thousands of frames over contested SATCOM / mesh links where transmitting raw video is impossible. Cloud Vision-Language Models (VLMs) can accurately score frame utility (`triage_keep`, `priority_score`, sensor `modality`, `perspective`, and tactical `scene_tags`), **but cannot run on a 3.5 W tactical edge module**.

Your task is to **distill multi-head cloud VLM labels into a compact INT8 classifier/ranker targeting Metis** that filters low-value frames locally on the Embedded 110m and captures high-priority intelligence within strict **1% and 5% uplink bandwidth budgets**.

---

## 2. Dataset Specification

- **Dataset Scale**: **`282` deduplicated clips**, **`947` anonymized JPEG frames** primarily across `rgb` and `night` imagery — uniformly extracted through the identical `ProcessedFrame` pipeline with VLM clean-source, OCR/emblem/hash dedup filtering and cryptographically shuffled clip IDs (`c4_XXXX`). (`test/` labels are held out in public archives.)
- **6 Binary Scene Tags (`scene_vector`)**: `[armored_vehicle, vehicle, person, drone, explosion_smoke, trench_building]`.
- **Teacher Label Schema (`dataset/{train,valid}/labels.jsonl`)**:
  - `triage_keep` (`bool`)
  - `priority_score` (`float` in `[0.0, 1.0]`, strictly `<= 0.30` when `triage_keep=False` and `>= 0.70` for high-priority `keep` frames)
  - `modality` (`"rgb" | "thermal" | "night"`)
  - `perspective` (`"from_drone" | "ground" | "of_drone" | "other"`)
  - `scene_tags` & `scene_vector` (`len == 6`)

| Split | Frames | Clips | `keep: True` | High Priority (`keep & >=0.7`) | `rgb / night / thermal` |
|---|---:|---:|---:|---:|---|
| `train/` | 698 | 212 | 147 | 105 (`15.04%`) | `505 / 193 / 0` |
| `valid/` | 138 | 41 | 31 | 21 (`15.22%`) | `111 / 26 / 1` |
| `test/` | 111 | 29 | — (held-out) | — (held-out) | — (held-out) |

- **INT8 Calibration Set (`calibration/`)**: `200` stratified `224x224` RGB images sampled **strictly from `train/`**.

### Interactive Data Explorer (`viewer.py`)

Explore frames by `triage_keep`, continuous `priority_score`, **Top 1% / Top 5% SATCOM bandwidth budgets**, modality, perspective, and the 6 tactical scene tags offline:

```bash
python3 viewer.py
# Optional prediction overlay (computes Recall@1% and Recall@5% live on valid/):
python3 viewer.py --preds starter-kit/triage_preds.jsonl --port 8767
```

---

## 3. Baseline & Axelera Compilation

1. **Train & Export Multi-Head CNN (`224x224`, `opset=17`)**:
   ```bash
   cd starter-kit
   uv run --with onnx --with onnxscript --with onnxruntime --with scipy \
     python train_triage_distill_axelera.py \
       --dataset-dir ../dataset \
       --epochs 8 --calibration-dir ../calibration \
       --output-dir runs/c4_triage
   ```
   - Exported `triage_distill.onnx`: `3.58 MB` FP32; input `image [1, 3, 224, 224]` scaled `[0, 1]` (ImageNet normalization baked in-graph); named outputs `keep_logit [1]`, `priority_score [1]`, `modality_logits [1, 3]`, `perspective_logits [1, 4]`, `scene_logits [1, 6]`.
   - The ONNX graph has static shapes. Metis compilation, operator placement, INT8 accuracy, and device latency still require validation; see `starter-kit/README.md` for SDK guidance.
   - Checkpoint selection and ONNX prediction both rank by the priority head alone, round scores to six decimals, and break score ties pessimistically. The keep head is an auxiliary output, not a multiplier. The reported smoke scores below are historical results from the checkpoint selected before this alignment fix; fresh training may select a different epoch.

### Smoke-Test Reference (2-Epoch MobileNetV3-Small on `valid/` Static ONNX Runtime CPU)

| Metric | 2-Epoch Smoke Result | Notes |
|---|---:|---|
| Bandwidth Capture **Recall@1%** (`valid/`) | **0.0000** | `0/1` top-1% valid slot positive on 2-epoch smoke checkpoint |
| Bandwidth Capture **Recall@5%** (`valid/`) | **0.2857** | `2/7` top-5% valid slots contain `triage_keep & priority>=0.7` |
| Priority Rank Correlation (`priority_spearman`) | **-0.1573** | Substantial ranking & Focal/ListNet headroom (`2.00 ms` CPU ORT) |

---

## 4. Metrics and Proposed Judging Weights

The scorer returns individual metrics, not a combined leaderboard score. The percentages below are a proposed judging rubric. Organizers must publish the aggregation, normalization, tie-break rules, and hardware measurement protocol before using this dataset for a ranked event. Local scoring uses `valid/`; final scoring requires organizer-held `test/` labels.

Participants submit complete JSONL (`{"id": ..., "triage_keep": bool, "priority_score": float}`) scored on `valid/` locally by `starter-kit/evaluate_submissions.py triage` (with pessimistic tie-breaking and exact ID validation):
- **Normalized budget capture (`55%`)**: legacy output keys `recall_at_1pct` and `recall_at_5pct` compute `high / min(k, total_high)` within the top `k = max(1, round(N * budget))` scores (Python rounding). This is budget-normalized capture, not conventional recall `high / total_high`; the score is 0 when no high-priority frames exist.
- **Priority Ranking (`priority_spearman`) (`25%`)**
- **Metis INT8 Throughput / Power / Flash Footprint (`20%`)** (bonus for 100% AIPU-resident execution & temporal clip dedup!)

Teacher labels are clip-level VLM-derived proxies copied to frames, not independently reviewed frame-level utility judgments. The modality head has no thermal training examples and only one thermal validation frame, so this release cannot establish thermal triage generalization. The validation budgets contain just 1 and 7 frames; report these counts with scores.
