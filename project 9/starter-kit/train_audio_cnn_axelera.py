"""Train a lightweight 2D Log-Mel Spectrogram CNN for Challenge 5 and export static ONNX.

Two-stage option:
  Stage 1 (--wake-only): backbone + wake head only, class loss disabled.
  Stage 2 (--init-checkpoint <stage1 .pt>): reload stage-1 weights, train wake + 4-class heads,
          backbone LR scaled by --backbone-lr-mult.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SOUND_CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")
_CLASS_INDEX = {name: i for i, name in enumerate(SOUND_CLASSES)}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpectrogramDataset(Dataset):
    """Challenge 5 paired spectrogram dataset loaded from dataset/<split>/labels.jsonl."""

    def __init__(self, challenge_dir: Path, split: str, img_size: int = 224) -> None:
        self.challenge_dir = challenge_dir
        rows_path = challenge_dir / "dataset" / split / "labels.jsonl"
        if not rows_path.exists():
            raise FileNotFoundError(f"Missing labels file: {rows_path}")
        self.rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"Empty dataset split {split!r} in {rows_path}")
        # Normalization lives inside the exported ONNX graph so raw calibration PNGs align.
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        with Image.open(self.challenge_dir / row["spectrogram"]) as im:
            x = self.transform(im.convert("RGB"))
        cls_idx = _CLASS_INDEX[str(row["sound_class"])]
        return {
            "spectrogram": x,
            "has_drone": torch.tensor(
                float(row["has_drone_audio"]), dtype=torch.float32
            ),
            "sound_class": torch.tensor(cls_idx, dtype=torch.long),
        }


class AcousticWakeCNN(nn.Module):
    """2D ConvNet over log-mel spectrogram PNGs for continuous <2ms Metis acoustic wake."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self, arch: str = "mobilenet_v3_small", *, pretrained: bool = True
    ) -> None:
        super().__init__()
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = int(backbone.fc.in_features)
            backbone.fc = nn.Identity()  # type: ignore[assignment]
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            first_linear = backbone.classifier[0]
            assert isinstance(first_linear, nn.Linear)
            feat_dim = int(first_linear.in_features)
            backbone.classifier = nn.Identity()  # type: ignore[assignment]

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.backbone = backbone
        self.head_wake = nn.Linear(feat_dim, 1)
        self.head_class = nn.Linear(feat_dim, len(SOUND_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_x = (x - self.mean) / self.std
        feat = self.backbone(norm_x)
        return self.head_wake(feat).squeeze(-1), self.head_class(feat)


def load_init_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load every tensor whose name and shape match; mismatching ones keep their fresh init."""
    state = torch.load(path, map_location=device, weights_only=True)
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    if not compatible:
        raise ValueError(f"No compatible tensor in {path} (wrong --arch?)")
    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(own) - set(compatible))
    print(
        f"Loaded {len(compatible)}/{len(own)} tensors from {path}; "
        f"re-initialised: {skipped[:6]}{' ...' if len(skipped) > 6 else ''}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-dir", type=Path, required=True)
    parser.add_argument(
        "--arch",
        choices=("mobilenet_v3_small", "resnet18"),
        default="mobilenet_v3_small",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/audio_cnn"))
    parser.add_argument(
        "--calibration-dir", type=Path, default=Path("../calibration_spectrograms")
    )
    # --- two-stage training ---
    parser.add_argument(
        "--wake-only",
        action="store_true",
        help="Stage 1: train backbone + wake head only (class loss disabled).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Stage 2: state_dict .pt to start from (tensors with matching shapes only).",
    )
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="Backbone LR = lr * mult (e.g. 0.1 in stage 2 to preserve stage-1 features).",
    )
    parser.add_argument(
        "--class-loss-weight",
        type=float,
        default=0.5,
        help="Weight of the 4-class CE term (ignored with --wake-only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_ds = SpectrogramDataset(args.challenge_dir, "train", args.img_size)
    valid_ds = SpectrogramDataset(args.challenge_dir, "valid", args.img_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pos_count = sum(1 for r in train_ds.rows if r["has_drone_audio"])
    neg_count = max(1, len(train_ds.rows) - pos_count)
    # Mild bounded positive weighting + post-hoc validation threshold sweep prevent runaway FPs.
    bounded_pos_weight = min(2.0, max(1.0, (neg_count / max(1, pos_count)) ** 0.5))
    pos_weight = torch.tensor([bounded_pos_weight], device=device)

    model = AcousticWakeCNN(args.arch, pretrained=not args.no_pretrained).to(device)
    if args.init_checkpoint is not None:
        load_init_checkpoint(model, args.init_checkpoint, device)

    head_params = list(model.head_wake.parameters()) + list(model.head_class.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()

    stage = "stage 1 (wake only)" if args.wake_only else "wake + 4-class"
    if args.init_checkpoint is not None:
        init_desc = str(args.init_checkpoint)
    else:
        init_desc = "random" if args.no_pretrained else "ImageNet"
    print(
        f"Training {stage} | arch={args.arch} | init={init_desc} "
        f"| backbone_lr={args.lr * args.backbone_lr_mult:.2e} head_lr={args.lr:.2e}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_utility = -1.0
    best_thresh = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            wake_l, cls_l = model(batch["spectrogram"].to(device))
            loss = bce(wake_l, batch["has_drone"].to(device))
            if not args.wake_only:
                loss = loss + args.class_loss_weight * ce(
                    cls_l, batch["sound_class"].to(device)
                )
            loss.backward()
            optimizer.step()

        model.eval()
        prob_batches, trues, cls_hits = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                wake_l, cls_l = model(batch["spectrogram"].to(device))
                prob_batches.append(torch.sigmoid(wake_l).cpu().numpy())
                trues.append(batch["has_drone"].numpy() > 0.5)
                cls_hits.append((cls_l.argmax(1).cpu() == batch["sound_class"]).numpy())
        probs_np = np.concatenate(prob_batches)
        t_np = np.concatenate(trues)
        cls_acc = float(np.concatenate(cls_hits).mean())
        ep_best_u = -1.0
        ep_best_t = 0.5
        ep_stats = (0.0, 0.0, 0, 0, 0)
        for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            p_np = probs_np >= thr
            tp = int((p_np & t_np).sum())
            fp = int((p_np & ~t_np).sum())
            fn = int((~p_np & t_np).sum())
            f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
            duty_cycle = float(p_np.mean())
            utility = f1 - 0.15 * duty_cycle
            if utility > ep_best_u:
                ep_best_u = utility
                ep_best_t = thr
                ep_stats = (f1, duty_cycle, tp, fp, fn)
        f1, duty_cycle, tp, fp, fn = ep_stats
        acc_txt = "n/a (wake-only)" if args.wake_only else f"{cls_acc:.4f}"
        print(
            f"Epoch {epoch}/{args.epochs} valid_wake_F1={f1:.4f} duty={duty_cycle:.2f} "
            f"@thr={ep_best_t:.2f} (TP={tp} FP={fp} FN={fn}) valid_cls_acc={acc_txt}"
        )
        if ep_best_u >= best_utility:
            best_utility = ep_best_u
            best_thresh = ep_best_t
            torch.save(model.state_dict(), args.output_dir / "audio_wake_best.pt")

    model.load_state_dict(
        torch.load(
            args.output_dir / "audio_wake_best.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval().cpu()
    dummy = torch.rand(1, 3, args.img_size, args.img_size)
    onnx_path = args.output_dir / "audio_wake_cnn.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version=17,
        input_names=["spectrogram"],
        output_names=["wake_logit", "sound_class_logits"],
        dynamo=False,
    )
    exported_model = onnx.load(onnx_path)
    metadata = {entry.key: entry.value for entry in exported_model.metadata_props}
    metadata["wake_threshold"] = str(best_thresh)
    metadata["training_stage"] = "wake_only" if args.wake_only else "wake_and_class"
    onnx.helper.set_model_props(exported_model, metadata)
    onnx.save(exported_model, onnx_path)
    print(
        f"Saved {onnx_path} (best_utility={best_utility:.4f}, best_wake_thresh={best_thresh:.2f}; input scaled [0, 1])"
    )
    if args.wake_only:
        print(
            "WARNING: stage-1 model — sound_class_logits are untrained. Do NOT submit; "
            f"run stage 2 with --init-checkpoint {args.output_dir / 'audio_wake_best.pt'}"
        )
    print(
        "\n--- Axelera Voyager SDK Quantization / Compilation Guide ---\n"
        "ONNX graph bakes ImageNet mean/std internally (input tensor expected in [0, 1]).\n"
        "SDK command template: first supply preprocess.py with the SDK callback\n"
        "for RGB resize, CHW float32, and [0, 1] scaling (no extra normalization).\n"
        "Verify SDK flags with `axcompile --help`; hardware placement is untested.\n"
        f"axcompile -i {onnx_path} --imageset {args.calibration_dir} "
        "--transform preprocess.py "
        f"--input-shape 1,3,{args.img_size},{args.img_size} -o {args.output_dir / 'metis_axm'}/\n"
    )


if __name__ == "__main__":
    main()
