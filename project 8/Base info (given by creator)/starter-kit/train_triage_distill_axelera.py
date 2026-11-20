"""Train a lightweight multi-head Edge Triage Distillation CNN and export static ONNX."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MODALITIES = ("rgb", "thermal", "night")
PERSPECTIVES = ("from_drone", "ground", "of_drone", "other")
SCENE_KEYS = (
    "armored_vehicle",
    "vehicle",
    "person",
    "drone",
    "explosion_smoke",
    "trench_building",
)
_MOD_INDEX = {name: i for i, name in enumerate(MODALITIES)}
_PERSP_INDEX = {name: i for i, name in enumerate(PERSPECTIVES)}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EdgeTriageDataset(Dataset):
    """Sanitized Challenge 4 frame dataset loaded from labels.jsonl."""

    def __init__(self, dataset_dir: Path, split: str, img_size: int = 224) -> None:
        self.dataset_dir = dataset_dir
        jsonl_path = dataset_dir / split / "labels.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing labels file: {jsonl_path}")
        self.rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"Empty dataset split {split!r} in {jsonl_path}")
        # Keep [0, 1] RGB scaling at the dataset boundary; ImageNet normalization lives
        # inside the exported ONNX graph so calibration images match training inputs.
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
        with Image.open(self.dataset_dir / row["image"]) as im:
            x = self.transform(im.convert("RGB"))
        mod_idx = _MOD_INDEX.get(str(row["modality"]), _MOD_INDEX["rgb"])
        persp_idx = _PERSP_INDEX.get(str(row["perspective"]), _PERSP_INDEX["other"])
        return {
            "image": x,
            "keep": torch.tensor(float(row["triage_keep"]), dtype=torch.float32),
            "priority": torch.tensor(float(row["priority_score"]), dtype=torch.float32),
            "modality": torch.tensor(mod_idx, dtype=torch.long),
            "perspective": torch.tensor(persp_idx, dtype=torch.long),
            "scene": torch.tensor(row["scene_vector"], dtype=torch.float32),
        }


class EdgeTriageNet(nn.Module):
    """Multi-head MobileNetV3-Small / ResNet18 distillation network (Metis AIPU friendly)."""

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
        self.head_keep = nn.Linear(feat_dim, 1)
        self.head_priority = nn.Linear(feat_dim, 1)
        self.head_modality = nn.Linear(feat_dim, len(MODALITIES))
        self.head_perspective = nn.Linear(feat_dim, len(PERSPECTIVES))
        self.head_scene = nn.Linear(feat_dim, len(SCENE_KEYS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        norm_x = (x - self.mean) / self.std
        feat = self.backbone(norm_x)
        return (
            self.head_keep(feat).squeeze(-1),
            torch.sigmoid(self.head_priority(feat).squeeze(-1)),
            self.head_modality(feat),
            self.head_perspective(feat),
            self.head_scene(feat),
        )


def recall_at_budget(
    scores: np.ndarray, targets: np.ndarray, budget_frac: float = 0.01
) -> float:
    """Compute fraction of top-priority frames captured within an uplink budget."""
    k = max(1, int(round(len(scores) * budget_frac)))
    top_idx = np.lexsort((np.arange(len(scores)), targets, -scores))[:k]
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    return float(targets[top_idx].sum() / min(k, positives))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--arch",
        choices=("mobilenet_v3_small", "resnet18"),
        default="mobilenet_v3_small",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/triage_distill"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("../calibration"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_ds = EdgeTriageDataset(args.dataset_dir, "train", args.img_size)
    valid_ds = EdgeTriageDataset(args.dataset_dir, "valid", args.img_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = EdgeTriageNet(args.arch, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            keep_l, prio_p, mod_l, persp_l, scene_l = model(batch["image"].to(device))
            loss = (
                bce(keep_l, batch["keep"].to(device))
                + 2.0 * mse(prio_p, batch["priority"].to(device))
                + 0.5 * ce(mod_l, batch["modality"].to(device))
                + 0.5 * ce(persp_l, batch["perspective"].to(device))
                + bce(scene_l, batch["scene"].to(device))
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["keep"])

        model.eval()
        val_scores, val_targets = [], []
        with torch.no_grad():
            for batch in valid_loader:
                _, prio_p, *_ = model(batch["image"].to(device))
                val_scores.append(prio_p.cpu().numpy())
                val_targets.append(
                    (batch["keep"].numpy() > 0.5) & (batch["priority"].numpy() >= 0.7)
                )
        # Match the submitted priority head, six-decimal JSON scores, and scorer ties.
        scores_np = np.asarray(
            [round(float(v), 6) for v in np.concatenate(val_scores)], dtype=np.float64
        )
        targets_np = np.concatenate(val_targets).astype(np.float32)
        r1 = recall_at_budget(scores_np, targets_np, 0.01)
        r5 = recall_at_budget(scores_np, targets_np, 0.05)
        composite = 0.5 * (r1 + r5)
        print(
            f"Epoch {epoch}/{args.epochs} loss={total_loss / len(train_ds):.4f} R@1%={r1:.4f} R@5%={r5:.4f}"
        )
        if composite >= best_score:
            best_score = composite
            torch.save(model.state_dict(), args.output_dir / "triage_best.pt")

    model.load_state_dict(
        torch.load(
            args.output_dir / "triage_best.pt", map_location="cpu", weights_only=True
        )
    )
    model.eval().cpu()
    dummy = torch.rand(1, 3, args.img_size, args.img_size)
    onnx_path = args.output_dir / "triage_distill.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=[
            "keep_logit",
            "priority_score",
            "modality_logits",
            "perspective_logits",
            "scene_logits",
        ],
        dynamo=False,
    )
    print(f"Saved {onnx_path} (best_composite={best_score:.4f}; input scaled [0, 1])")
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
