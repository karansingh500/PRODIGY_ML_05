"""Train EfficientNet-B0 on Food-101."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    CLASS_INDEX_PATH,
    EPOCHS,
    FREEZE_EPOCHS,
    HISTORY_PATH,
    LABEL_SMOOTHING,
    LR_FINETUNE,
    LR_HEAD,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloaders
from src.model import build_model, freeze_backbone, save_checkpoint


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    pred = logits.topk(k, dim=1).indices
    return pred.eq(targets.view(-1, 1).expand_as(pred)).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch = images.size(0)
        total_loss += loss.item() * batch
        total_top1 += accuracy(logits, labels, 1) * batch
        total_top5 += accuracy(logits, labels, 5) * batch
        n += batch
    return {
        "loss": total_loss / n,
        "top1": total_top1 / n,
        "top5": total_top5 / n,
    }


def train_one_epoch(model, loader, device, criterion, optimizer, scaler) -> dict:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch = images.size(0)
        total_loss += loss.item() * batch
        total_top1 += accuracy(logits, labels, 1) * batch
        n += batch
    return {"loss": total_loss / n, "top1": total_top1 / n}


def plot_history(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_top1"], label="train")
    axes[1].plot(history["val_top1"], label="val")
    axes[1].set_title("Top-1 accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train food classifier on Food-101")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(batch_size=args.batch_size)
    CLASS_INDEX_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Classes: {len(class_names)} | train batches: {len(train_loader)}")

    from src.nutrition import load_nutrition

    nutrition = load_nutrition()
    missing = [name for name in class_names if name not in nutrition]
    if missing:
        raise SystemExit(f"nutrition.json missing classes: {missing}")

    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    freeze_backbone(model, freeze=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [],
        "train_top1": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
    }
    best_top1 = 0.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("Unfreezing backbone for fine-tuning")
            freeze_backbone(model, freeze=False)
            optimizer = AdamW(
                [
                    {"params": model.features.parameters(), "lr": LR_FINETUNE},
                    {"params": model.classifier.parameters(), "lr": LR_FINETUNE * 5},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.freeze_epochs))

        train_metrics = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_metrics = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_top1"].append(train_metrics["top1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_top1"].append(val_metrics["top1"])
        history["val_top5"].append(val_metrics["top5"])

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train loss {train_metrics['loss']:.3f} acc {train_metrics['top1']:.3f}  "
            f"val loss {val_metrics['loss']:.3f} top1 {val_metrics['top1']:.3f} "
            f"top5 {val_metrics['top5']:.3f}"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                class_names,
                {"best_top1": best_top1, "best_top5": val_metrics["top5"], "epoch": epoch},
            )
            print(f"  saved best model ({best_top1:.3f})")

        HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_history(history, CHECKPOINT_DIR / "training_curves.png")

    elapsed = (time.time() - started) / 60
    print(f"Done in {elapsed:.1f} min. Best val top-1: {best_top1:.3f}")
    print(f"Checkpoint: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
