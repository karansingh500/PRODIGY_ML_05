"""Evaluate a trained checkpoint on the Food-101 test split."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.config import BEST_MODEL_PATH, LABEL_SMOOTHING
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, metrics = load_checkpoint(args.checkpoint, device)
    _, test_loader, _ = get_dataloaders(batch_size=args.batch_size, download=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    result = evaluate(model, test_loader, device, criterion)
    print(f"Classes: {len(class_names)}")
    print(f"Saved metrics: {metrics}")
    print(
        f"Test loss {result['loss']:.4f}  "
        f"top-1 {result['top1']:.4f}  top-5 {result['top5']:.4f}"
    )


if __name__ == "__main__":
    main()
