"""Generate evaluation plots from checkpoints/history.json into docs/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "checkpoints" / "history.json"
OUT_DIR = ROOT / "docs"
FREEZE_EPOCHS = 3


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
        }
    )


def save_jpg(fig, name: str) -> None:
    fig.savefig(
        OUT_DIR / name,
        dpi=160,
        format="jpg",
        pil_kwargs={"quality": 92},
        facecolor="white",
        edgecolor="white",
    )


def mark_unfreeze(ax):
    ax.axvline(FREEZE_EPOCHS + 0.5, color="#888888", linestyle="--", linewidth=1.2)
    ymin, ymax = ax.get_ylim()
    ax.text(
        FREEZE_EPOCHS + 0.55,
        ymax - 0.04 * (ymax - ymin),
        "unfreeze backbone",
        color="#555555",
        fontsize=9,
        va="top",
    )


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_top1"])) + 1
    best_top1 = max(history["val_top1"])
    best_top5 = max(history["val_top5"])

    # Loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax.plot(epochs, history["val_loss"], marker="s", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.set_xticks(epochs)
    ax.legend()
    mark_unfreeze(ax)
    fig.tight_layout()
    save_jpg(fig, "loss.jpg")
    plt.close(fig)

    # Top-1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, np.array(history["train_top1"]) * 100, marker="o", label="Train top-1")
    ax.plot(epochs, np.array(history["val_top1"]) * 100, marker="s", label="Val top-1")
    ax.scatter([best_epoch], [best_top1 * 100], color="red", zorder=5, label=f"Best val ({best_top1 * 100:.1f}%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-1 accuracy")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.legend()
    mark_unfreeze(ax)
    fig.tight_layout()
    save_jpg(fig, "top1_accuracy.jpg")
    plt.close(fig)

    # Top-5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, np.array(history["val_top5"]) * 100, marker="s", color="#2a9d8f", label="Val top-5")
    ax.scatter([int(np.argmax(history["val_top5"])) + 1], [best_top5 * 100], color="red", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-5 validation accuracy")
    ax.set_xticks(epochs)
    ax.set_ylim(70, 100)
    ax.legend()
    mark_unfreeze(ax)
    fig.tight_layout()
    save_jpg(fig, "top5_accuracy.jpg")
    plt.close(fig)

    # Generalization gap
    gap = (np.array(history["train_top1"]) - np.array(history["val_top1"])) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, gap, marker="o", color="#e76f51")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train − val top-1 (pp)")
    ax.set_title("Generalization gap")
    ax.set_xticks(epochs)
    mark_unfreeze(ax)
    fig.tight_layout()
    save_jpg(fig, "generalization_gap.jpg")
    plt.close(fig)

    # Overview 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], marker="s", label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].set_xticks(epochs)

    axes[0, 1].plot(epochs, np.array(history["train_top1"]) * 100, marker="o", label="Train")
    axes[0, 1].plot(epochs, np.array(history["val_top1"]) * 100, marker="s", label="Val")
    axes[0, 1].set_title("Top-1 accuracy (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].set_xticks(epochs)
    axes[0, 1].set_ylim(0, 100)

    axes[1, 0].plot(epochs, np.array(history["val_top5"]) * 100, marker="s", color="#2a9d8f")
    axes[1, 0].set_title("Top-5 validation accuracy (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_xticks(epochs)
    axes[1, 0].set_ylim(70, 100)

    axes[1, 1].bar(["Best top-1", "Best top-5", "Final train top-1"], [best_top1 * 100, best_top5 * 100, history["train_top1"][-1] * 100], color=["#457b9d", "#2a9d8f", "#e9c46a"])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_title("Final metrics")
    for i, v in enumerate([best_top1 * 100, best_top5 * 100, history["train_top1"][-1] * 100]):
        axes[1, 1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)

    for ax in axes.ravel()[:3]:
        ax.axvline(FREEZE_EPOCHS + 0.5, color="#888888", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Food-101 EfficientNet-B0 evaluation", fontsize=14)
    fig.tight_layout()
    save_jpg(fig, "evaluation_overview.jpg")
    plt.close(fig)

    print(f"Wrote plots to {OUT_DIR}")
    print(f"Best val top-1: {best_top1 * 100:.2f}% (epoch {best_epoch})")
    print(f"Best val top-5: {best_top5 * 100:.2f}%")


if __name__ == "__main__":
    main()
