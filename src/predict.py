"""Predict food class and calorie estimate from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import BEST_MODEL_PATH, NUTRITION_PATH
from src.dataset import inference_transform
from src.model import load_checkpoint
from src.nutrition import display_name, estimate_intake, load_nutrition


def predict_image(image_path: str | Path, checkpoint: str | Path = BEST_MODEL_PATH, top_k: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, _ = load_checkpoint(checkpoint, device)
    nutrition = load_nutrition(NUTRITION_PATH)
    transform = inference_transform()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    values, indices = probs.topk(top_k)
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        class_name = class_names[idx]
        entry = nutrition[class_name]
        intake = estimate_intake(entry)
        results.append(
            {
                "class_name": class_name,
                "label": display_name(class_name),
                "confidence": score,
                "nutrition": intake,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to a food image")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--grams", type=float, default=None, help="Override portion size in grams")
    args = parser.parse_args()

    predictions = predict_image(args.image, args.checkpoint, args.top_k)
    nutrition = load_nutrition()
    print(f"Image: {args.image}\n")
    for i, item in enumerate(predictions, start=1):
        entry = nutrition[item["class_name"]]
        intake = estimate_intake(entry, grams=args.grams)
        print(
            f"{i}. {item['label']}  ({item['confidence'] * 100:.1f}%)\n"
            f"   {intake['calories']} kcal for {intake['grams']:.0f} g "
            f"({intake['serving_description']})\n"
            f"   P {intake['protein_g']} g  F {intake['fat_g']} g  C {intake['carbs_g']} g\n"
        )


if __name__ == "__main__":
    main()
