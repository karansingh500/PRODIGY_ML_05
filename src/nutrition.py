import json
from pathlib import Path

from src.config import NUTRITION_PATH


def load_nutrition(path: Path = NUTRITION_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def estimate_intake(entry: dict, grams: float | None = None) -> dict:
    serving_g = float(grams if grams is not None else entry["typical_serving_g"])
    scale = serving_g / 100.0
    return {
        "food": entry["name"],
        "grams": serving_g,
        "serving_description": entry["serving_description"],
        "calories": round(entry["calories_per_100g"] * scale, 1),
        "protein_g": round(entry["protein_g"] * scale, 1),
        "fat_g": round(entry["fat_g"] * scale, 1),
        "carbs_g": round(entry["carbs_g"] * scale, 1),
        "calories_per_100g": entry["calories_per_100g"],
        "notes": entry.get("notes", ""),
    }
