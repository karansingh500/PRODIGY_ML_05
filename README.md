# Food Recognition and Calorie Estimation

Recognize dishes from photos (Food-101, 101 classes) and estimate calories plus protein, fat, and carbs from typical USDA-style values. Built for Prodigy InfoTech ML intern task 05.

## What it does

1. Classifies an image with a fine-tuned **EfficientNet-B0** model.
2. Maps the predicted dish to **calories and macros per 100 g**.
3. Scales those values to a **typical serving**, which you can change in grams.
4. Lets you **log meals** in a Streamlit tracker.

Food-101 has labels only — no calorie targets. Calorie output is a lookup after classification, not a pixel-to-kcal regressor. That is the usual, honest design for this dataset.

Dataset: [Food-101 on Kaggle](https://www.kaggle.com/dansbecker/food-101) (same data as ETH Zurich Food-101: 101,000 images, 750 train / 250 test per class).

## Setup

Python 3.10+ with PyTorch (CUDA recommended).

```bash
pip install -r requirements.txt
```

## Train

Place the Kaggle Food-101 extract at `data/food-101/food-101/` (with `images/` and `meta/`). Then fine-tune EfficientNet-B0:

- 3 epochs with a frozen backbone (train the classifier head)
- remaining epochs with the backbone unfrozen at a lower learning rate
- mixed precision, label smoothing, and the official train/test split

```bash
python -m src.train
```

Useful flags:

```bash
python -m src.train --epochs 12 --freeze-epochs 3 --batch-size 16
```

On a 4 GB GPU, keep `--batch-size` at 16 (or 8 if you hit out-of-memory). Best weights go to `checkpoints/best_model.pt`.

## Evaluate and predict

```bash
python -m src.evaluate
python -m src.predict path/to/food.jpg --grams 200
```

## App

```bash
streamlit run app.py
```

Upload or capture a photo, confirm the top prediction, set portion size, and add the meal to a daily log.

## Project layout

| Path | Role |
|------|------|
| `src/train.py` | Training loop |
| `src/model.py` | EfficientNet-B0 classifier |
| `src/dataset.py` | Food-101 loaders and transforms |
| `src/predict.py` | CLI inference |
| `src/nutrition.py` | Serving-size scaling |
| `data/nutrition.json` | Per-class calories and macros |
| `app.py` | Streamlit tracker |

## Notes

- Nutrition figures are typical averages. Restaurant recipes, oil, and sides vary.
- Images do not encode true plate weight; always adjust grams.
- Expect roughly mid-70s to low-80s **top-1** and much higher **top-5** after a full fine-tune, depending on epochs and hardware.
