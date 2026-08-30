import streamlit as st
import torch
from PIL import Image

from src.config import BEST_MODEL_PATH
from src.dataset import inference_transform
from src.model import load_checkpoint
from src.nutrition import display_name, estimate_intake, load_nutrition

st.set_page_config(page_title="Food Calorie Tracker", page_icon="🥗", layout="wide")


@st.cache_resource
def load_assets():
    if not BEST_MODEL_PATH.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, metrics = load_checkpoint(BEST_MODEL_PATH, device)
    nutrition = load_nutrition()
    return model, device, class_names, metrics, nutrition, inference_transform()


def predict(model, device, class_names, transform, image: Image.Image, top_k: int = 3):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    values, indices = probs.topk(top_k)
    return [
        {"class_name": class_names[i], "confidence": float(p)}
        for p, i in zip(values, indices)
    ]


loaded = load_assets()

st.title("Food Recognition & Calorie Tracker")
st.caption(
    "Upload a meal photo. The model identifies the Food-101 dish, then estimates calories "
    "from typical USDA-style nutrition values. Portion size is an estimate — adjust grams to match your plate."
)

if loaded is None:
    st.error("No trained model found. From the project root run: `python -m src.train`")
    st.stop()

model, device, class_names, metrics, nutrition, transform = loaded
best = metrics.get("best_top1")
if best is not None:
    st.sidebar.metric("Validation top-1", f"{best * 100:.1f}%")
st.sidebar.metric("Device", str(device).upper())
st.sidebar.write(f"{len(class_names)} food classes")

if "log" not in st.session_state:
    st.session_state.log = []

col_left, col_right = st.columns([1.1, 1])

with col_left:
    uploaded = st.file_uploader("Meal photo", type=["jpg", "jpeg", "png", "webp"])
    camera = st.camera_input("Or take a photo")
    image_file = uploaded or camera
    image = Image.open(image_file) if image_file is not None else None
    if image is None:
        st.info("Add a photo to classify a dish and log calories.")
    else:
        st.image(image, caption="Input", use_container_width=True)

with col_right:
    if image is not None:
        predictions = predict(model, device, class_names, transform, image, top_k=3)
        choice_labels = [
            f"{display_name(p['class_name'])} ({p['confidence'] * 100:.1f}%)"
            for p in predictions
        ]
        selected = st.radio("Top predictions — pick the dish that matches", choice_labels, index=0)
        selected_pred = predictions[choice_labels.index(selected)]
        entry = nutrition[selected_pred["class_name"]]
        grams = st.slider(
            "Portion size (grams)",
            min_value=20,
            max_value=800,
            value=int(entry["typical_serving_g"]),
            step=5,
            help=entry["serving_description"],
        )
        intake = estimate_intake(entry, grams=grams)
        st.subheader(intake["food"])
        st.write(entry["serving_description"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories", f"{intake['calories']:.0f} kcal")
        m2.metric("Protein", f"{intake['protein_g']:.1f} g")
        m3.metric("Fat", f"{intake['fat_g']:.1f} g")
        m4.metric("Carbs", f"{intake['carbs_g']:.1f} g")
        st.caption(f"{entry['calories_per_100g']} kcal per 100 g. {entry.get('notes', '')}")
        if st.button("Add to today's log", type="primary"):
            st.session_state.log.append(
                {
                    "food": intake["food"],
                    "grams": intake["grams"],
                    "calories": intake["calories"],
                    "protein_g": intake["protein_g"],
                    "fat_g": intake["fat_g"],
                    "carbs_g": intake["carbs_g"],
                    "confidence": selected_pred["confidence"],
                }
            )
            st.success("Added to log")

st.divider()
st.subheader("Today's intake")
if not st.session_state.log:
    st.write("Nothing logged yet.")
else:
    total_kcal = sum(item["calories"] for item in st.session_state.log)
    total_p = sum(item["protein_g"] for item in st.session_state.log)
    total_f = sum(item["fat_g"] for item in st.session_state.log)
    total_c = sum(item["carbs_g"] for item in st.session_state.log)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Calories", f"{total_kcal:.0f}")
    t2.metric("Protein", f"{total_p:.1f} g")
    t3.metric("Fat", f"{total_f:.1f} g")
    t4.metric("Carbs", f"{total_c:.1f} g")
    st.dataframe(st.session_state.log, use_container_width=True, hide_index=True)
    if st.button("Clear log"):
        st.session_state.log = []
        st.rerun()

st.caption(
    "Estimates use class-level nutrition lookups, not 3D volume from the photo. "
    "Use them as a tracking aid, not medical advice."
)
