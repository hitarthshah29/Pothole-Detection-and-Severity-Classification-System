"""
app.py

Streamlit web application for Pothole Severity Classification
(Minor, Moderate, Severe) using MobileNetV2 transfer learning models.

Features:
- Robust path resolution relative to app.py location via pathlib.Path.
- Sidebar Model Selector:
  1. Experiment 1 — Class-Weighted (Recommended & Default)
  2. Original Baseline
  3. Experiment 2 — Damped & Anchored
- Optional Experiment 3 Minor Threshold Calibration (tau = 0.450) toggle.
- Clear display of selected model, calibration mode, predicted severity,
  model confidence note, and class probability distribution.

IMPORTANT SCIENTIFIC LIMITATION:
The severity categories (Minor, Moderate, Severe) are experimental proxy labels based on
the 2D bounding-box area within the dataset images. They do not measure physical depth,
sub-surface roadbed damage, or engineering-certified structural severity.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd

# Base directory resolved relative to app.py location
BASE_DIR = Path(__file__).resolve().parent

# Model configuration catalog
MODEL_CONFIGS = {
    "Experiment 1 — Class-Weighted (Recommended)": {
        "model_path": BASE_DIR / "models" / "experiments" / "pothole_severity_model.keras",
        "class_names_path": BASE_DIR / "models" / "experiments" / "class_names.json",
        "description": "Trained with balanced class weighting to address class imbalance, boosting Minor recall to 51.85% (vs. 22.22% baseline).",
        "supports_calibration": True
    },
    "Original Baseline": {
        "model_path": BASE_DIR / "models" / "pothole_severity_model.keras",
        "class_names_path": BASE_DIR / "models" / "class_names.json",
        "description": "Original unweighted transfer learning model. High Moderate/Severe recall (67.9% / 74.0%) but lowest Minor recall (22.22%).",
        "supports_calibration": False
    },
    "Experiment 2 — Damped & Anchored": {
        "model_path": BASE_DIR / "models" / "experiments" / "experiment_2" / "pothole_severity_model.keras",
        "class_names_path": BASE_DIR / "models" / "experiments" / "experiment_2" / "class_names.json",
        "description": "Trained with square-root damped, Moderate-anchored class weights to reduce Moderate-to-Minor confusion.",
        "supports_calibration": False
    }
}

DEFAULT_MODEL_NAME = "Experiment 1 — Class-Weighted (Recommended)"
CALIBRATION_THRESHOLD = 0.450


@st.cache_resource
def load_classification_model(model_path_str: str, class_names_path_str: str):
    """
    Loads the trained Keras model and the class name mapping.
    Cached by file paths to avoid redundant reloading across sessions.
    """
    model_path = Path(model_path_str)
    class_names_path = Path(class_names_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found at: {class_names_path}")

    model = load_model(str(model_path))

    with open(class_names_path, "r") as f:
        class_mapping = json.load(f)

    class_names = {int(k): v for k, v in class_mapping.items()}
    return model, class_names


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepares uploaded image to match MobileNetV2 expectations:
    1. Converts non-RGB images to 3-channel RGB
    2. Resizes to 224x224 pixels
    3. Adds batch dimension: shape (1, 224, 224, 3)
    4. Applies MobileNetV2 preprocess_input (scales to [-1, 1])
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def apply_decision_rule(probabilities: np.ndarray, use_calibration: bool) -> tuple[int, str]:
    """
    Computes final class index based on calibration toggle:
    - If calibration enabled: predict Minor (0) if p(Minor) >= 0.450, else argmax(Moderate, Severe).
    - Otherwise: standard argmax across all 3 classes.
    """
    if use_calibration:
        p_minor = float(probabilities[0])
        p_moderate = float(probabilities[1])
        p_severe = float(probabilities[2])

        if p_minor >= CALIBRATION_THRESHOLD:
            predicted_idx = 0
            rule_note = f"Calibrated Decision Rule (Minor probability {p_minor*100:.1f}% ≥ {CALIBRATION_THRESHOLD*100:.1f}%)"
        else:
            predicted_idx = 1 if p_moderate >= p_severe else 2
            higher_class = "Moderate" if p_moderate >= p_severe else "Severe"
            rule_note = f"Calibrated Fallback (Minor < {CALIBRATION_THRESHOLD*100:.1f}%; higher between Moderate & Severe selected: {higher_class})"
    else:
        predicted_idx = int(np.argmax(probabilities))
        rule_note = "Standard Argmax (Highest Softmax Probability)"

    return predicted_idx, rule_note


def main():
    st.set_page_config(page_title="Pothole Severity Classification", layout="centered")

    st.title("Pothole Detection and Severity Classification System")
    st.write(
        "Upload an image of a road surface. The system predicts pothole severity as "
        "**Minor, Moderate, or Severe** using fine-tuned MobileNetV2 transfer learning models."
    )

    # Sidebar: Model Selection
    st.sidebar.header("Model Configuration")

    model_options = list(MODEL_CONFIGS.keys())
    selected_model_name = st.sidebar.selectbox(
        "Select Classification Model:",
        options=model_options,
        index=model_options.index(DEFAULT_MODEL_NAME)
    )

    config = MODEL_CONFIGS[selected_model_name]
    st.sidebar.caption(config["description"])

    # Sidebar: Experiment 3 Threshold Calibration Toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Decision Post-Processing")

    if config["supports_calibration"]:
        apply_calibration = st.sidebar.checkbox(
            f"Apply Experiment 3 Minor Threshold Calibration (τ = {CALIBRATION_THRESHOLD:.3f})",
            value=False,
            help="Post-processing rule pre-selected on validation data: predicts Minor if p(Minor) ≥ 0.450; otherwise selects higher between Moderate and Severe."
        )
        if apply_calibration:
            st.sidebar.info(f"Calibration Active: Minor threshold set to {CALIBRATION_THRESHOLD:.3f}.")
        else:
            st.sidebar.caption("Standard argmax classification is active.")
    else:
        apply_calibration = False
        st.sidebar.checkbox(
            f"Apply Experiment 3 Minor Threshold Calibration (τ = {CALIBRATION_THRESHOLD:.3f})",
            value=False,
            disabled=True,
            help="Threshold calibration was validated and optimized specifically for Experiment 1."
        )
        st.sidebar.caption("Note: Experiment 3 threshold calibration is available exclusively for Experiment 1.")

    # Load Selected Model & Class Mapping
    try:
        model, class_names = load_classification_model(
            str(config["model_path"]),
            str(config["class_names_path"])
        )
    except Exception as e:
        st.error(f"Error loading model '{selected_model_name}': {e}")
        st.stop()

    # Image Uploader
    uploaded_file = st.file_uploader(
        "Choose a road image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Road Image", use_container_width=True)

            if st.button("Classify Pothole Severity", type="primary"):
                with st.spinner("Analyzing image..."):
                    processed_img = preprocess_image(image)
                    probabilities = model.predict(processed_img, verbose=0)[0]

                    predicted_index, rule_note = apply_decision_rule(probabilities, apply_calibration)
                    predicted_class = class_names[predicted_index]
                    highest_prob = float(np.max(probabilities)) * 100.0

                    st.markdown("---")
                    st.subheader("Classification Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Active Model:** `{selected_model_name}`")
                        st.markdown(f"**Calibration Mode:** `{'Enabled (τ = 0.450)' if apply_calibration else 'Disabled (Standard Argmax)'}`")
                        st.caption(f"*Decision Logic:* {rule_note}")

                    with col2:
                        st.success(f"### Predicted Severity: {predicted_class}")
                        st.info(
                            f"**Model Confidence:** {highest_prob:.2f}%\n\n"
                            "*(Note: Reflects highest softmax probability from model output; "
                            "indicates relative model confidence, not a guarantee of prediction correctness.)*"
                        )

                    st.markdown("### Class Probability Distribution")
                    prob_dict = {
                        class_names[i]: float(probabilities[i] * 100.0)
                        for i in range(len(class_names))
                    }
                    df_probs = pd.DataFrame(
                        list(prob_dict.values()),
                        index=list(prob_dict.keys()),
                        columns=["Probability (%)"]
                    )
                    st.bar_chart(df_probs)

        except Exception as e:
            st.error(f"An error occurred while analyzing the image: {e}")

    # Scientific Disclaimer
    st.markdown("---")
    st.caption(
        "**Scientific Disclaimer:** Severity classifications (Minor, Moderate, Severe) are based on "
        "2D bounding-box area proxy labels derived during dataset preparation. They do not measure physical "
        "pothole depth, structural sub-base damage, or vehicle hazard ratings."
    )


if __name__ == "__main__":
    main()
