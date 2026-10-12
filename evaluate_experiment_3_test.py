"""
evaluate_experiment_3_test.py

Final Test-Set Evaluation for Experiment 3 using the Fixed Threshold tau = 0.450
(pre-selected from validation results).

Methodology & Protocol:
1. Loads the existing Experiment 1 model (models/experiments/pothole_severity_model.keras)
   in read-only mode without retraining or adjusting weights.
2. Evaluates on the test split (data/processed/test, 133 images).
3. Applies fixed decision rule at tau = 0.450:
   - If p(Minor) >= 0.450 -> predict Minor (0)
   - Else -> argmax between Moderate (1) and Severe (2)
4. Computes ordinary argmax predictions from the same model on the same images
   for a fair side-by-side comparison.
5. Saves all outputs exclusively under outputs/experiments/experiment_3/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configurations
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ['Minor', 'Moderate', 'Severe']
FIXED_TAU = 0.450

# Paths resolved relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed', 'test')

# Model path from Experiment 1
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'experiments', 'pothole_severity_model.keras')
CLASS_NAMES_PATH = os.path.join(SCRIPT_DIR, 'models', 'experiments', 'class_names.json')

# Output destination for Experiment 3 test evaluation
OUTPUTS_EXP3_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'experiments', 'experiment_3')
REPORT_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'test_evaluation_report.txt')
CSV_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'test_comparison_summary.csv')
CM_CALIBRATED_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'test_confusion_matrix_calibrated.png')
CM_ARGMAX_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'test_confusion_matrix_argmax.png')


def verify_inputs():
    """Confirms required input model, class names, and test dataset exist."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Experiment 1 model not found at '{MODEL_PATH}'.")
    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names mapping not found at '{CLASS_NAMES_PATH}'.")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test dataset directory not found at '{TEST_DIR}'.")
    
    os.makedirs(OUTPUTS_EXP3_DIR, exist_ok=True)


def load_test_generator():
    """
    Loads test split:
    - Preprocessing: MobileNetV2 preprocess_input
    - Image size: (224, 224)
    - Classes: ['Minor', 'Moderate', 'Severe']
    - Shuffle: False
    """
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print("Loading test dataset split...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    return test_generator


def apply_decision_rule(probabilities, tau):
    """
    Decision Rule:
    - If p_Minor >= tau -> predict Minor (0)
    - Else -> argmax between Moderate (1) and Severe (2)
    """
    p_minor = probabilities[:, 0]
    p_moderate = probabilities[:, 1]
    p_severe = probabilities[:, 2]

    fallback = np.where(p_moderate >= p_severe, 1, 2)
    return np.where(p_minor >= tau, 0, fallback)


def evaluate_test_set():
    print("=" * 75)
    print(f"Experiment 3: Final Test-Set Evaluation with Fixed Threshold tau = {FIXED_TAU:.3f}")
    print("=" * 75)

    verify_inputs()

    # Load class mapping
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_mapping = json.load(f)
    print(f"Verified Class Mapping: {class_mapping}")

    # Load test data
    test_generator = load_test_generator()
    y_true = test_generator.classes
    class_labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
    total_test = len(y_true)
    print(f"Total Test Samples: {total_test}")
    labels, counts = np.unique(y_true, return_counts=True)
    for lbl, cnt in zip(labels, counts):
        print(f" - {class_labels[lbl]} (class {lbl}): {cnt} samples ({cnt/total_test*100:.2f}%)")

    # Load Experiment 1 model
    print(f"\nLoading Experiment 1 model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Run inference on test set
    print("Generating predictions on test set...")
    test_generator.reset()
    probabilities = model.predict(test_generator)

    # 1. Calibrated Predictions (tau = 0.450)
    y_pred_calibrated = apply_decision_rule(probabilities, FIXED_TAU)

    # 2. Ordinary Argmax Predictions
    y_pred_argmax = np.argmax(probabilities, axis=1)

    # Compute metrics for calibrated model
    cm_cal = confusion_matrix(y_true, y_pred_calibrated, labels=[0, 1, 2])
    rep_cal = classification_report(y_true, y_pred_calibrated, target_names=class_labels, digits=4, output_dict=True)
    rep_cal_str = classification_report(y_true, y_pred_calibrated, target_names=class_labels, digits=4)

    # Compute metrics for ordinary argmax
    cm_arg = confusion_matrix(y_true, y_pred_argmax, labels=[0, 1, 2])
    rep_arg = classification_report(y_true, y_pred_argmax, target_names=class_labels, digits=4, output_dict=True)
    rep_arg_str = classification_report(y_true, y_pred_argmax, target_names=class_labels, digits=4)

    # Save Confusion Matrix Plots
    # Calibrated
    disp_cal = ConfusionMatrixDisplay(confusion_matrix=cm_cal, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_cal.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title(f'Test Confusion Matrix (Calibrated tau = {FIXED_TAU:.3f})')
    plt.tight_layout()
    plt.savefig(CM_CALIBRATED_PATH, dpi=150)
    plt.close()

    # Argmax
    disp_arg = ConfusionMatrixDisplay(confusion_matrix=cm_arg, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_arg.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title('Test Confusion Matrix (Ordinary Argmax / Exp 1)')
    plt.tight_layout()
    plt.savefig(CM_ARGMAX_PATH, dpi=150)
    plt.close()

    # Build Comparison DataFrame
    comp_data = [
        {
            'Metric': 'Overall Accuracy',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['accuracy']*100:.2f}%",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['accuracy']*100:.2f}%",
            'Difference': f"{(rep_cal['accuracy'] - rep_arg['accuracy'])*100:+.2f}%"
        },
        {
            'Metric': 'Minor Precision',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Minor']['precision']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Minor']['precision']:.4f}",
            'Difference': f"{(rep_cal['Minor']['precision'] - rep_arg['Minor']['precision']):+.4f}"
        },
        {
            'Metric': 'Minor Recall',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Minor']['recall']*100:.2f}%",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Minor']['recall']*100:.2f}%",
            'Difference': f"{(rep_cal['Minor']['recall'] - rep_arg['Minor']['recall'])*100:+.2f}%"
        },
        {
            'Metric': 'Minor F1-Score',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Minor']['f1-score']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Minor']['f1-score']:.4f}",
            'Difference': f"{(rep_cal['Minor']['f1-score'] - rep_arg['Minor']['f1-score']):+.4f}"
        },
        {
            'Metric': 'Moderate Precision',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Moderate']['precision']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Moderate']['precision']:.4f}",
            'Difference': f"{(rep_cal['Moderate']['precision'] - rep_arg['Moderate']['precision']):+.4f}"
        },
        {
            'Metric': 'Moderate Recall',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Moderate']['recall']*100:.2f}%",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Moderate']['recall']*100:.2f}%",
            'Difference': f"{(rep_cal['Moderate']['recall'] - rep_arg['Moderate']['recall'])*100:+.2f}%"
        },
        {
            'Metric': 'Moderate F1-Score',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Moderate']['f1-score']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Moderate']['f1-score']:.4f}",
            'Difference': f"{(rep_cal['Moderate']['f1-score'] - rep_arg['Moderate']['f1-score']):+.4f}"
        },
        {
            'Metric': 'Severe Precision',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Severe']['precision']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Severe']['precision']:.4f}",
            'Difference': f"{(rep_cal['Severe']['precision'] - rep_arg['Severe']['precision']):+.4f}"
        },
        {
            'Metric': 'Severe Recall',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Severe']['recall']*100:.2f}%",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Severe']['recall']*100:.2f}%",
            'Difference': f"{(rep_cal['Severe']['recall'] - rep_arg['Severe']['recall'])*100:+.2f}%"
        },
        {
            'Metric': 'Severe F1-Score',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['Severe']['f1-score']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['Severe']['f1-score']:.4f}",
            'Difference': f"{(rep_cal['Severe']['f1-score'] - rep_arg['Severe']['f1-score']):+.4f}"
        },
        {
            'Metric': 'Macro F1-Score',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['macro avg']['f1-score']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['macro avg']['f1-score']:.4f}",
            'Difference': f"{(rep_cal['macro avg']['f1-score'] - rep_arg['macro avg']['f1-score']):+.4f}"
        },
        {
            'Metric': 'Weighted F1-Score',
            'Uncalibrated Argmax (Exp 1)': f"{rep_arg['weighted avg']['f1-score']:.4f}",
            'Calibrated tau = 0.450 (Exp 3)': f"{rep_cal['weighted avg']['f1-score']:.4f}",
            'Difference': f"{(rep_cal['weighted avg']['f1-score'] - rep_arg['weighted avg']['f1-score']):+.4f}"
        }
    ]

    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(CSV_PATH, index=False)

    # Write Complete Test Report
    with open(REPORT_PATH, 'w') as f:
        f.write("Experiment 3: Final Test-Set Evaluation Report\n")
        f.write("Model Evaluated: Experiment 1 MobileNetV2 (Phase 2 Fine-Tuned)\n")
        f.write(f"Decision Rule: p(Minor) >= {FIXED_TAU:.3f} -> Minor; Else argmax(Moderate, Severe)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"TEST RESULTS (CALIBRATED MODEL, tau = {FIXED_TAU:.3f}):\n")
        f.write(f"- Test Accuracy: {rep_cal['accuracy']*100:.2f}% ({int(np.sum(np.diag(cm_cal)))} / {total_test})\n")
        f.write(f"- Macro Precision: {rep_cal['macro avg']['precision']:.4f}\n")
        f.write(f"- Macro Recall:    {rep_cal['macro avg']['recall']:.4f}\n")
        f.write(f"- Macro F1-Score:  {rep_cal['macro avg']['f1-score']:.4f}\n")
        f.write(f"- Weighted F1-Score: {rep_cal['weighted avg']['f1-score']:.4f}\n\n")

        f.write("CALIBRATED CONFUSION MATRIX (tau = 0.450):\n")
        f.write(f"                Pred Minor  Pred Mod  Pred Sev\n")
        f.write(f"  Actual Minor     {cm_cal[0,0]:<10} {cm_cal[0,1]:<9} {cm_cal[0,2]:<8} | {np.sum(cm_cal[0])}\n")
        f.write(f"  Actual Mod       {cm_cal[1,0]:<10} {cm_cal[1,1]:<9} {cm_cal[1,2]:<8} | {np.sum(cm_cal[1])}\n")
        f.write(f"  Actual Sev       {cm_cal[2,0]:<10} {cm_cal[2,1]:<9} {cm_cal[2,2]:<8} | {np.sum(cm_cal[2])}\n\n")

        f.write("CALIBRATED CLASSIFICATION REPORT:\n")
        f.write(rep_cal_str + "\n\n")

        f.write("-" * 70 + "\n")
        f.write("SIDE-BY-SIDE COMPARISON: ORDINARY ARGMAX (EXP 1) vs. CALIBRATED (EXP 3):\n")
        f.write("-" * 70 + "\n\n")

        f.write("ORDINARY ARGMAX CONFUSION MATRIX (EXP 1):\n")
        f.write(f"                Pred Minor  Pred Mod  Pred Sev\n")
        f.write(f"  Actual Minor     {cm_arg[0,0]:<10} {cm_arg[0,1]:<9} {cm_arg[0,2]:<8} | {np.sum(cm_arg[0])}\n")
        f.write(f"  Actual Mod       {cm_arg[1,0]:<10} {cm_arg[1,1]:<9} {cm_arg[1,2]:<8} | {np.sum(cm_arg[1])}\n")
        f.write(f"  Actual Sev       {cm_arg[2,0]:<10} {cm_arg[2,1]:<9} {cm_arg[2,2]:<8} | {np.sum(cm_arg[2])}\n\n")

        f.write("ORDINARY ARGMAX CLASSIFICATION REPORT:\n")
        f.write(rep_arg_str + "\n\n")

        f.write("DETAILED METRIC COMPARISON TABLE:\n")
        f.write(df_comp.to_string(index=False) + "\n")

    print(f"\nFinal test evaluation completed successfully!")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Summary CSV saved to: {CSV_PATH}")


if __name__ == '__main__':
    try:
        evaluate_test_set()
    except Exception as e:
        print(f"\nTest evaluation failed with error:\n{e}", file=sys.stderr)
        sys.exit(1)
