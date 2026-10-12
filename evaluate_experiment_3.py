"""
evaluate_experiment_3.py

Experiment 3: Validation-Only Minor-Class Decision-Threshold Calibration
using the trained Experiment 1 MobileNetV2 Model.

Safeguards & Methodology:
1. Loads the verified Experiment 1 model and class names (Minor=0, Moderate=1, Severe=2)
   in read-only mode without modifying or retraining weights.
2. Evaluates strictly on the 20% validation split (data/processed/train).
   The test set is NOT loaded or evaluated.
3. Sweeps Minor decision thresholds from 0.250 to 0.450 (step 0.025).
4. Decision Rule:
   - If p_Minor >= tau -> predict Minor (0)
   - Else -> argmax between Moderate (1) and Severe (2)
5. Selection Criteria:
   - Condition: Minor Recall (validation) >= 0.40
   - Primary: Highest Validation Macro F1
   - Tie-Breaker: Highest Validation Accuracy
   - If no threshold meets Minor Recall >= 0.40, report that no eligible threshold exists
     (do NOT silently relax the constraint).
6. All artifacts saved exclusively under outputs/experiments/experiment_3/.
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
THRESHOLDS = [0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450]
MIN_MINOR_RECALL = 0.40

# Paths resolved relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed', 'train')

# Model path from Experiment 1
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'experiments', 'pothole_severity_model.keras')
CLASS_NAMES_PATH = os.path.join(SCRIPT_DIR, 'models', 'experiments', 'class_names.json')

# Output destination for Experiment 3
OUTPUTS_EXP3_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'experiments', 'experiment_3')
REPORT_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'validation_threshold_report.txt')
CSV_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'validation_sweep_metrics.csv')
PLOT_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'validation_threshold_curves.png')
CM_PATH = os.path.join(OUTPUTS_EXP3_DIR, 'val_confusion_matrix_selected.png')


def verify_inputs():
    """Confirms required input model, class names, and dataset directories exist."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Experiment 1 model not found at '{MODEL_PATH}'.")
    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names mapping not found at '{CLASS_NAMES_PATH}'.")
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training dataset directory not found at '{TRAIN_DIR}'.")
    
    os.makedirs(OUTPUTS_EXP3_DIR, exist_ok=True)


def load_validation_generator():
    """
    Loads validation split identically to Experiment 1:
    - Preprocessing: MobileNetV2 preprocess_input
    - Image size: (224, 224)
    - Validation split: 0.2 (20%)
    - Classes: ['Minor', 'Moderate', 'Severe']
    - Shuffle: False (to guarantee 1:1 ground-truth alignment)
    """
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    print("Loading validation dataset split (unaugmented)...")
    val_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    return val_generator


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


def evaluate_threshold_sweep():
    print("=" * 75)
    print("Experiment 3: Minor-Class Decision-Threshold Calibration (Validation Only)")
    print("=" * 75)

    verify_inputs()

    # Load class mapping
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_mapping = json.load(f)
    print(f"Verified Class Mapping: {class_mapping}")

    # Load validation data
    val_generator = load_validation_generator()
    y_true = val_generator.classes
    class_labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
    total_val = len(y_true)
    print(f"Total Validation Samples: {total_val}")
    unique_labels, counts = np.unique(y_true, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        print(f" - {class_labels[lbl]} (class {lbl}): {cnt} samples ({cnt/total_val*100:.2f}%)")

    # Load Experiment 1 model (read-only)
    print(f"\nLoading Experiment 1 model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Compute validation probabilities once
    print("Running inference on validation set...")
    val_generator.reset()
    probabilities = model.predict(val_generator)

    # Evaluate each threshold
    print("\n--- Sweeping Thresholds tau in [0.250, 0.450] on Validation Set ---")
    results = []

    for tau in THRESHOLDS:
        y_pred = apply_decision_rule(probabilities, tau)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)

        minor_p = report_dict['Minor']['precision']
        minor_r = report_dict['Minor']['recall']
        minor_f1 = report_dict['Minor']['f1-score']

        mod_p = report_dict['Moderate']['precision']
        mod_r = report_dict['Moderate']['recall']
        mod_f1 = report_dict['Moderate']['f1-score']

        sev_p = report_dict['Severe']['precision']
        sev_r = report_dict['Severe']['recall']
        sev_f1 = report_dict['Severe']['f1-score']

        macro_p = report_dict['macro avg']['precision']
        macro_r = report_dict['macro avg']['recall']
        macro_f1 = report_dict['macro avg']['f1-score']

        acc = report_dict['accuracy']
        is_eligible = (minor_r >= MIN_MINOR_RECALL)

        results.append({
            'threshold': tau,
            'minor_prec': minor_p,
            'minor_recall': minor_r,
            'minor_f1': minor_f1,
            'mod_prec': mod_p,
            'mod_recall': mod_r,
            'mod_f1': mod_f1,
            'sev_prec': sev_p,
            'sev_recall': sev_r,
            'sev_f1': sev_f1,
            'macro_prec': macro_p,
            'macro_recall': macro_r,
            'macro_f1': macro_f1,
            'accuracy': acc,
            'eligible': is_eligible,
            'confusion_matrix': cm
        })

    # Save CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in results])
    df.to_csv(CSV_PATH, index=False)
    print(f"Validation metrics table saved to: {CSV_PATH}")

    # Threshold Selection based strictly on Validation Criteria:
    # 1. Condition: minor_recall >= 0.40
    # 2. Primary: highest macro_f1
    # 3. Tie-breaker: highest accuracy
    eligible_candidates = [r for r in results if r['eligible']]
    
    if eligible_candidates:
        # Sort by macro_f1 descending, then accuracy descending
        eligible_candidates.sort(key=lambda x: (x['macro_f1'], x['accuracy']), reverse=True)
        selected_result = eligible_candidates[0]
        selected_tau = selected_result['threshold']
        selection_status = "SELECTED"
        print(f"\n[SELECTION SUCCESS] Eligible threshold meeting Minor Recall >= {MIN_MINOR_RECALL*100:.0f}% found:")
        print(f"-> Selected Threshold tau*: {selected_tau:.3f}")
        print(f"   Validation Macro F1: {selected_result['macro_f1']:.4f}")
        print(f"   Validation Accuracy: {selected_result['accuracy']*100:.2f}%")
        print(f"   Validation Minor Recall: {selected_result['minor_recall']*100:.2f}%")
    else:
        selected_result = None
        selected_tau = None
        selection_status = "NO_ELIGIBLE_THRESHOLD"
        print(f"\n[SELECTION NOTICE] No threshold met the operational constraint of Minor Recall >= {MIN_MINOR_RECALL*100:.0f}%.")

    # Generate Visualization: Validation Threshold Curves
    plt.figure(figsize=(12, 6))
    taus = [r['threshold'] for r in results]
    plt.plot(taus, [r['minor_recall'] for r in results], marker='o', linewidth=2, label='Minor Recall (Class 0)', color='#e74c3c')
    plt.plot(taus, [r['mod_recall'] for r in results], marker='s', linewidth=2, label='Moderate Recall (Class 1)', color='#3498db')
    plt.plot(taus, [r['sev_recall'] for r in results], marker='^', linewidth=2, label='Severe Recall (Class 2)', color='#2ecc71')
    plt.plot(taus, [r['macro_f1'] for r in results], marker='D', linewidth=2.5, linestyle='--', label='Macro F1', color='#9b59b6')
    plt.plot(taus, [r['accuracy'] for r in results], marker='x', linewidth=2, linestyle=':', label='Overall Accuracy', color='#34495e')

    plt.axhline(y=MIN_MINOR_RECALL, color='red', linestyle=':', alpha=0.7, label=f'Minor Recall Constraint ({MIN_MINOR_RECALL*100:.0f}%)')
    
    if selected_tau is not None:
        plt.axvline(x=selected_tau, color='green', linestyle='-', alpha=0.8, linewidth=2, label=f'Selected tau* = {selected_tau:.3f}')

    plt.title('Validation Metrics vs. Minor Decision Threshold tau (Experiment 3)', fontsize=14, fontweight='bold')
    plt.xlabel('Minor Class Decision Threshold (tau)', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.xticks(THRESHOLDS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.35), ncol=3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Validation threshold curve plot saved to: {PLOT_PATH}")

    # Generate Confusion Matrix Plot if selected
    if selected_result is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=selected_result['confusion_matrix'], display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
        plt.title(f'Validation Confusion Matrix at tau* = {selected_tau:.3f}')
        plt.tight_layout()
        plt.savefig(CM_PATH, dpi=150)
        plt.close()
        print(f"Selected threshold validation confusion matrix saved to: {CM_PATH}")

    # Write Validation Summary Report
    with open(REPORT_PATH, 'w') as f:
        f.write("Experiment 3: Minor-Class Decision-Threshold Calibration Report\n")
        f.write("Model Evaluated: Experiment 1 MobileNetV2 (Phase 2 Fine-Tuned)\n")
        f.write("Evaluation Subset: 20% Validation Split ONLY (Zero Test Leakage)\n")
        f.write("=" * 70 + "\n\n")

        f.write("DECISION RULE:\n")
        f.write("  If p(Minor) >= tau -> predict Minor (0)\n")
        f.write("  Else -> argmax between Moderate (1) and Severe (2)\n\n")

        f.write(f"SELECTION CRITERIA:\n")
        f.write(f"  1. Constraint: Minor Recall >= {MIN_MINOR_RECALL*100:.0f}%\n")
        f.write(f"  2. Primary Metric: Maximum Validation Macro F1\n")
        f.write(f"  3. Tie-Breaker: Maximum Overall Validation Accuracy\n\n")

        if selected_result is not None:
            f.write(f"SELECTION RESULT: {selection_status}\n")
            f.write(f"  Selected Optimal Threshold (tau*): {selected_tau:.3f}\n")
            f.write(f"  Validation Macro F1: {selected_result['macro_f1']:.4f}\n")
            f.write(f"  Validation Accuracy: {selected_result['accuracy']*100:.2f}%\n")
            f.write(f"  Validation Minor Precision: {selected_result['minor_prec']:.4f}\n")
            f.write(f"  Validation Minor Recall: {selected_result['minor_recall']:.4f}\n")
            f.write(f"  Validation Minor F1: {selected_result['minor_f1']:.4f}\n")
            f.write(f"  Validation Moderate Recall: {selected_result['mod_recall']:.4f}\n")
            f.write(f"  Validation Severe Recall: {selected_result['sev_recall']:.4f}\n\n")

            f.write("VALIDATION CONFUSION MATRIX AT SELECTED THRESHOLD:\n")
            cm_sel = selected_result['confusion_matrix']
            f.write(f"                Pred Minor  Pred Mod  Pred Sev\n")
            f.write(f"  Actual Minor     {cm_sel[0,0]:<10} {cm_sel[0,1]:<9} {cm_sel[0,2]:<8}\n")
            f.write(f"  Actual Mod       {cm_sel[1,0]:<10} {cm_sel[1,1]:<9} {cm_sel[1,2]:<8}\n")
            f.write(f"  Actual Sev       {cm_sel[2,0]:<10} {cm_sel[2,1]:<9} {cm_sel[2,2]:<8}\n\n")
        else:
            f.write(f"SELECTION RESULT: {selection_status}\n")
            f.write(f"  No evaluated threshold in [0.250, 0.450] achieved Minor Recall >= {MIN_MINOR_RECALL*100:.0f}%.\n\n")

        f.write("FULL VALIDATION THRESHOLD SWEEP TABLE:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'tau':<7} | {'Minor Rec':<9} | {'Minor Prec':<10} | {'Minor F1':<8} | {'Mod Rec':<7} | {'Sev Rec':<7} | {'Macro F1':<8} | {'Accuracy':<8} | {'Eligible':<8}\n")
        f.write("-" * 90 + "\n")
        for r in results:
            elig_str = "YES" if r['eligible'] else "NO"
            f.write(f"{r['threshold']:<7.3f} | {r['minor_recall']*100:>8.2f}% | {r['minor_prec']*100:>9.2f}% | {r['minor_f1']:>8.4f} | {r['mod_recall']*100:>6.2f}% | {r['sev_recall']*100:>6.2f}% | {r['macro_f1']:>8.4f} | {r['accuracy']*100:>7.2f}% | {elig_str:<8}\n")
        f.write("-" * 90 + "\n")

    print(f"Validation report saved to: {REPORT_PATH}")
    print("\nExperiment 3 validation evaluation complete!")


if __name__ == '__main__':
    try:
        evaluate_threshold_sweep()
    except Exception as e:
        print(f"\nEvaluation failed with error:\n{e}", file=sys.stderr)
        sys.exit(1)
