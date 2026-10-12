"""
train_experiment_2.py

Controlled Experiment 2: Pothole Severity Classification (Minor, Moderate, Severe)
using MobileNetV2 Transfer Learning with Square-Root Damped, Moderate-Anchored Class Weighting.

Objective:
Alleviate the high Moderate-to-Minor confusion observed in Experiment 1 while maintaining
strong Minor-class detection sensitivity and protecting Severe-class performance.

Weighting Formulation:
1. Raw balanced inverse weights: w_raw_j = N_train / (K * n_j)
2. Sub-linear power damping (alpha = 0.5): w_damped_j = sqrt(w_raw_j)
3. Moderate anchor normalization: w_final_j = w_damped_j / w_damped_Moderate
   - Anchors Moderate at 1.0 (eliminates the "cheap error" penalty discount)
   - Calibrates Minor at ~1.80x (down from 3.24x in Experiment 1)
   - Protects Severe at ~1.11x (> 1.0 to prevent severe leakage)

Key Experiment Setup:
- Architecture: MobileNetV2 pretrained on ImageNet
- Class Order: Minor = 0, Moderate = 1, Severe = 2
- Data Augmentation: Applied strictly to training split (flips, rotation, zoom)
- Validation Split: 20% of training data without data augmentation
- Model Selection: Minimum unweighted validation loss across Phase 1 and Phase 2
- Isolated Paths: All outputs saved exclusively in models/experiments/experiment_2/
  and outputs/experiments/experiment_2/
- Anti-Leakage: All weights computed dynamically from training subset; test set evaluated
  only once after model selection.

SCIENTIFIC LIMITATION NOTE:
The severity categories (Minor, Moderate, Severe) are proxy labels derived from
relative bounding-box area in 2D images. They do not represent physical depth,
sub-surface roadbed damage, or engineering-certified structural severity.
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Configurations
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
CLASSES = ['Minor', 'Moderate', 'Severe']

# Paths resolved relative to script location for robust execution from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed', 'train')
TEST_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed', 'test')

# Dedicated Experiment 2 directories
MODELS_EXP2_DIR = os.path.join(SCRIPT_DIR, 'models', 'experiments', 'experiment_2')
OUTPUTS_EXP2_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'experiments', 'experiment_2')

# Model output paths inside models/experiments/experiment_2/
FINAL_MODEL_PATH = os.path.join(MODELS_EXP2_DIR, 'pothole_severity_model.keras')
PHASE1_MODEL_PATH = os.path.join(MODELS_EXP2_DIR, 'model_phase1.keras')
PHASE2_MODEL_PATH = os.path.join(MODELS_EXP2_DIR, 'model_phase2.keras')
CLASS_NAMES_PATH = os.path.join(MODELS_EXP2_DIR, 'class_names.json')

# Output artifact paths inside outputs/experiments/experiment_2/
REPORT_PATH = os.path.join(OUTPUTS_EXP2_DIR, 'training_report.txt')
PLOT_PATH = os.path.join(OUTPUTS_EXP2_DIR, 'training_history.png')
CM_PATH = os.path.join(OUTPUTS_EXP2_DIR, 'confusion_matrix.png')


def verify_prerequisites():
    """Verifies dataset directories exist before training."""
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Prepared training dataset not found at '{TRAIN_DIR}'.")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Prepared test dataset not found at '{TEST_DIR}'.")
    
    # Ensure dedicated experiment 2 directories exist
    os.makedirs(MODELS_EXP2_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_EXP2_DIR, exist_ok=True)


def build_data_generators():
    """
    Creates training (augmented), validation (unaugmented), and test data generators.
    Enforces class ordering: Minor = 0, Moderate = 1, Severe = 2.
    """
    train_datagen_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print("Loading training data (with augmentation)...")
    train_generator = train_datagen_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    print("Loading validation data (unaugmented)...")
    val_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator, test_generator


def compute_experiment_2_class_weights(train_generator):
    """
    Computes square-root damped, Moderate-anchored class weights directly from the
    actual training subset distribution:
    1. Raw balanced weights: w_raw_j = N_train / (K * n_j)
    2. Square-root damping: w_damped_j = sqrt(w_raw_j)
    3. Moderate anchor normalization: w_final_j = w_damped_j / w_damped_Moderate
    """
    train_classes = train_generator.classes
    classes = np.unique(train_classes)
    
    # 1. Compute raw inverse-frequency balanced weights
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_classes
    )
    raw_dict = {int(c): float(w) for c, w in zip(classes, raw_weights)}
    
    # 2. Square-root damping
    damped_dict = {c: float(np.sqrt(w)) for c, w in raw_dict.items()}
    
    # 3. Dynamic lookup of Moderate class index and anchor normalization
    moderate_idx = train_generator.class_indices['Moderate']
    moderate_anchor = damped_dict[moderate_idx]
    
    final_weights = {c: float(w / moderate_anchor) for c, w in damped_dict.items()}
    return final_weights, raw_dict


def build_model():
    """
    Builds the transfer learning model with MobileNetV2 base (frozen)
    and custom classification head for 3 classes.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.2, name='dropout')(x)
    predictions = Dense(3, activation='softmax', name='severity_output')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='pothole_mobilenetv2_severity_exp2')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


def train_pipeline():
    print("=" * 75)
    print("Starting Pothole Severity Controlled Experiment 2 (Damped & Anchored Weights)")
    print("=" * 75)

    # 1. Verify Directories
    verify_prerequisites()

    # 2. Build Generators
    train_generator, val_generator, test_generator = build_data_generators()

    # Verify and save class mapping
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    print(f"\nClass Mapping verified: {class_indices}")
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to: {CLASS_NAMES_PATH}")

    # 3. Calculate Damped & Anchored Class Weights
    class_weight_dict, raw_dict = compute_experiment_2_class_weights(train_generator)
    print("\nActual Training Subset Distribution & Weight Computation:")
    labels, counts = np.unique(train_generator.classes, return_counts=True)
    for lbl, cnt in zip(labels, counts):
        name = class_names[lbl]
        print(f" - {name} (class {lbl}): {cnt} samples | Raw: {raw_dict[lbl]:.4f} | Exp 2 Damped & Anchored: {class_weight_dict[lbl]:.4f}")

    # 4. Model Construction
    print("\nConstructing MobileNetV2 transfer learning model...")
    model, base_model = build_model()

    # 5. Phase 1: Train Top Classification Head
    print(f"\n--- Phase 1: Training top layers (up to {INITIAL_EPOCHS} epochs) with Exp 2 weights ---")
    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=PHASE1_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_phase1 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict
    )

    best_phase1_val_loss = min(history_phase1.history['val_loss'])
    best_phase1_idx = history_phase1.history['val_loss'].index(best_phase1_val_loss)
    best_phase1_epoch = history_phase1.epoch[best_phase1_idx] + 1
    print(f"\n[Phase 1] Best Validation Loss: {best_phase1_val_loss:.4f} at epoch {best_phase1_epoch}")

    # 6. Phase 2: Fine-Tuning Base Model Top Layers
    print(f"\n--- Phase 2: Fine-tuning MobileNetV2 top layers (up to {FINE_TUNE_EPOCHS} epochs) ---")
    base_model.trainable = True

    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Freeze BatchNormalization layers to preserve moving statistics
    for layer in base_model.layers[-30:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=PHASE2_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_phase2 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history_phase1.epoch),
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict
    )

    best_phase2_val_loss = min(history_phase2.history['val_loss'])
    best_phase2_idx = history_phase2.history['val_loss'].index(best_phase2_val_loss)
    best_phase2_epoch = history_phase2.epoch[best_phase2_idx] + 1
    print(f"\n[Phase 2] Best Validation Loss: {best_phase2_val_loss:.4f} at epoch {best_phase2_epoch}")

    # 7. Select Overall Best Model using Validation Loss
    print("\n--- Model Selection based on Validation Loss ---")
    print(f"Phase 1 best val_loss: {best_phase1_val_loss:.4f}")
    print(f"Phase 2 best val_loss: {best_phase2_val_loss:.4f}")

    if best_phase1_val_loss <= best_phase2_val_loss:
        selected_phase = "Phase 1"
        selected_val_loss = best_phase1_val_loss
        shutil.copy2(PHASE1_MODEL_PATH, FINAL_MODEL_PATH)
        print(f"-> Phase 1 model selected and copied to {FINAL_MODEL_PATH}")
    else:
        selected_phase = "Phase 2"
        selected_val_loss = best_phase2_val_loss
        shutil.copy2(PHASE2_MODEL_PATH, FINAL_MODEL_PATH)
        print(f"-> Phase 2 model selected and copied to {FINAL_MODEL_PATH}")

    # 8. Evaluation on Test Set (strictly post-selection)
    print(f"\n--- Evaluating Best Model ({selected_phase}) on Unseen Test Set ---")
    best_model = load_model(FINAL_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    test_generator.reset()
    predictions = best_model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = [class_names[i] for i in range(len(class_names))]

    # 9. Generate and Save Outputs
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy', linewidth=2)
    plt.plot(val_acc, label='Val Accuracy', linewidth=2)
    plt.axvline(x=len(history_phase1.epoch) - 1, color='r', linestyle='--', label='Fine-Tuning Start')
    plt.title('Experiment 2 Accuracy (Damped & Anchored)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss', linewidth=2)
    plt.plot(val_loss, label='Val Loss', linewidth=2)
    plt.axvline(x=len(history_phase1.epoch) - 1, color='r', linestyle='--', label='Fine-Tuning Start')
    plt.title('Experiment 2 Loss (Damped & Anchored)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Training plot saved to: {PLOT_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title('Test Set Confusion Matrix (Experiment 2)')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {CM_PATH}")

    # Classification Report
    report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print("\nClassification Report:")
    print(report_str)

    with open(REPORT_PATH, 'w') as f:
        f.write("Pothole Classification Controlled Experiment 2 Report\n")
        f.write("Strategy: Square-Root Damped, Moderate-Anchored Class Weighting\n")
        f.write("=" * 65 + "\n\n")
        f.write("EXPERIMENT 2 OBJECTIVE:\n")
        f.write("Reduce Moderate-to-Minor misclassifications observed in Experiment 1 while\n")
        f.write("maintaining strong Minor recall and preserving Severe precision and recall.\n\n")
        f.write("SCIENTIFIC LIMITATION:\n")
        f.write("The classes (Minor, Moderate, Severe) are PROXY labels based on 2D bounding-box area.\n")
        f.write("They do NOT measure physical depth, structural road damage, or engineering road safety.\n\n")
        f.write("CLASS WEIGHTS APPLIED (CALCULATED FROM TRAINING SUBSET):\n")
        for lbl, weight in class_weight_dict.items():
            f.write(f"- Class {lbl} ({class_names[lbl]}): {weight:.4f} (Raw: {raw_dict[lbl]:.4f})\n")
        f.write("\nVALIDATION TRACKING & MODEL SELECTION:\n")
        f.write(f"- Best Phase 1 Val Loss: {best_phase1_val_loss:.4f} (Epoch {best_phase1_epoch})\n")
        f.write(f"- Best Phase 2 Val Loss: {best_phase2_val_loss:.4f} (Epoch {best_phase2_epoch})\n")
        f.write(f"- Selected Model: {selected_phase} (Val Loss: {selected_val_loss:.4f})\n\n")
        f.write("HISTORICAL BENCHMARKS (TEST SET):\n")
        f.write("- Baseline (Unweighted) Test Accuracy: 60.90% (Minor Recall: 0.2222, Mod Recall: 0.6786, Sev Recall: 0.7400)\n")
        f.write("- Experiment 1 (Raw Inverse Weights) Test Accuracy: 59.40% (Minor Recall: 0.5185, Mod Recall: 0.5179, Sev Recall: 0.7200)\n")
        f.write(f"- Experiment 2 (Damped & Anchored) Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
        f.write("OVERALL TEST RESULTS:\n")
        f.write(f"- Test Loss: {test_loss:.4f}\n")
        f.write(f"- Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(report_str + "\n")

    print(f"Comprehensive report saved to: {REPORT_PATH}")
    print("\nExperiment 2 pipeline completed successfully!")


if __name__ == '__main__':
    try:
        train_pipeline()
    except Exception as e:
        print(f"\nExperiment 2 failed with error:\n{e}", file=sys.stderr)
        sys.exit(1)
