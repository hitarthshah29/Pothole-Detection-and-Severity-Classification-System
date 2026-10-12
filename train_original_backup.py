"""
train.py

This script trains an experimental three-class image classifier (Minor, Moderate, Severe)
using MobileNetV2 transfer learning. It uses a controlled two-phase training experiment 
with data augmentation enabled, but class weights disabled.

IMPORTANT SCIENTIFIC LIMITATION:
The labels used in this model are proxy labels based on bounding-box area, not verified 
real-world severity labels. The test accuracy represents how well the model predicts these 
proxy classes, NOT its accuracy at predicting real-world pothole depth, structural damage, 
or road safety.
"""

import os
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Configurations
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

TRAIN_DIR = 'data/processed/train'
TEST_DIR = 'data/processed/test'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'pothole_severity_model.keras')
PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, 'model_phase1.keras')
PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, 'model_phase2.keras')

CLASS_NAMES_PATH = os.path.join(MODELS_DIR, 'class_names.json')
REPORT_PATH = os.path.join(OUTPUTS_DIR, 'training_report.txt')
PLOT_PATH = os.path.join(OUTPUTS_DIR, 'training_history.png')
CM_PATH = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')

def main():
    print("Starting model training pipeline...")

    # 1. Verify Dataset Structure
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Prepared dataset not found. Please run prepare_data.py first.")
        return

    # 2. Data Generators
    # Data Augmentation only applied to training
    train_datagen_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1
    )
    
    # Validation datagen should not have augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print("Loading training data...")
    train_generator = train_datagen_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    # Save class names mapping
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {CLASS_NAMES_PATH}")

    # Note: Class weights are explicitly disabled for this controlled experiment.

    # 3. Build Initial Model (Transfer Learning with MobileNetV2)
    print("Building model based on MobileNetV2...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    # Freeze the base model initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Phase 1: Initial Training
    print(f"\nPhase 1: Training top layers for up to {INITIAL_EPOCHS} epochs...")
    callbacks_initial = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=PHASE1_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_initial = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_initial
    )

    # Record best val_loss for phase 1
    best_phase1_val_loss = min(history_initial.history['val_loss'])
    best_phase1_idx = history_initial.history['val_loss'].index(best_phase1_val_loss)
    best_phase1_epoch = history_initial.epoch[best_phase1_idx] + 1
    print(f"Phase 1 Best Validation Loss: {best_phase1_val_loss:.4f} at epoch {best_phase1_epoch}")

    # 5. Phase 2: Fine-tuning
    print("\nPhase 2: Fine-tuning the top layers of the base model...")
    # The best weights from Phase 1 are already restored because restore_best_weights=True
    
    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Ensure BatchNormalization layers remain frozen
    for layer in base_model.layers[-30:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # Recompile with a small learning rate (1e-5)
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_finetune = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=PHASE2_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_finetune = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history_initial.epoch),
        validation_data=val_generator,
        callbacks=callbacks_finetune
    )

    # Record best val_loss for phase 2
    best_phase2_val_loss = min(history_finetune.history['val_loss'])
    best_phase2_idx = history_finetune.history['val_loss'].index(best_phase2_val_loss)
    best_phase2_epoch = history_finetune.epoch[best_phase2_idx] + 1
    print(f"Phase 2 Best Validation Loss: {best_phase2_val_loss:.4f} at epoch {best_phase2_epoch}")

    # Combine histories for plotting
    acc = history_initial.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_initial.history['loss'] + history_finetune.history['loss']
    val_loss = history_initial.history['val_loss'] + history_finetune.history['val_loss']

    # 6. Select the Overall Best Model
    print("\nComparing Phase 1 and Phase 2 models...")
    if best_phase1_val_loss <= best_phase2_val_loss:
        print("Phase 1 model is better (or equal). Selecting Phase 1.")
        shutil.copy2(PHASE1_MODEL_PATH, FINAL_MODEL_PATH)
        selected_phase = "Phase 1"
    else:
        print("Phase 2 model is better. Selecting Phase 2.")
        shutil.copy2(PHASE2_MODEL_PATH, FINAL_MODEL_PATH)
        selected_phase = "Phase 2"

    # 7. Evaluate on Test Set
    print(f"\nEvaluating the OVERALL BEST model ({selected_phase}) on test set...")
    best_model = load_model(FINAL_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate Predictions for Metrics and Confusion Matrix
    print("Generating predictions...")
    test_generator.reset()
    predictions = best_model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    class_labels = list(class_indices.keys())

    # 8. Save Outputs
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Plot Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.axvline(x=len(history_initial.epoch)-1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=len(history_initial.epoch)-1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Training history plot saved to {PLOT_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Test Set')
    plt.savefig(CM_PATH)
    plt.close()
    print(f"Confusion matrix saved to {CM_PATH}")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    
    with open(REPORT_PATH, 'w') as f:
        f.write("Pothole Classification Controlled Experiment Report\n")
        f.write("===================================================\n\n")
        f.write("IMPORTANT SCIENTIFIC LIMITATION:\n")
        f.write("The classes (Minor, Moderate, Severe) are PROXY labels based solely on 2D bounding-box area.\n")
        f.write("They do NOT represent medically or engineering-verified real-world severity labels (such as pothole depth, structural damage, or risk to vehicles).\n")
        f.write("The accuracy below only indicates how well the model predicts these proxy visual categories, NOT real-world physical severity.\n\n")
        
        f.write("Validation Tracking:\n")
        f.write(f"- Best Phase 1 Val Loss: {best_phase1_val_loss:.4f} (Epoch {best_phase1_epoch})\n")
        f.write(f"- Best Phase 2 Val Loss: {best_phase2_val_loss:.4f} (Epoch {best_phase2_epoch})\n")
        f.write(f"- Selected Model: {selected_phase}\n\n")

        f.write("Historical Comparisons:\n")
        f.write("- Original Baseline Test Accuracy: 62.41%\n")
        f.write("- Class-Weighted Test Accuracy: 59.40%\n")
        f.write(f"- NEW Experiment Test Accuracy: {test_accuracy*100:.2f}%\n\n")
        
        f.write("Overall Test Results:\n")
        f.write(f"- Test Loss: {test_loss:.4f}\n")
        f.write(f"- Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
        
        f.write("Classification Report (Per-Class Metrics):\n")
        f.write(report)
        
    print(f"Training report saved to {REPORT_PATH}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nTraining Failed! Error details:\n{e}")
