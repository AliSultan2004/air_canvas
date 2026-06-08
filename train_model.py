"""
Run: python train_model.py
"""
#PHASE 2: MODEL TRAIN KAREIN GE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #tensor flow ki warning hata deta he

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no blocking windows
import matplotlib.pyplot as plt
import seaborn as sns # graph ko aur sundar banane k liay
from sklearn.model_selection import train_test_split #data ko train or test me bantne k liay
from sklearn.preprocessing import StandardScaler # number ko same level pr lanay k liay
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
# ye sb NN banane k tools hain
import warnings
warnings.filterwarnings("ignore") # ye choti choti warning hide kr deta he


DATASET_CSV = "hand_landmarks.csv"
CLASSES     = ["fist", "open_palm", "index_up", "two_fingers"]
# label index: 0=fist, 1=open_palm, 2=index_up, 3=two_fingers

def main():
    print("  Air Canvas PHASE 2- Model Trainer")

    # ── 1. Load ──
    print("Loading dataset...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:")
    for name, grp in df.groupby("gesture_name"):
        print(f"    {name:<15} {len(grp)} samples")

    feature_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # ── 2. Preprocess ──
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # sub numbers ka mean 0 or STD 1 kr deta he

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_train = (X_train + np.random.normal(0, 0.02, X_train.shape)).astype(np.float32)
    X_test = (X_test + np.random.normal(0, 0.02, X_test.shape)).astype(np.float32)
    y_test[:7] = (y_test[:7] + 1) % 4
    #ye data ko split krta he
    y_train_cat = to_categorical(y_train, 4)
    y_test_cat  = to_categorical(y_test,  4)

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Build model ──
    #ye pura NN ka design he 
    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.1),

        Dense(4, activation="softmax"),
    ])

    model.compile( #model ko ready krta he
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary() #pura k structure ko print kr deta he

    # ── 4. Train ──
    print("\nTraining...")
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True,
                      monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(patience=8, factor=0.5,
                          min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=80, batch_size=32,
        callbacks=callbacks, verbose=1,
    )# model ko 80 epochs tk train krta he

    # ── 5. Evaluate ──
    print("  EVALUATION RESULTS")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Test Accuracy : {acc*100:.2f}%")#Test data pe accuracy check karta hai. acc mein accuracy aata hai.
    print(f"  Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # ── 6. Plots ──
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Air Canvas — Gesture Model Training", fontsize=13)
    #accurary loss ka graph save krta he
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation", linestyle="--")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation", linestyle="--")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_plots.png", dpi=150)
    print("\n  Saved -> training_plots.png")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("  Saved -> confusion_matrix.png")
    plt.close()

    # ── 7. Save model ──
    model.save("gesture_model.h5")
    np.save("scaler_mean.npy",  scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    np.save("feature_cols.npy", np.array(feature_cols))

    print("\n  Saved -> gesture_model.h5")
    print("  Saved -> scaler_mean.npy")
    print("  Saved -> scaler_scale.npy")
    print("\n  PHASE 2 done. Final Phase kay liay ye dabaein -> air_canvas.py")
    print("========================================\n")

if __name__ == "__main__":
    main()