import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import preparing

NUM_FOLDS = 5
RANDOM_STATE = 42
CLASS_NAMES = ["Under", "Good", "Over"]

MODEL_DIR = r"C:/Users/MatAnd/Github/FlowRateAnalyzer/models/model_pp"
JSON_DIR  = r"C:\Users\MatAnd\Github\FlowRateAnalyzer\json\model_pp"

MODEL="model_pp_fold"

DATA_PATH = "../all_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

OUTPUT_CM = "./confusion_matrices/model"
OUTPUT_LC = "./learning_curves/model"
RESULTS_TXT = "cv_results_model.txt"

os.makedirs(OUTPUT_CM, exist_ok=True)
os.makedirs(OUTPUT_LC, exist_ok=True)



def load_validation_data_for_fold(fold_no, data_prep):
    images, labels = data_prep.load(crop=False)

    kf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for idx, (train_idx, val_idx) in enumerate(kf.split(images, labels), 1):
        if idx == fold_no:
            return images[val_idx], labels[val_idx], to_categorical(labels[val_idx], 3)

    return None, None, None


def plot_conf_matrix(cm, fold_no):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - Fold {fold_no}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    path = os.path.join(OUTPUT_CM, f"confusion_matrix_fold{fold_no}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✓ Zapisano CM fold {fold_no}: {path}")



def plot_learning_curves(fold_no, history):
    loss = history["loss"]
    val_loss = history["val_loss"]
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 6))

    # LOSS
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title(f"Fold {fold_no} - Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()

    # ACCURACY
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title(f"Fold {fold_no} - Accuracy")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()

    path = os.path.join(OUTPUT_LC, f"learning_curve_fold{fold_no}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✓ Zapisano learning curve fold {fold_no}: {path}")




def evaluate_all_folds():

    data_prep = preparing.DataPrep(
        path=DATA_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE
    )

    all_metrics = []
    all_cms = []

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n========== FOLD {fold} ==========")

        model_path = os.path.join(MODEL_DIR, f"{MODEL}{fold}.h5")
        if not os.path.exists(model_path):
            print(f"Brak modelu: {model_path}")
            continue

        model = load_model(model_path)
        print(f"✓ Wczytano model: {model_path}")


        x_val, y_val, y_val_cat = load_validation_data_for_fold(fold, data_prep)


        loss, acc = model.evaluate(x_val, y_val_cat, verbose=0)
        y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

        cm = confusion_matrix(y_val, y_pred)
        plot_conf_matrix(cm, fold)
        all_cms.append(cm)

        precision_macro = precision_score(y_val, y_pred, average="macro")
        recall_macro = recall_score(y_val, y_pred, average="macro")
        f1_macro = f1_score(y_val, y_pred, average="macro")

        all_metrics.append({
            "loss": loss,
            "accuracy": acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro
        })

        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision macro: {precision_macro:.4f}")
        print(f"Recall macro:    {recall_macro:.4f}")
        print(f"F1 macro:        {f1_macro:.4f}")

        json_path = os.path.join(JSON_DIR, f"{MODEL}{fold}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                history = json.load(f)
            plot_learning_curves(fold, history)
        else:
            print(f"Brak JSON treningowego: {json_path}")




    mean_cm = np.mean(all_cms, axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Mean Confusion Matrix (Średnia z 5 foldów)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    mean_cm_path = os.path.join(OUTPUT_CM, "confusion_matrix_mean.png")
    plt.tight_layout()
    plt.savefig(mean_cm_path, dpi=300)
    plt.close()

    print(f"\n Zapisano średnią macierz pomyłek: {mean_cm_path}\n")



    losses = [m["loss"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    precs = [m["precision_macro"] for m in all_metrics]
    recalls = [m["recall_macro"] for m in all_metrics]
    f1s = [m["f1_macro"] for m in all_metrics]

    with open(RESULTS_TXT, "w") as f:

        f.write("===== 5-FOLD CROSS VALIDATION RESULTS =====\n\n")

        for i, m in enumerate(all_metrics, 1):
            f.write(f"--- Fold {i} ---\n")
            f.write(f"Loss: {m['loss']:.4f}\n")
            f.write(f"Accuracy: {m['accuracy']:.4f}\n")
            f.write(f"Precision macro: {m['precision_macro']:.4f}\n")
            f.write(f"Recall macro:    {m['recall_macro']:.4f}\n")
            f.write(f"F1 macro:        {m['f1_macro']:.4f}\n\n")

        f.write("\n===== SUMMARY (MEAN ± STD) =====\n\n")
        f.write(f"Loss:            {np.mean(losses):.4f} ± {np.std(losses):.4f}\n")
        f.write(f"Accuracy:        {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Precision macro: {np.mean(precs):.4f} ± {np.std(precs):.4f}\n")
        f.write(f"Recall macro:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1 macro:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")

    print(f"✓ Wyniki zapisane do: {RESULTS_TXT}")


if __name__ == "__main__":
    evaluate_all_folds()
