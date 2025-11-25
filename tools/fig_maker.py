import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_fscore_support
)
import json
import os
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical




MODEL_PATH = "models/mobilenet_fold"
NUM_FOLDS = 5

CLASS_NAMES = ['Under', 'Good', 'Over']
CLASS_LABELS = ['100-under', '010-good', '001-over']

DATA_PATH = "../all_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
RANDOM_STATE = 42


def load_validation_data_for_fold(fold_no, data_prep):
    images, labels = data_prep.load(crop=False)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for idx, (train_idx, val_idx) in enumerate(kf.split(images, labels), 1):
        if idx == fold_no:
            x_val = images[val_idx]
            y_val = labels[val_idx]
            y_val_categorical = to_categorical(y_val, num_classes=3)
            return x_val, y_val, y_val_categorical
    
    return None, None, None


def plot_confusion_matrix_detailed(cm, class_names, fold_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, square=True, linewidths=2, cbar_kws={'label': 'Liczba próbek'})
    ax1.set_title(f'{fold_name} - Confusion Matrix (liczby)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prawdziwa klasa\n(co faktycznie jest)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predykcja modelu\n(co model przewidział)', fontsize=12, fontweight='bold')

    for i in range(len(class_names)):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                     edgecolor='green', lw=4))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, square=True, linewidths=2, vmin=0, vmax=100,
                cbar_kws={'label': 'Procent [%]'})
    ax2.set_title(f'{fold_name} - Confusion Matrix (procenty)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Prawdziwa klasa', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predykcja modelu', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano: {save_path}")
    plt.close()


def calculate_and_print_metrics(y_true, y_pred, class_names, fold_name):
    print(f"\n{'='*80}")
    print(f"METRYKI DLA {fold_name}")
    print('='*80)

    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    print("\n CLASSIFICATION REPORT:")
    print(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    
    print("\n SZCZEGÓŁOWE METRYKI PER KLASA:")
    print("-" * 80)
    print(f"{'Klasa':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:>12.4f} {recall[i]:>12.4f} "
              f"{f1[i]:>12.4f} {support[i]:>12.0f}")
    
    print("-" * 80)
    print(f"{'ŚREDNIA':<15} {np.mean(precision):>12.4f} {np.mean(recall):>12.4f} "
          f"{np.mean(f1):>12.4f} {np.sum(support):>12.0f}")
    print('='*80)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'accuracy': (y_pred == y_true).sum() / len(y_true)
    }


def explain_metrics_for_class(cm, class_idx, class_name):

    print(f"\n{'='*80}")
    print(f"WYJAŚNIENIE METRYK DLA KLASY: {class_name}")
    print('='*80)

    TP = cm[class_idx, class_idx]

    FP = cm[:, class_idx].sum() - TP

    FN = cm[class_idx, :].sum() - TP

    TN = cm.sum() - (TP + FP + FN)
    
    print(f"\n MACIERZ POMYŁEK DLA KLASY '{class_name}':")
    print("-" * 80)
    print(f"  True Positives (TP):   {TP:>4} - Poprawnie rozpoznane jako {class_name}")
    print(f"  False Positives (FP):  {FP:>4} - Błędnie sklasyfikowane jako {class_name}")
    print(f"  False Negatives (FN):  {FN:>4} - Prawdziwe {class_name}, ale NIE rozpoznane")
    print(f"  True Negatives (TN):   {TN:>4} - Poprawnie rozpoznane jako NIE-{class_name}")

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    print(f"\n PRECISION (Precyzja) dla '{class_name}':")
    print(f"   = TP / (TP + FP)")
    print(f"   = {TP} / ({TP} + {FP})")
    print(f"   = {precision:.4f} ({precision*100:.2f}%)")
    print(f"\n    INTERPRETACJA:")
    print(f"   Gdy model klasyfikuje obraz jako '{class_name}',")
    print(f"   w {precision*100:.1f}% przypadków ma rację.")
    
    print(f"\n RECALL (Czułość) dla '{class_name}':")
    print(f"   = TP / (TP + FN)")
    print(f"   = {TP} / ({TP} + {FN})")
    print(f"   = {recall:.4f} ({recall*100:.2f}%)")
    print(f"\n    INTERPRETACJA:")
    print(f"   Z wszystkich prawdziwych '{class_name}' w danych,")
    print(f"   model znalazł {recall*100:.1f}% z nich.")
    
    print(f"\nF1-SCORE (Średnia harmoniczna P i R):")
    print(f"   = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   = 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"   = {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n    INTERPRETACJA:")
    print(f"   F1-Score balansuje Precision i Recall.")
    print(f"   Wartość {f1*100:.1f}% oznacza {'doskonały' if f1 > 0.95 else 'bardzo dobry' if f1 > 0.85 else 'dobry' if f1 > 0.75 else 'średni'} balans.")

    print(f"\n ANALIZA BŁĘDÓW:")
    print("-" * 80)
    if FP > 0:
        print(f"   {FP} obrazów zostało BŁĘDNIE sklasyfikowanych jako '{class_name}':")
        for i in range(len(cm)):
            if i != class_idx and cm[i, class_idx] > 0:
                print(f"      - {cm[i, class_idx]} obrazów było faktycznie '{CLASS_NAMES[i]}'")
    
    if FN > 0:
        print(f"\n   {FN} prawdziwych '{class_name}' zostało PRZEOCZONYCH:")
        for i in range(len(cm)):
            if i != class_idx and cm[class_idx, i] > 0:
                print(f"      - {cm[class_idx, i]} zostało pomylonych z '{CLASS_NAMES[i]}'")
    
    print('='*80)

def evaluate_all_folds():
    print("\n" + "="*80)
    print("ROZPOCZYNAM KOMPLEKSOWĄ EWALUACJĘ MODELI")
    print("="*80)

    import sys
    sys.path.append('/Users\MatAnd\Github\FlowRateAnalyzer')
    import preparing

    data_prep = preparing.DataPrep(
        path=DATA_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE
    )

    output_dir = '/confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    all_confusion_matrices = []

    for fold_no in range(1, NUM_FOLDS + 1):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold_no}/{NUM_FOLDS}")
        print(f"{'#'*80}\n")

        model_path = f"{MODEL_PATH}{fold_no}.h5"
        if not os.path.exists(model_path):
            print(f"Model nie znaleziony: {model_path}")
            print(f"Sprawdź ścieżkę lub pomiń ten fold")
            continue
        
        print(f"Wczytuję model: {model_path}")
        model = load_model(model_path)

        print(f"Przygotowuję dane walidacyjne...")
        x_val, y_val, y_val_cat = load_validation_data_for_fold(fold_no, data_prep)
        
        if x_val is None:
            print(f"Nie udało się wczytać danych dla folda {fold_no}")
            continue
        
        print(f"   Liczba próbek walidacyjnych: {len(x_val)}")
        print(f"   Rozkład klas: Under={np.sum(y_val==0)}, Good={np.sum(y_val==1)}, Over={np.sum(y_val==2)}")

        print(f"Generuję predykcje...")
        y_pred_proba = model.predict(x_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        print(f"Tworzę confusion matrix...")
        cm = confusion_matrix(y_val, y_pred)
        all_confusion_matrices.append(cm)

        cm_path = f'{output_dir}/confusion_matrix_fold{fold_no}.png'
        plot_confusion_matrix_detailed(cm, CLASS_NAMES, f"Fold {fold_no}", cm_path)

        metrics = calculate_and_print_metrics(y_val, y_pred, CLASS_NAMES, f"Fold {fold_no}")
        all_metrics.append(metrics)

        for class_idx, class_name in enumerate(CLASS_NAMES):
            explain_metrics_for_class(cm, class_idx, class_name)

    print(f"\n{'='*80}")
    print("PODSUMOWANIE WSZYSTKICH FOLDÓW")
    print('='*80)
    
    if len(all_metrics) > 0:

        avg_precision = np.mean([m['precision'] for m in all_metrics], axis=0)
        avg_recall = np.mean([m['recall'] for m in all_metrics], axis=0)
        avg_f1 = np.mean([m['f1'] for m in all_metrics], axis=0)
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        
        print(f"\nŚREDNIE METRYKI ZE WSZYSTKICH FOLDÓW:")
        print("-" * 80)
        print(f"{'Klasa':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        print("-" * 80)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name:<15} {avg_precision[i]:>12.4f} {avg_recall[i]:>12.4f} {avg_f1[i]:>12.4f}")
        print("-" * 80)
        print(f"{'ŚREDNIA':<15} {np.mean(avg_precision):>12.4f} {np.mean(avg_recall):>12.4f} {np.mean(avg_f1):>12.4f}")
        print(f"\n{'Overall Accuracy':<15} {avg_accuracy:>12.4f}")
        print('='*80)

        avg_cm = np.mean(all_confusion_matrices, axis=0).astype(int)
        cm_path = f'{output_dir}/confusion_matrix_AVERAGE.png'
        plot_confusion_matrix_detailed(avg_cm, CLASS_NAMES, "ŚREDNIA ze wszystkich foldów", cm_path)

        results = {
            'average_metrics': {
                'precision': [float(p) for p in avg_precision],
                'recall': [float(r) for r in avg_recall],
                'f1_score': [float(f) for f in avg_f1],
                'accuracy': float(avg_accuracy)
            },
            'per_fold_metrics': []
        }
        
        for i, metrics in enumerate(all_metrics, 1):
            results['per_fold_metrics'].append({
                'fold': i,
                'precision': [float(p) for p in metrics['precision']],
                'recall': [float(r) for r in metrics['recall']],
                'f1_score': [float(f) for f in metrics['f1']],
                'accuracy': float(metrics['accuracy'])
            })
        
        with open('../outputs/detailed_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n✓ Wszystkie wyniki zapisane w /mnt/user-data/outputs/")
        print("  - confusion_matrices/ - wszystkie confusion matrices")
        print("  - metrics_explanation.png - wyjaśnienie metryk")
        print("  - detailed_metrics.json - szczegółowe metryki w JSON")
    
    print(f"\n{'='*80}")
    print("EWALUACJA ZAKOŃCZONA!")
    print('='*80)



if __name__ == "__main__":
    evaluate_all_folds()