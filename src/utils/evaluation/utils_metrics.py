# src/evaluation/utils_metrics.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from typing import List, Tuple, Dict, Any

CLASSES = ["Plaga", "Sana"] 

def plot_confusion(cm: np.ndarray, class_names: List[str], save_path: str, title: str = "Matriz de Confusión") -> None:
    """
    Dibuja y guarda la Matriz de Confusión con anotaciones de recuento.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Verdadera")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    # Anotar los valores en el centro de cada celda
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color=color, fontsize=12)
            
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close('all')
    print(f"✅ Matriz de Confusión guardada en: {save_path}")

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Calcula la matriz de confusión y el reporte de clasificación (precision, recall, f1-score).
    """
    cm = confusion_matrix(y_true, y_pred)
    # output_dict=True permite serializar el reporte a JSON
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    return report_dict, cm

def save_report_and_plot_cm(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str], 
    results_dir: str, 
    model_name: str, 
    threshold: float = 0.5
) -> None:
    """
    Genera el reporte, guarda el JSON y plotea la Matriz de Confusión.
    """
    report_dict, cm = generate_classification_report(y_true, y_pred, class_names)

    # 1. Guardar el reporte JSON
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Genera un string de umbral (e.g., t050)
    umbral_str = f"t{int(threshold * 100):02d}" 
    
    report_filename = f"report_{model_name}_{timestamp}_{umbral_str}.json"
    report_path = os.path.join(results_dir, report_filename)
    
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    
    print(f"\n✅ Reporte de Clasificación guardado en: {report_path}")

    # 2. Plotear y guardar la Matriz de Confusión
    plot_filename = report_filename.replace('.json', '_confusion.png')
    plot_path = os.path.join(results_dir, plot_filename)
    plot_confusion(cm, class_names, plot_path, title=f"Matriz de Confusión ({model_name}, t={threshold})")

    # 3. Imprimir el resumen
    print("\n--- RESUMEN DEL REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))