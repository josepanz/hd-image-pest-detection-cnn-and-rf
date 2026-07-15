# src/evaluation/utils_metrics.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from datetime import datetime
from typing import List, Tuple, Dict, Any

CLASSES = ["Plaga", "Sana"] 

def plot_confusion(cm: np.ndarray, class_names: List[str], save_path: str, name: str, title: str = "Matriz de Confusión") -> None:
    """
    Dibuja y guarda la Matriz de Confusión con anotaciones de recuento.

    BUG CORREGIDO: llamaba a plt.show() (bloqueante) seguido de plt.close('all') -
    con el backend interactivo por defecto (fuera de tests, que fuerzan 'Agg'), esto
    congelaba la ejecución hasta cerrar la ventana a mano. Se llama una vez por cada
    train.py (post_train_val) y una vez por cada evaluate.py: para correr los 6
    modelos x 3 umbrales de la bitácora de forma desatendida esto colgaba el script
    en cada corrida. Ahora usa show(block=False)+pause() para dibujarla sin bloquear,
    y ya no se cierra: queda visible en pantalla (plt.close('all') además cerraría
    también los gráficos de entrenamiento que haya abiertos de antes).
    """
    plot_path = os.path.join(save_path, name)
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
    plt.savefig(plot_path)
    plt.show(block=False)
    plt.pause(0.001)
    print(f"✅ Matriz de Confusión guardada en: {plot_path}")

    print('Matriz de confusión:')
    print(cm)

    name = name.replace('.png', '.md')
    report_path_md = os.path.join(save_path, f"{name}")
    with open(report_path_md, "w", encoding="utf-8") as f:
      f.write(f"# Matriz de confusión\n\n")
      f.write(f"- **Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(f"- **Modelo:** {name}\n")
      f.write("```text\n")
      f.write(str(cm))
      f.write("\n```\n\n")
      f.write("---\n")
      f.write("*Generado automáticamente por el sistema de detección de plagas.*")

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
    plot_confusion(cm, class_names, results_dir, name=plot_filename, title=f"Matriz de Confusión ({model_name}, t={threshold})")

    # 3. Imprimir el resumen
    text_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n--- RESUMEN DEL REPORTE DE CLASIFICACIÓN ---")
    print(text_report)

    report_path_md = os.path.join(results_dir, f"report_table_{model_name}_{timestamp}_{umbral_str}.md")
    with open(report_path_md, "w", encoding="utf-8") as f:
      f.write(f"# Reporte de Clasificación - {model_name}\n\n")
      f.write(f"- **Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(f"- **Modelo:** {model_name}\n")
      f.write(f"- **Umbral de decisión:** {threshold}\n\n")
      f.write("## Métricas por Clase\n\n")
      f.write("```text\n")
      f.write(text_report)
      f.write("\n```\n\n")
      f.write("---\n")
      f.write("*Generado automáticamente por el sistema de detección de plagas.*")

    print(f"✅ Reporte Markdown (Tabla) guardado en: {report_path_md}")

def plot_roc_curve_and_auc(y_true: np.ndarray, y_scores: np.ndarray, results_dir: str, model_name: str, threshold: float = 0.5) -> None:
  """
  Plotea la curva ROC y calcula el AUC, luego guarda la figura.

  BUG CORREGIDO: plt.show() (bloqueante) seguido de plt.close('all') - ver el mismo
  fix en plot_confusion más arriba. Ahora show(block=False)+pause() para que quede
  visible sin bloquear ni cerrar las demás figuras ya abiertas.
  """

  fpr, tpr, _ = roc_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  report_filename = f"ROC_{model_name}_{timestamp}_t{threshold}.png"
  report_path = os.path.join(results_dir, report_filename)

  plt.figure()
  plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
  plt.plot([0,1], [0,1], linestyle='--')
  plt.xlabel('Tasa de Falsos Positivos')
  plt.ylabel('Tasa de Verdaderos Positivos')
  plt.title(f'Curva ROC - {model_name} (t={threshold})')
  plt.legend()
  plt.savefig(report_path)
  plt.show(block=False)
  plt.pause(0.001)
  print(f"✅ Curva ROC guardada en: {report_path}")

  save_roc_data(y_true, y_scores, os.path.join(results_dir, f"ROC_data_{model_name}_{timestamp}_t{threshold}.npz"))


def save_roc_data(y_true, y_score, output_path):
    """
    Guarda FPR, TPR y AUC para curvas ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    np.savez(
        output_path,
        fpr=fpr,
        tpr=tpr,
        auc=roc_auc
    )

    print(f"ROC guardada en {output_path} (AUC={roc_auc:.4f})")
