# src/utils/utils_train.py

import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from typing import List, Tuple, Union

def save_history_and_plot(
    history: tf.keras.callbacks.History, 
    base_dir: str, 
    epochs: int,
    suffix: str = "",
    isRgb: bool = False,
    loss_type: str = "focal_loss"
) -> None:
    """
    Guarda el historial en JSON y plotea las curvas de entrenamiento.

    BUG CORREGIDO: cada gráfico llamaba a plt.show() (bloqueante) inmediatamente
    seguido de plt.close('all') al final de la función. Con el backend interactivo
    por defecto (no el 'Agg' forzado en tests, ver tests/conftest.py), plt.show() sin
    block=False congela la ejecución hasta cerrar la ventana a mano - 3 veces por
    corrida de train.py (accuracy/loss/recall), lo que colgaba una corrida
    desatendida de los 6 modelos de la bitácora. Ahora se usa show(block=False) +
    pause() para que la ventana se dibuje y quede visible SIN bloquear, y ya no se
    cierran las figuras al final: quedan en pantalla mientras el script sigue.
    """
    imgType = 'RGB' if isRgb else 'MULTIESPECTRAL'
    HISTORY_DIR = os.path.join(base_dir, f'history/{imgType}/{loss_type}')
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # 1. Serialización del Historial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    history_file_name = f"history_{timestamp}_epochs_{epochs}_{imgType}{suffix}.json"
    final_save_path = os.path.join(HISTORY_DIR, history_file_name)
    
    with open(final_save_path, "w") as f:
        json.dump(history.history, f, indent=2)
        
    print(f"\nHistorial guardado en '{final_save_path}'")
    
    # 2. Ploteo de Resultados
    epochs_trained = range(len(history.history['accuracy']))
    
    # Curva de Precisión
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history.history['accuracy'], label='train_acc')
    plt.plot(epochs_trained, history.history['val_accuracy'], label='val_acc')
    plt.title(f'Precisión durante el entrenamiento {imgType}{suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.xticks(epochs_trained, [e + 1 for e in epochs_trained])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    acc_path = os.path.join(HISTORY_DIR, f"accuracy_plot_{timestamp}_epochs_{epochs}_{imgType}{suffix}.png")
    plt.savefig(acc_path)
    plt.show(block=False)
    plt.pause(0.001)

    # Curva de Pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history.history['loss'], label='train_loss')
    plt.plot(epochs_trained, history.history['val_loss'], label='val_loss')
    plt.title(f'Pérdida durante el entrenamiento {imgType}{suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.xticks(epochs_trained, [e + 1 for e in epochs_trained])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    loss_path = os.path.join(HISTORY_DIR, f"loss_plot_{timestamp}_epochs_{epochs}_{imgType}{suffix}.png")
    plt.savefig(loss_path)
    plt.show(block=False)
    plt.pause(0.001)

    # Curva de Recall de Plaga (recall_plaga, no 'recall' - ver plaga_metrics.py:
    # 'recall' mide Sana por el bug ya corregido en cnn_model.py/callbacks.py)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history.history['recall_plaga'], label='train_rec_plaga')
    plt.plot(epochs_trained, history.history['val_recall_plaga'], label='val_rec_plaga')
    plt.title(f'Recall de Plaga durante el entrenamiento {imgType}{suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Recall (Plaga)')
    plt.xticks(epochs_trained, [e + 1 for e in epochs_trained])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    acc_path = os.path.join(HISTORY_DIR, f"recall_plot_{timestamp}_epochs_{epochs}_{imgType}{suffix}.png")
    plt.savefig(acc_path)
    plt.show(block=False)
    plt.pause(0.001)

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, auc

def encontrar_umbral_optimo(model, x_val, y_val):
    # 1. Obtenemos las probabilidades (no las etiquetas fijas)
    y_preds_proba = model.predict(x_val).ravel()
    
    # 2. Calculamos la curva ROC
    fpr, tpr, thresholds = roc_curve(y_val, y_preds_proba)
    
    # 3. Calculamos el índice de Youden: J = Sensibilidad + Especificidad - 1
    # El valor de threshold que maximice J es nuestro "punto dulce"
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n--- ANÁLISIS DE UMBRAL ÓPTIMO ---")
    print(f"Mejor Umbral detectado: {best_threshold:.4f}")
    
    # 4. Mostramos cómo quedaría la matriz con ese nuevo umbral
    y_preds_final = (y_preds_proba >= best_threshold).astype(int)
    print("\nNueva Matriz de Confusión con Umbral Óptimo:")
    print(confusion_matrix(y_val, y_preds_final))
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, y_preds_final))
    print("\nAuc:", auc(fpr, tpr))
    
    return best_threshold