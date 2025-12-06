# src/utils/utils_train.py

import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from typing import List, Tuple, Union

def create_cnn_callbacks(base_dir: str, isRgb: bool = False, loss_type: str = "focal_loss", monitor: str = 'val_recall') -> Tuple[List[tf.keras.callbacks.Callback], str]:
    """
    Crea y devuelve la lista estándar de callbacks y la ruta de guardado del mejor modelo.
    """
    # Directorio para guardar el mejor modelo
    model_save_dir = os.path.join(base_dir, 'best_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Ruta donde se guardará el modelo con mejor precisión
    model_path = os.path.join(model_save_dir, f'best_model_{monitor}_{"RGB" if isRgb else "MULTIESPECTRAL"}_{loss_type}.keras')
    
    callbacks = [
        # 1. Detención temprana (si la pérdida de validación no mejora)
        EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        # 2. Guardado del mejor modelo (basado en la precisión)
        ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor=monitor,
            verbose=1,
            mode='max'
        ),
        # 3. Reducción de la tasa de aprendizaje (para evitar estancamientos)
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode='max'
        )
    ]
    return callbacks, model_path

def save_history_and_plot(
    history: tf.keras.callbacks.History, 
    base_dir: str, 
    epochs: int,
    suffix: str = "",
    isRgb: bool = False
) -> None:
    """
    Guarda el historial en JSON y plotea las curvas de entrenamiento.
    """
    imgType = 'RGB' if isRgb else 'MULTIESPECTRAL'
    HISTORY_DIR = os.path.join(base_dir, f'history/{imgType}')
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
    plt.show()
    
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
    plt.show()

    # Curva de Recall
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_trained, history.history['recall'], label='train_rec')
    plt.plot(epochs_trained, history.history['val_recall'], label='val_rec')
    plt.title(f'Recall durante el entrenamiento {imgType}{suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.xticks(epochs_trained, [e + 1 for e in epochs_trained])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    acc_path = os.path.join(HISTORY_DIR, f"recall_plot_{timestamp}_epochs_{epochs}_{imgType}{suffix}.png")
    plt.savefig(acc_path)
    plt.show()

    plt.close('all')