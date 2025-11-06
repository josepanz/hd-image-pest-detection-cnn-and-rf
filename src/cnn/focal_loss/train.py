# src/cnn/focal_loss/train.py

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras as ks

import os
from datetime import datetime

from data_loader import crear_datasets
from model import crear_modelo


def plot_history(history: tf.keras.callbacks.History, history_dir: str, train_epochs: int, alpha: float) -> None:
    """
    Dibuja y muestra las curvas de precisión y pérdida de entrenamiento y validación.
    """
    epochs = range(len(history.history['accuracy']))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Curva de Precisión
    plt.figure()
    plt.plot(epochs, history.history['accuracy'], label='train_acc')
    plt.plot(epochs, history.history['val_accuracy'], label='val_acc')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # GUARDAR la figura en el directorio especificado
    acc_path = os.path.join(history_dir, f"accuracy_plot_{timestamp}_epochs_{train_epochs}_alpha_{alpha}.png")
    plt.savefig(acc_path)

    plt.show()
    plt.close() # Cierra la figura para liberar memoria
    print(f"Gráfico de Precisión guardado en: {acc_path}")

    # Curva de Pérdida
    plt.figure()
    plt.plot(epochs, history.history['loss'], label='train_loss')
    plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # GUARDAR la figura en el directorio especificado
    loss_path = os.path.join(history_dir, f"loss_plot_{timestamp}_epochs_{train_epochs}_alpha_{alpha}.png")
    plt.savefig(loss_path)
    plt.show()
    plt.close() # Cierra la figura
    print(f"Gráfico de Pérdida guardado en: {loss_path}")

# el valor basico seria epochs 10, alpha 0.15
def entrenar(data_dir: str, epochs: int = 20, alpha: float = 0.50) -> None:
    """
    Realiza el pipeline completo de entrenamiento:
    1. Carga datos.
    2. Construye el modelo.
    3. Calcula steps_per_epoch para un tf.data.Dataset finito.
    4. Entrena con callbacks.
    5. Guarda modelo e historial.
    6. Plotea métricas.
    """
    # Define el batch_size aquí para usarlo consistentemente
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    VAL_SPLIT = 0.2
    SEED = 123

    # Definir el directorio donde se ejecuta este script:
    # Esto da la ruta: C:\workspace\tesis_plagas\src\cnn\binary_crossentropy\
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. Carga datasets
    # Ahora esperamos n_sanas_in_train como cuarto valor de retorno de crear_datasets
    train_ds, val_ds, clases, n_sanas_in_train = crear_datasets( 
        data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    # 2. Construye el modelo
    # Se usará el alpha y gamma predefinidos en src/model.py (0.15 y 3.0 por ahora)
    model = crear_modelo(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), alpha=alpha)

    # 3. Calcula steps_per_epoch para train_ds y val_steps para val_ds
    # El dataset de entrenamiento balanceado tiene 2 * n_sanas_in_train muestras.
    total_balanced_samples_in_train = 2 * n_sanas_in_train # Total de muestras en el TRAIN_DS balanceado
    steps_per_epoch = int(np.ceil(total_balanced_samples_in_train / BATCH_SIZE)) # <<<<<<<<<< CAMBIO AQUÍ: Usamos n_sanas_in_train

    total_samples = 2152
    # total_val_samples = int(np.floor(total_samples * VAL_SPLIT)) # 430
    total_val_samples = int(tf.data.experimental.cardinality(val_ds.unbatch()).numpy())
    if total_val_samples == tf.data.INFINITE_CARDINALITY:
      # Si es infinito (por el .repeat()), asumimos el tamaño real de validación (ej. 430/32 = 14)
      # Este valor debe ser estimado por ti basado en el total de tu dataset real
      # Asumiendo 2152 total -> 430 validación -> 14 batches
        total_val_samples = int(np.floor(total_samples * VAL_SPLIT))
    else:
        total_val_samples = int(tf.data.experimental.cardinality(val_ds.unbatch()).numpy())
        
    val_steps = int(np.ceil(total_val_samples / BATCH_SIZE))
    
    print(f"DEBUG: Estimando val_steps a {val_steps} basado en VAL_SPLIT.")
    print(f"DEBUG: Número de muestras sana en TRAIN_DS (base undersampling): {n_sanas_in_train}")
    print(f"DEBUG: Tamaño del TRAIN_DS balanceado: {total_balanced_samples_in_train}")
    print(f"DEBUG: Tamaño estimado del VAL_DS: {total_val_samples}")
    print(f"Steps per epoch (calculado): {steps_per_epoch}, Validation steps: {val_steps}")

    # 4. Define callbacks
    best_model_path = os.path.join(BASE_DIR, 'best_model.keras')
    callbacks = [
      # Monitoriza la precisión de validación o el recall, no solo la pérdida.
      # Usaremos 'val_recall' (Recall de la clase 'Sana' o positiva) o 'val_accuracy'.
      # src/cnn/focal_loss/train.py
      EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, verbose=1, mode='max'), 
      # Guarda el modelo que tiene la mejor Precisión/Recall en Validación
      ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_recall', mode='max', verbose=1),
      # Reduce la tasa de aprendizaje cuando la métrica de interés se estanca
      ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1)
    ]

    # 5. Entrena el modelo (train_ds ahora emite solo (x, y) porque sample_weights están desactivados)
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks
        # No pasamos class_weight ni sample_weight porque la pérdida focal ya los maneja
    )

    # 6. Guarda modelo e historial
    modelo_hd_path = os.path.join(BASE_DIR, 'modelo_hd.keras')
    ks.saving.save_model(model, modelo_hd_path)

    # Definir el subdirectorio de resultados
    HISTORY_DIR = os.path.join(BASE_DIR, 'history')
    
    # Crear el directorio 'results' si no existe
    # El argumento exist_ok=True evita un error si la carpeta ya existe.
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Construir la ruta final: Directorio de Resultados + Nombre del reporte
    # Formato de fecha y hora: AAAA-MM-DD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    history_file_name = f"history_{timestamp}_epochs_{epochs}_alpha_{alpha}.json"
    final_save_path = os.path.join(HISTORY_DIR, history_file_name)
    with open(final_save_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"\Historial guardado en '{final_save_path}'")
    # 7. Plotea resultados
    plot_history(history, HISTORY_DIR, epochs, alpha)


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
    args = parser.parse_args()
    entrenar(args.data_dir, args.epochs, args.alpha)


if __name__ == "__main__":
    main()