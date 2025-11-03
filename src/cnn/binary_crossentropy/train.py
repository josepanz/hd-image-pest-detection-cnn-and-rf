# src/train.py

import argparse
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras as ks

import os
from datetime import datetime

from data_loader import crear_datasets
from model import crear_modelo

def plot_history(history: tf.keras.callbacks.History) -> None:
    """
    Dibuja y muestra las curvas de precisión y pérdida de entrenamiento y validación.
    """
    epochs = range(len(history.history['accuracy']))

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
    plt.show()

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
    plt.show()


def entrenar(data_dir: str, epochs: int = 10) -> None:
    """
    Realiza el pipeline completo de entrenamiento:
    1. Carga datos (con aumentación).
    2. Construye el modelo (con binary_crossentropy).
    3. Calcula class_weights para el desbalance.
    4. Entrena con callbacks y class_weights.
    5. Guarda modelo e historial.
    6. Plotea métricas.
    """
    # Define parámetros
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    VAL_SPLIT = 0.2
    SEED = 123

    # Definir el directorio donde se ejecuta este script:
    # Esto da la ruta: C:\workspace\tesis_plagas\src\cnn\binary_crossentropy\
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. Carga datasets
    # Ahora capturamos n_plagas_train y n_sanas_train
    train_ds, val_ds, clases, n_plagas_train, n_sanas_train = crear_datasets( # <<<<<<<<<< CAMBIO
        data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    # 2. Construye el modelo (ahora usa binary_crossentropy)
    model = crear_modelo(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # --- INICIO: Cálculo de Class Weights ---
    # Esto implementa la "ponderación de clases" 
    total_train_samples = n_plagas_train + n_sanas_train
    
    # Fórmula de Keras para calcular pesos:
    # peso_para_clase_i = (total_muestras / (num_clases * muestras_de_clase_i))
    
    # Clase 0 (Plaga)
    weight_for_0 = (total_train_samples / (2.0 * n_plagas_train))
    # Clase 1 (Sana)
    weight_for_1 = (total_train_samples / (2.0 * n_sanas_train))

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f"Total de muestras de entrenamiento: {total_train_samples}")
    print(f"Peso para Plaga (Clase 0): {weight_for_0:.2f}")
    print(f"Peso para Sana (Clase 1): {weight_for_1:.2f}")
    # --- FIN: Cálculo de Class Weights ---

    # 4. Define callbacks 
    best_model_path = os.path.join(BASE_DIR, 'best_model.keras')
    callbacks = [
      # Monitoriza la precisión de validación o el recall, no solo la pérdida.
      # Usaremos 'val_recall' (Recall de la clase 'Sana' o positiva) o 'val_accuracy'.
      EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, verbose=1, mode='max'), 
      # Guarda el modelo que tiene la mejor Precisión/Recall en Validación
      ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_recall', mode='max', verbose=1),
      # Reduce la tasa de aprendizaje cuando la métrica de interés se estanca
      ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1)
    ]

    # 5. Entrena el modelo
    history = model.fit(
        train_ds,
        # Ya no necesitamos steps_per_epoch ni validation_steps
        # Keras los infiere automáticamente porque los datasets no tienen .repeat()
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight # <<< CAMBIO: Pasamos los pesos aquí
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
    history_file_name = f"history_{timestamp}.json"
    final_save_path = os.path.join(HISTORY_DIR, history_file_name)
    with open(final_save_path, "w") as f:
        json.dump(history.history, f, indent=2)

    # 7. Plotea resultados
    plot_history(history)

def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.15, help="Alpha")
    args = parser.parse_args()
    entrenar(args.data_dir, args.epochs)


if __name__ == "__main__":
    main()