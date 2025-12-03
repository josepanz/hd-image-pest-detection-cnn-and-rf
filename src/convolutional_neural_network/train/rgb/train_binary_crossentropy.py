# src/cnn/train_bce.py
import argparse
import os
from math import ceil
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils'))

# Importaciones de Módulos
from src.data_management.convolutional_neural_network.rgb.loader_binary_crossentropy_rgb import crear_datasets_cnn_rgb
from src.data_management.base_loader import calculate_class_weights
from src.models.convolutional_neural_factory import crear_modelo_cnn
from src.utils.extract_data_to_img import crear_datasets_cnn_multiespectral
from src.utils.utils_train import create_cnn_callbacks, save_history_and_plot
from src.utils.print_utils import print_time_and_step

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def run_training(data_dir: str, epochs: int, batch_size: int = 32, base_dir: str = BASE_DIR) -> None:
    IMG_SIZE = (224, 224)
    SEED = 123
    
    # 1. Carga de Datos y Cálculo de Pesos
    print_time_and_step("1. Cargando datos con Class Weighting...", timestamp=timestamp, start_time=start_time)
    # train_ds, val_ds, _, train_counts = crear_datasets_cnn_multiespectral(
    #     data_dir, batch_size=batch_size, img_size=IMG_SIZE, seed=SEED, mode='class_weight'
    # )
    train_ds, val_ds, _, train_counts = crear_datasets_cnn_multiespectral(
            data_dir='Ignored', # Placeholder para la firma
            batch_size=32, 
            img_size=(224, 224),
            seed=42, 
            mode='class_weight',
            isRgb=True
        )
    class_weight = calculate_class_weights(train_counts[0], train_counts[1])
    
    # 2. Construcción del Modelo (3 canales, BCE)
    print_time_and_step('2', "Creando y Compilando Modelo (MobileNetV2 + BCE)...", timestamp=timestamp, start_time=start_time)
    model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), loss_type='binary_crossentropy', learning_rate=0.0001)
    
    # 3. Callbacks y Pasos
    print_time_and_step('3', "Configurando Callbacks y Pasos...", timestamp=timestamp, start_time=start_time)
    callbacks, _ = create_cnn_callbacks(base_dir)
    train_size = sum(train_counts.values()) 
    steps_per_epoch = ceil(train_size / batch_size)
    
    # TODO: verificar si esto funciona correctamente
    #val_size = tf.data.experimental.cardinality(val_ds).numpy() * batch_size 
    #validation_steps = ceil(val_size / batch_size) if val_size > 0 else 1 

    val_total_samples = len(val_ds[1]) 
    validation_steps = ceil(val_total_samples / batch_size) if val_total_samples > 0 else 1

    # 4. Entrenamiento
    print_time_and_step("4", "Iniciando entrenamiento...", timestamp=timestamp, start_time=start_time)
    # history = model.fit(
    #     train_ds,
    #     epochs=epochs,
    #     steps_per_epoch=steps_per_epoch,
    #     validation_data=val_ds,
    #     validation_steps=validation_steps,
    #     callbacks=callbacks,
    #     class_weight=class_weight,
    #     verbose=1
    # )

    # Desempaquetamos train_ds en X_train (train_ds[0]) y y_train (train_ds[1])
    # Nota: val_ds es una tupla (X_test, y_test) y funciona correctamente en validation_data.
    history = model.fit(
        train_ds[0],  # X_train (Features/Imágenes)
        train_ds[1],  # y_train (Labels/Etiquetas)
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    print_time_and_step('5', 'Guardado y Ploteo (Usando utils_train)', timestamp=timestamp, start_time=start_time)
    save_history_and_plot(history, base_dir, epochs, suffix="_BCE")
    
def main():
    # ... (Lógica de argparse) ...
    # Nota: Aquí iría tu lógica de argparse y la llamada a run_training
    # Asegúrate de que las rutas de importación sean correctas.
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.15, help="Alpha")
    args = parser.parse_args()
    run_training(args.data_dir, args.epochs)


if __name__ == "__main__":
    main()