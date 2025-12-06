# src/cnn/train_focal.py
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
from src.models.convolutional_neural_factory import crear_modelo_cnn
from src.utils.utils_train import create_cnn_callbacks, save_history_and_plot
from src.utils.print_utils import print_time_and_step
from src.utils.extract_data_to_img import crear_datasets_cnn_multiespectral

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def run_training(data_dir: str, epochs: int, base_dir: str = BASE_DIR, alpha: float = 0.50, gamma: float = 3.0, batch_size: int = 32) -> None:
    IMG_SIZE = (224, 224)
    
    # 1. Carga de Datos (Undersampling)
    print_time_and_step('1', "Cargando datos con Undersampling...", timestamp=timestamp, start_time=start_time)
    train_ds, val_ds, _, train_counts = crear_datasets_cnn_multiespectral(isRgb=True)
    
    # 2. Construcción del Modelo (3 canales, Focal Loss)
    print_time_and_step('2', f"\n2. Creando y Compilando Modelo (MobileNetV2 + Focal Loss α={alpha}, γ={gamma})...", timestamp=timestamp, start_time=start_time)
    model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), loss_type='focal_loss', learning_rate=0.0001, alpha=alpha, gamma=gamma)
    
    # 3. Callbacks y Pasos
    print_time_and_step('3', "Configurando Callbacks y Pasos...", timestamp=timestamp, start_time=start_time)
    callbacks, _ = create_cnn_callbacks(base_dir)
    train_size = sum(train_counts.values()) 
    steps_per_epoch = ceil((2 * train_size) / batch_size)
    
    val_total_samples = len(val_ds[1]) 
    validation_steps = ceil(val_total_samples / batch_size) if val_total_samples > 0 else 1

    # 4. Entrenamiento
    print_time_and_step("4", "Iniciando entrenamiento...", timestamp=timestamp, start_time=start_time)
    history = model.fit(
        train_ds[0],  # X_train (Features/Imágenes)
        train_ds[1],  # y_train (Labels/Etiquetas)
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    print_time_and_step('5', 'Guardado y Ploteo (Usando utils_train)', timestamp=timestamp, start_time=start_time)
    save_history_and_plot(history, base_dir, epochs, suffix=f"_Focal_a{alpha}_g{gamma}")

def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
    args = parser.parse_args()
    run_training(args.data_dir, args.epochs, alpha = args.alpha)

if __name__ == "__main__":
    main()