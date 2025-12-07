import argparse
import os
from math import ceil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# Importaciones de Módulos
from src.utils.data_management.base_loader import calculate_class_weights
from src.utils.models.convolutional_neural_factory import crear_modelo_cnn
from src.utils.data_management.extract_data_to_img import crear_datasets_cnn_multiespectral
from src.utils.utils_train import create_cnn_callbacks, save_history_and_plot
from src.utils.print_utils import print_time_and_step

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def run_training(data_dir: str, epochs: int, loss_type: str, isRgb: bool, alpha: float, gamma: float, batch_size: int = 32, base_dir: str = BASE_DIR) -> None:
    IMG_SIZE = (224, 224)
    SEED = 42
    VAL_SPLIT = 0.2

    print_time_and_step('init', f'Iniciando entrenamiento {"RGB" if isRgb else "MULTIESPECTRAL"} con perdida {"Focal Loss" if loss_type == "focal_loss" else "Binary Crossentropy"}', timestamp=timestamp, start_time=start_time)
    # 1. Carga de Datos y Cálculo de Pesos
    print_time_and_step('1', f"1. Cargando datos con {'Undersampling' if loss_type == 'focal_loss' else 'Class Weighting'}...", timestamp=timestamp, start_time=start_time)
    train_ds, val_ds, _, train_counts = crear_datasets_cnn_multiespectral(data_dir=data_dir, isRgb=isRgb, img_size=IMG_SIZE, val_split=VAL_SPLIT, seed=SEED, batch_size=batch_size)
    class_weight = calculate_class_weights(train_counts[0], train_counts[1])
    
    # 2. Construcción del Modelo (3 canales, BCE)
    print_time_and_step('2', f"Creando y Compilando Modelo (MobileNetV2 + {'Focal' if loss_type == 'focal_loss' else 'BCE'})...", timestamp=timestamp, start_time=start_time)
    if loss_type == 'focal_loss':
      model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), loss_type='focal_loss', learning_rate=0.0001, alpha=alpha, gamma=gamma)
    else:
      model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), loss_type='binary_crossentropy', learning_rate=0.0001)
    
    # 3. Callbacks y Pasos
    print_time_and_step('3', "Configurando Callbacks y Pasos...", timestamp=timestamp, start_time=start_time)
    callbacks, _ = create_cnn_callbacks(base_dir, isRgb, loss_type)
    train_size = sum(train_counts.values()) 
    steps_per_epoch = ceil(train_size / batch_size)
    
    val_total_samples = len(val_ds[1]) 
    validation_steps = ceil(val_total_samples / batch_size) if val_total_samples > 0 else 1

    # 4. Entrenamiento
    print_time_and_step("4", "Iniciando entrenamiento...", timestamp=timestamp, start_time=start_time)
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
        class_weight= None if loss_type == 'focal_loss' else class_weight,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    print_time_and_step('5', 'Guardado y Ploteo (Usando utils_train)', timestamp=timestamp, start_time=start_time)
    suffix = f"_Focal_a{alpha}_g{gamma}" if loss_type == 'focal_loss' else "_BCE"
    save_history_and_plot(history, base_dir, epochs, suffix=suffix, isRgb=isRgb)
    
def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
    parser.add_argument("-g", "--gamma", type=float, default=3.0, help="Gamma")
    parser.add_argument("-lt", "--loss_type", type=str, required=True, choices=["focal_loss", "binary_crossentropy"], help="Tipo de funcion de perdida")
    parser.add_argument("-rgb", "--rgb", action='store_true', default=False, help="Es RGB?")
    args = parser.parse_args()
    run_training(args.data_dir, args.epochs, args.loss_type, args.rgb, args.alpha, args.gamma)

if __name__ == "__main__":
    main()