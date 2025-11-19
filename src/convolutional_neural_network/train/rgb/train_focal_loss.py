# src/cnn/train_focal.py
import argparse
import os
from math import ceil
import tensorflow as tf

# Importaciones de Módulos
from src.data_management.convolutional_neural_network.rgb.loader_binary_crossentropy_rgb import crear_datasets_cnn_rgb
from src.models.convolutional_neural_factory import crear_modelo_cnn
from src.utils.utils_train import create_cnn_callbacks, save_history_and_plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_training(data_dir: str, epochs: int, base_dir: str = BASE_DIR, alpha: float = 0.50, gamma: float = 3.0, batch_size: int = 32) -> None:
    IMG_SIZE = (224, 224)
    
    # 1. Carga de Datos (Undersampling)
    print(f"1. Cargando datos con Undersampling...")
    train_ds, val_ds, _, n_minority_samples = crear_datasets_cnn_rgb(
        data_dir, batch_size=batch_size, img_size=IMG_SIZE, mode='undersample'
    )
    
    # 2. Construcción del Modelo (3 canales, Focal Loss)
    print(f"\n2. Creando y Compilando Modelo (MobileNetV2 + Focal Loss α={alpha}, γ={gamma})...")
    model = crear_modelo_cnn(
        input_shape=(*IMG_SIZE, 3), 
        loss_type='focal',
        learning_rate=0.0001,
        alpha=alpha,
        gamma=gamma
    )
    
    # 3. Callbacks y Pasos
    callbacks, _ = create_cnn_callbacks(base_dir)
    steps_per_epoch = ceil((2 * n_minority_samples) / batch_size) 
    
    # Nota: validation_steps es un tema complejo con .repeat() en val_ds. 
    # Usaremos un valor fijo si el loader usa .repeat().
    validation_steps = 10 

    # 4. Entrenamiento
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    suffix = f"_Focal_a{alpha}_g{gamma}"
    save_history_and_plot(history, base_dir, epochs, model, suffix=suffix)

def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
    parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
    args = parser.parse_args()
    run_training(args.data_dir, args.epochs, alpha = args.alpha)


if __name__ == "__main__":
    main()