# src/cnn/train_bce.py
import argparse
import os
from math import ceil
import tensorflow as tf

# Importaciones de Módulos
from src.data_management.convolutional_neural_network.rgb.loader_binary_crossentropy_rgb import crear_datasets_cnn_rgb
from src.data_management.base_loader import calculate_class_weights
from src.models.convolutional_neural_factory import crear_modelo_cnn
from src.utils.utils_train import create_cnn_callbacks, save_history_and_plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_training(data_dir: str, epochs: int, batch_size: int = 32, base_dir: str = BASE_DIR) -> None:
    IMG_SIZE = (224, 224)
    SEED = 123
    
    # 1. Carga de Datos y Cálculo de Pesos
    print("1. Cargando datos con Class Weighting...")
    train_ds, val_ds, _, train_counts = crear_datasets_cnn_rgb(
        data_dir, batch_size=batch_size, img_size=IMG_SIZE, seed=SEED, mode='class_weight'
    )
    class_weight = calculate_class_weights(train_counts[0], train_counts[1])
    
    # 2. Construcción del Modelo (3 canales, BCE)
    print("\n2. Creando y Compilando Modelo (MobileNetV2 + BCE)...")
    model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), loss_type='bce', learning_rate=0.0001)
    
    # 3. Callbacks y Pasos
    callbacks, _ = create_cnn_callbacks(base_dir)
    train_size = sum(train_counts.values()) 
    steps_per_epoch = ceil(train_size / batch_size)
    val_size = tf.data.experimental.cardinality(val_ds).numpy() * batch_size 
    validation_steps = ceil(val_size / batch_size) if val_size > 0 else 1 

    # 4. Entrenamiento
    print("\n4. Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    save_history_and_plot(history, base_dir, epochs, model, suffix="_BCE")
    
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