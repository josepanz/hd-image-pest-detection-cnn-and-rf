# src/cnn/train_ms.py
import argparse
import os
import tensorflow as tf

# Ajustar la ruta de importación si es necesario
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Importaciones de Módulos
# from data_management.loader_cnn_ms import load_multiespectral_data
from src.models.convolutional_neural_factory import crear_modelo_cnn
from utils.utils_train import create_cnn_callbacks, save_history_and_plot
from utils.extract_data_to_img import extract_data_to_img_for_train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_training(patch_size: int, epochs: int, loss_type: str, batch_size: int = 32,  base_dir: str = BASE_DIR) -> None:
    
    # 1. Carga de Datos (Numpy Arrays de 5 canales)
    print("1. Cargando y Extrayendo parches Multiespectrales (5 canales)...")
    # X_train, X_val, Y_train, Y_val = load_multiespectral_data(patch_size=patch_size, test_split=0.2)
    X_train, X_val, Y_train, Y_val, _ = extract_data_to_img_for_train()
    
    # 2. Construcción del Modelo (5 canales)
    print("\n2. Creando y Compilando Modelo Multiespectral (MobileNetV2 5ch)...")
    model = crear_modelo_cnn(
        input_shape=(patch_size, patch_size, 5), 
        loss_type=loss_type, 
        learning_rate=0.0001
    )
    
    # 3. Callbacks
    callbacks, _ = create_cnn_callbacks(base_dir)

    # 4. Entrenamiento (sin steps_per_epoch para Numpy Arrays)
    print("\n4. Iniciando entrenamiento...")
    history = model.fit(
        X_train, 
        Y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Guardado y Ploteo (Usando utils_train)
    suffix = f"_MS_{loss_type.upper()}"
    save_history_and_plot(history, base_dir, epochs, model, suffix=suffix)

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de CNN Multiespectral con Focal/BCE Loss")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Número de epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Tamaño del lote.")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=[224, 224], help="Tamaño de la imagen (alto ancho).")
    parser.add_argument("-t", "--test_split", type=float, default=0.2, help="Proporción del conjunto de prueba/validación.")
    parser.add_argument("-d", "--dropout", type=float, default=0.5, help="Tasa de Dropout.")
    parser.add_argument("-ft", "--fine_tune", action="store_true", help="Permite el fine-tuning de la capa base.")
    parser.add_argument("-f", "--loss_function", choices=['focal_loss', 'binary_crossentropy'], default="binary_crossentropy", help="Función de pérdida")
    parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
    parser.add_argument("-g", "--gamma", type=float, default=3.0, help="Gamma")
    args = parser.parse_args()
    run_training(args.size[0], args.epochs, args.loss_function, args.batch_size)

if __name__ == "__main__":
    main()