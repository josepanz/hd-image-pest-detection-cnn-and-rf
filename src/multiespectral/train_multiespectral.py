import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from datetime import datetime
import json

# Importa las funciones clave
# Nota: Asegúrate de que los archivos 'data_loader_multiespectral.py' 
# y 'model_multiespectral.py' estén en una ruta accesible.
try:
    from data_loader_multiespectral import load_multiespectral_data
    from model_multiespectral import crear_modelo_multiespectral
    # Si vas a usar Focal Loss, asegúrate de que esté importado o definido aquí.
    # from your_focal_loss_module import BinaryFocalLoss 
except ImportError as e:
    print(f"Error de importación: {e}")
    print("Asegúrate de que 'data_loader_multiespectral.py' y 'model_multiespectral.py' están disponibles.")
    sys.exit(1)


# --- CONFIGURACIÓN DE RUTAS ---
# Usamos una carpeta separada para los resultados multiespectrales
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'results_multispectral')
MODEL_SAVE_DIR = os.path.join(RESULTS_BASE_DIR, 'best_models_ms')
HISTORY_SAVE_DIR = os.path.join(RESULTS_BASE_DIR, 'history_results_multispectral')
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(HISTORY_SAVE_DIR, exist_ok=True)

CLASSES = ["Plaga", "Sana"] 


# --- FUNCIÓN DE PLOTEO (Adaptada) ---
def plot_history(history: tf.keras.callbacks.History, history_dir: str, title_suffix: str) -> None:
    """Plotea y guarda las curvas de precisión y pérdida."""
    
    epochs = range(len(history.history['loss']))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Curva de Precisión (Accuracy)
    plt.figure(figsize=(10, 6)) 
    plt.plot(epochs, history.history['accuracy'], label='train_acc')
    plt.plot(epochs, history.history['val_accuracy'], label='val_acc')
    plt.title(f'Precisión durante el entrenamiento ({title_suffix})')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    acc_path = os.path.join(history_dir, f"accuracy_plot_{timestamp}_{title_suffix}.png")
    plt.savefig(acc_path)
    plt.show()
    plt.close() 
    print(f"Gráfico de Precisión guardado en: {acc_path}")

    # 2. Curva de Pérdida (Loss)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['loss'], label='train_loss')
    plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.title(f'Pérdida durante el entrenamiento ({title_suffix})')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    loss_path = os.path.join(history_dir, f"loss_plot_{timestamp}_{title_suffix}.png")
    plt.savefig(loss_path)
    plt.show()
    plt.close() 
    print(f"Gráfico de Pérdida guardado en: {loss_path}")


# --- FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
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

    # 1. Carga de Datos Multiespectrales
    IMG_SIZE_TUPLE = tuple(args.size)
    print("Iniciando la carga de datos multiespectrales (puede tardar)...")
    try:
        # X: (N_samples, 224, 224, 5), Y: (N_samples,)
        # X, Y = load_multiespectral_data(patch_size=IMG_SIZE_TUPLE[0])  # mucho consumo de ram, superior a 10.9 GB
        X, Y = load_multiespectral_data(patch_size=128) # reducir el tamaño del parche
    except Exception as e:
        print(f"\n❌ Error Crítico al cargar los datos multiespectrales: {e}")
        sys.exit(1)
        
    # La CNN necesita etiquetas One-Hot Encoded
    Y_one_hot = to_categorical(Y, num_classes=len(CLASSES))

    # 2. División de Datos (Train / Validation)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_one_hot, test_size=args.test_split, random_state=42, stratify=Y
    )

    print(f"\nDatos de Entrenamiento: {X_train.shape[0]} parches.")
    print(f"Datos de Validación: {X_val.shape[0]} parches.")
    
    # 3. Creación del Modelo Multiespectral
    # La forma de entrada es (Alto, Ancho, 5)
    INPUT_SHAPE = X_train.shape[1:] 
    
    modelo = crear_modelo_multiespectral(
        #input_shape=INPUT_SHAPE, # mucho consumo de ram, superior a 10.9 GB
        input_shape=(128, 128, 5),
        num_classes=1, # Para Binary Crossentropy/Focal Loss, la salida es 1 neurona con Sigmoid.
        fine_tune_layers=args.fine_tune,
        dropout_rate=args.dropout,
        loss_function=args.loss_function,
        alpha=args.alpha,
        gamma=args.gamma
    )
    
    # 4. Callbacks para entrenamiento
    # Para guardar el mejor modelo basado en la precisión de validación (val_accuracy)
    model_name = "best_model_ms_" + ("ft" if args.fine_tune else "base") + ".keras"
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # Detiene si la pérdida no mejora después de 5 epochs
            verbose=1
        )
    ]

    # 5. Entrenamiento del Modelo
    print("\nIniciando entrenamiento del Modelo Multiespectral...")
    history = modelo.fit(
        X_train, Y_train[:, 1], # Usamos Y_train[:, 1] si el modelo espera un array de una dimensión (Sigmoid)
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, Y_val[:, 1]),
        callbacks=callbacks,
        verbose=1
    )

    # 6. Ploteo y Guardado de Resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    history_file_name = f"history_{timestamp}_Multiespectral.json"
    final_save_path = os.path.join(HISTORY_SAVE_DIR, history_file_name)
    with open(final_save_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"\Historial guardado en '{final_save_path}'")

    plot_history(history, HISTORY_SAVE_DIR, title_suffix="Multiespectral")
    print(f"\n✅ Entrenamiento Multiespectral Finalizado. Mejor modelo guardado en: {model_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt # Importar Matplotlib aquí para evitar conflictos antes del main
    main()