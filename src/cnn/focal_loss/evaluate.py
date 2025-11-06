"""
evaluate.py

Script para evaluar el modelo HD-only:
- Carga el modelo guardado.
- Calcula y muestra la matriz de confusión de validación.
- Genera y guarda el reporte de clasificación (precision, recall, f1-score).
- Plotea la matriz de confusión con anotaciones.
- Permite especificar umbral de decisión para clasificador binario.
"""

import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from datetime import datetime
import os

from data_loader import crear_datasets
from model import focal_loss # Importar focal_loss para cargar el modelo si lo necesita

def generar_nombre_reporte(threshold: float = 0.5) -> str:
    """
    Genera un nombre de archivo único incluyendo la fecha y el umbral.
    Ejemplo: classification_report_20251102_t075.json
    """
    # Formato de fecha y hora: AAAA-MM-DD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Formato del umbral: de 0.75 a t075 (removiendo el punto)
    umbral_str = f"t{int(threshold * 100):02d}"

    return f"classification_report_{timestamp}_{umbral_str}.json"


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    """
    Dibuja la matriz de confusión con anotaciones de recuento.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Verdadera")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color=color)
    fig.tight_layout()

    # GUARDAR la figura
    plt.savefig(save_path)

    plt.show()
    plt.close() # Cierra la figura
    print(f"\nMatriz de Confusión guardada en: {save_path}")


def evaluar(
    data_dir: str,
    model_path: str = "best_model.keras",
    report_path: str = "classification_report.json",
    threshold: float = 0.5,
    steps: float = 14
) -> None:
    """
    Evalúa el modelo en el conjunto de validación:
    1. Carga datos de validación.
    2. Carga el modelo guardado sin compilar para evitar errores con funciones personalizadas.
    3. Calcula etiquetas verdaderas y predichas.
    4. Imprime y plotea la matriz de confusión.
    5. Genera y guarda el reporte de clasificación en JSON.

    Args:
        data_dir: Directorio raíz con subcarpetas de clases.
        model_path: Ruta al archivo de modelo guardado.
        report_path: Ruta para guardar el reporte JSON.
        threshold: Umbral de decisión para clasificador binario (0-1).
    """
    # Define los parámetros de carga del dataset de validación
    # Asegúrate de que estos coincidan con los usados en train.py
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    SEED = 123

    # Definir el directorio donde se ejecuta este script:
    # Esto da la ruta: C:\workspace\tesis_plagas\src\cnn\binary_crossentropy\
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Definir el subdirectorio de resultados
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')

    # Crear el directorio 'results' si no existe
    # El argumento exist_ok=True evita un error si la carpeta ya existe.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Carga solo el dataset de validación y nombres de clases
    # Captura el cuarto valor (n_sanas) con un guion bajo para ignorarlo.
    # Pasa los parámetros necesarios a crear_datasets
    train_ds, val_ds, class_names, n_sanas_in_train = crear_datasets( 
        data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    print(f"Clases detectadas para evaluación: {class_names}")

    # 2. Carga el modelo sin compilar, pero con custom_objects si la función de pérdida es personalizada
    # Aunque compile=False, es una buena práctica pasar custom_objects para funciones de pérdida
    # en caso de que el modelo necesite sus definiciones internas para la inferencia.
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'focal_loss': focal_loss()}) # <<<<< CAMBIO AQUI

    # 3. Recopila etiquetas verdaderas
    # Es crucial que las etiquetas (y) estén 'aplanadas' para np.concatenate
    # Asumimos que val_steps = 14 (o el valor que te dio el log de train.py)
    # Recopila solo los batches necesarios con .take()
    y_true = np.concatenate([y.numpy().flatten() for x, y in val_ds.take(steps)], axis=0).astype(int) 

    # 4. Predicciones
    probs = model.predict(val_ds, steps=steps)
    
    # Asegúrate de que las probabilidades sean 1D para el umbral binario
    # Si probs tiene forma (N, 1), lo convertimos a (N,)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs[:, 0]
        
    # Clasificador binario: threshold configurable
    y_pred = (probs > threshold).astype(int)
    
    # Nota: No necesitamos la lógica de 'else: Multiclase' porque tu modelo siempre es binario.

    # 5. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\nMatriz de confusión:")
    print(cm)
    # Generar el nombre para el gráfico basado en el nombre del reporte JSON
    # Se reemplaza la extensión .json por .png
    plot_file_name = report_path.replace('.json', '_confusion.png')
    final_plot_path = os.path.join(RESULTS_DIR, plot_file_name)
    plot_confusion(cm, class_names, final_plot_path )

    # 6. Reporte de clasificación
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0 # Evita warnings si alguna clase no tiene predicciones
    )
    print("\nReporte de clasificación:")
    print(json.dumps(report_dict, indent=2))
    
    # Construir la ruta final: Directorio de Resultados + Nombre del reporte
    final_save_path = os.path.join(RESULTS_DIR, report_path)
    
    with open(final_save_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReporte guardado en '{final_save_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa el modelo HD-only en imágenes de validación"
    )
    parser.add_argument(
        "data_dir",
        help="Ruta al directorio con subcarpetas de clases para validación"
    )
    parser.add_argument(
        "-m", "--model",
        default="best_model.keras",
        help="Archivo del modelo guardado"
    )
    parser.add_argument(
        "-r", "--report",
        help="Ruta para guardar el reporte JSON"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        #default=0.5, 0.50
        default=0.65, # Con el umbral mas optimo para F1-Score por defecto
        help="Umbral de decisión para clasificador binario"
    )
    parser.add_argument(
        "-s", "--steps",
        default=14,
        help="Pasos o batches por cada epoch"
    )
    args = parser.parse_args()
    if args.report is None:
        # Si el usuario no especificó -r, generamos el nombre automáticamente
        if args.threshold is None:
          reporte_final_path = generar_nombre_reporte()
        else:
          reporte_final_path = generar_nombre_reporte(args.threshold)
    else:
        # Si el usuario especificó -r, usamos su ruta
        reporte_final_path = args.report

    print("reporte_final_path: ", reporte_final_path)
    evaluar(
        args.data_dir,
        model_path=args.model,
        report_path=reporte_final_path,
        threshold=args.threshold,
        steps=args.steps
    )

if __name__ == "__main__":
    main()