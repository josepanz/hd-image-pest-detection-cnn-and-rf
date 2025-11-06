# src/rf/evaluate.py
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib 
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Ajustar la ruta de importación si es necesario
# ... (código de ajuste de ruta, omitido para brevedad)

from data_loader import crear_datasets 

CLASSES = ["Plaga", "Sana"] 

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

def generar_nombre_reporte() -> str:
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"classification_report_rf_{timestamp}.json"


def evaluar_modelo(data_dir: str, model_path: str, report_path: str) -> None:
    
    # Definir el directorio donde se ejecuta este script:
    # Esto da la ruta: C:\workspace\tesis_plagas\src\cnn\binary_crossentropy\
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Definir el subdirectorio de resultados
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')

    # Crear el directorio 'results' si no existe
    # El argumento exist_ok=True evita un error si la carpeta ya existe.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Cargar Datos de Validación (Features)
    _, X_val, _, y_val, class_names = crear_datasets(
        data_dir, test_split=0.2, seed=123
    )
    
    # 2. Cargar Modelo Random Forest
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error cargando el modelo RF: {e}")
        return

    # 3. Predicciones
    y_pred = model.predict(X_val) 

    # 4. Reporte de clasificación y Matriz de Confusión
    cm = confusion_matrix(y_val, y_pred)
    # Generar el nombre para el gráfico basado en el nombre del reporte JSON
    # Se reemplaza la extensión .json por .png
    plot_file_name = report_path.replace('.json', '_confusion.png')
    final_plot_path = os.path.join(RESULTS_DIR, plot_file_name)
    plot_confusion(cm, class_names, final_plot_path )

    report_dict = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0 
    )
    
    # 5. Guardar Reporte
    final_save_path = os.path.join(RESULTS_DIR, report_path)
    with open(final_save_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nReporte guardado en '{final_save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo Random Forest")
    parser.add_argument("data_dir", help="Ruta al directorio con subcarpetas de clases")
    parser.add_argument("-m", "--model", help="Archivo del modelo guardado (.joblib)")
    parser.add_argument("-r", "--report", type=str, default=generar_nombre_reporte(),
                        help="Nombre del archivo para guardar el reporte JSON")
    args = parser.parse_args()
    
    evaluar_modelo(args.data_dir, args.model, args.report)


if __name__ == "__main__":
    main()