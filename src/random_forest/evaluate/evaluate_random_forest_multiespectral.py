# src/rf/evaluaterf_multiespectral.py

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib 
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# --- IMPORTACIÓN CLAVE ---
# Usamos el nuevo loader para extraer las 7 características espectrales
from src.data_management.random_forest.loader_random_forest_multiespectral import crear_datasets_rf_ms 

CLASSES = ["Plaga", "Sana"] 
RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_multispectral_rf')
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    """
    Dibuja la matriz de confusión y la guarda. (Función de utilería estándar).
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
    plt.savefig(save_path)
    print(f"Gráfico de Matriz de Confusión guardado en: {save_path}")
    plt.close()


def generar_nombre_reporte(model_path: str) -> str:
    """
    Genera un nombre de archivo único basado en el modelo.
    """
    model_name = os.path.basename(model_path).replace('.joblib', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"evaluation_report_{model_name}_{timestamp}.json"


def evaluar_rf_ms(model_path: str, patch_size: int = 224) -> None:
    """
    Evalúa el modelo Random Forest Multiespectral.
    """
    
    # 1. Carga del set de Validación (solo necesitamos X_val y y_val)
    print("1. Cargando y Extrayendo características de validación...")
    # Solo necesitamos los conjuntos de validación (X_val, y_val) y las clases
    _, X_val, _, y_val, class_names = crear_datasets_rf_ms(
        patch_size=patch_size, 
        test_split=0.2 # Debe coincidir con el split de entrenamiento
    )

    # 2. Carga del Modelo
    print(f"\n2. Cargando modelo desde: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error cargando el modelo RF: {e}")
        return

    # 3. Predicciones
    y_pred = model.predict(X_val) 
    y_true = y_val

    # 4. Reporte de clasificación y Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Generar y guardar el nombre del reporte JSON
    report_file_name = generar_nombre_reporte(model_path)
    final_report_path = os.path.join(RESULTS_BASE_DIR, report_file_name)
    
    # Guardar el gráfico de Matriz de Confusión
    plot_file_name = report_file_name.replace('.json', '_confusion.png')
    final_plot_path = os.path.join(RESULTS_BASE_DIR, plot_file_name)
    plot_confusion(cm, class_names, final_plot_path )

    # Generar el reporte de clasificación (precision, recall, f1, etc.)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0 
    )
    
    # 5. Guardar Reporte
    with open(final_report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    print("\n--- Reporte de Clasificación Final ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"\n✅ Reporte guardado en: {final_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo Random Forest Multiespectral.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo guardado (.joblib).")
    parser.add_argument("-p", "--patch_size", type=int, default=224, help="Tamaño del parche usado para la extracción de features.")
    args = parser.parse_args()
    
    evaluar_rf_ms(args.model, args.patch_size)

if __name__ == "__main__":
    main()