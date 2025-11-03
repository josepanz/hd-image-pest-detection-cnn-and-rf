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

def plot_confusion(cm: np.ndarray, class_names: list[str]) -> None:
    # ... (código de plot_confusion, mantener el original)
    pass 

def generar_nombre_reporte() -> str:
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"classification_report_rf_{timestamp}.json"


def evaluar_modelo(data_dir: str, model_path: str, report_path: str) -> None:
    
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
    plot_confusion(cm, class_names)

    report_dict = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0 
    )
    
    # 5. Guardar Reporte
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
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