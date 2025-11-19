# src/rf/train_rf.py
import argparse
import os
import joblib 
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importaciones de Módulos
from src.data_management.random_forest.loader_random_forest_rgb import crear_datasets_rf
from src.models.model_random_forest_rgb import crear_modelo_rf 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_rf(model, X_val, y_val) -> None:
    """Evalúa e imprime las métricas de clasificación para el modelo RF."""
    y_pred = model.predict(X_val)
    
    print("\n--- Resultados de Evaluación Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
    print("---------------------------------------------")

def run_training(data_dir: str, base_dir: str = BASE_DIR) -> None:
    
    # 1. Carga datasets (Extracción de features y split)
    print("1. Extrayendo características y cargando datos...")
    X_train, X_val, y_train, y_val, clases = crear_datasets_rf(data_dir)

    # 2. Construye el modelo
    model = crear_modelo_rf(class_weight='balanced')
    
    print("\n2. Iniciando entrenamiento de Random Forest...")

    # 3. Entrena el modelo
    model.fit(X_train, y_train)

    # 4. Evaluación
    evaluate_rf(model, X_val, y_val)

    # 5. Guarda el modelo con joblib
    MODEL_DIR = os.path.join(base_dir, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_file_name = f"random_forest_{timestamp}.joblib"
    final_save_path = os.path.join(MODEL_DIR, model_file_name)
    
    joblib.dump(model, final_save_path)
    print(f"\nModelo Random Forest guardado en: {final_save_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo Random Forest")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    args = parser.parse_args()
    run_training(args.data_dir)

if __name__ == "__main__":
    main()