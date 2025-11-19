# src/rf/trainrf_multiespectral.py

import argparse
import os
import joblib # Usamos joblib para guardar modelos sklearn
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report

# Importaciones clave:
# 1. El nuevo loader Multiespectral para RF
from src.data_management.random_forest.loader_random_forest_multiespectral import crear_datasets_rf_ms 
# 2. El modelo de RF (modelrf.py)
from src.models.model_random_forest_multiespectral import crear_modelo_rf_ms

def entrenar_rf_ms(patch_size: int, base_dir: str) -> None:
    """
    Realiza el entrenamiento del modelo Random Forest Multiespectral.
    """
    SEED = 123
    
    # Directorios de Guardado
    MODEL_DIR = os.path.join(base_dir, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Carga datasets (obtenemos features X y etiquetas y)
    print("1. Cargando y Extrayendo características Multiespectrales...")
    X_train, X_val, y_train, y_val, clases = crear_datasets_rf_ms(
        patch_size=patch_size, 
        seed=SEED,
        test_split=0.2 # Usamos el valor por defecto del loader
    )
    
    # Verificación
    print(f"\nNúmero de características por muestra: {X_train.shape[1]}") # Debe ser 7
    print(f"Clases de entrenamiento: {np.unique(y_train)}")

    # 2. Construye el modelo
    # Utilizamos 'balanced' para manejar el desbalance, ya que no hicimos Undersampling en el loader.
    model = crear_modelo_rf_ms(
        n_estimators=100, # Hiperparámetros de ejemplo
        max_depth=10, 
        class_weight='balanced', 
        random_state=SEED
    )
    
    print("\n2. Iniciando entrenamiento de Random Forest Multiespectral...")

    # 3. Entrena el modelo (entrenamiento simple de sklearn)
    model.fit(X_train, y_train)

    print("✅ Entrenamiento completado.")
    
    # 4. Evaluación en el set de Validación (Para control de calidad)
    y_pred_val = model.predict(X_val)
    report = classification_report(y_val, y_pred_val, target_names=clases, zero_division=0)
    print("\n--- Reporte de Clasificación en Validación ---")
    print(report)
    
    # 5. Guarda el modelo con joblib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Incluimos el número de features (7) en el nombre
    model_file_name = f"random_forest_ms_7features_{timestamp}.joblib" 
    final_save_path = os.path.join(MODEL_DIR, model_file_name)
    
    joblib.dump(model, final_save_path)
    print(f"\n💾 Modelo guardado en: {final_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo Random Forest Multiespectral.")
    # El data_dir no es necesario si load_multiespectral_data busca en una ruta fija (BASE_DATA_DIR)
    # Si load_multiespectral_data necesita la ruta, se debería añadir. Por ahora, asumimos que usa una constante interna.
    parser.add_argument("-p", "--patch_size", type=int, default=224, help="Tamaño del parche (coherente con el loader MS).")
    # Directorio base para los resultados RF Multiespectral
    parser.add_argument("-b", "--base_dir", default="results/random_forest_ms", help="Directorio base para guardar modelos y reportes.") 
    args = parser.parse_args()
    
    entrenar_rf_ms(args.patch_size, args.base_dir)

if __name__ == "__main__":
    main()