# src/rf/train.py
import argparse
import os
import joblib # Usamos joblib para guardar modelos sklearn
import numpy as np
import sys
from datetime import datetime

# Ajustar la ruta de importación si es necesario
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from data_loader import crear_datasets
from model import crear_modelo

def entrenar_rf(data_dir: str) -> None:
    """
    Realiza el entrenamiento del modelo Random Forest.
    """
    IMG_SIZE = (224, 224)
    SEED = 123
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    
    # 1. Carga datasets (obtenemos features X y etiquetas y)
    X_train, _, y_train, _, clases = crear_datasets(
        data_dir, img_size=IMG_SIZE, seed=SEED
    )

    # 2. Construye el modelo
    model = crear_modelo(class_weight='balanced', random_state=SEED)
    
    print("\nIniciando entrenamiento de Random Forest...")

    # 3. Entrena el modelo (entrenamiento simple de sklearn)
    model.fit(X_train, y_train)

    print("Entrenamiento completado.")
    
    # 4. Guarda el modelo con joblib
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_file_name = f"random_forest_{timestamp}.joblib"
    final_save_path = os.path.join(MODEL_DIR, model_file_name)
    
    joblib.dump(model, final_save_path)
    print(f"\nModelo Random Forest guardado en '{final_save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo Random Forest")
    parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
    args = parser.parse_args()
    entrenar_rf(args.data_dir)


if __name__ == "__main__":
    main()