# src/rf/inference_rf.py

import argparse
import os
import tensorflow as tf # Necesario para el feature extractor
from src.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results
from src.data_management.random_forest.loader_random_forest_rgb import crear_feature_extractor # Reutiliza la función del data loader

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo Random Forest.")
    parser.add_argument("path", help="Ruta a un archivo de imagen o carpeta con imágenes RGB.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Random Forest (.joblib).")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carga del Modelo RF
    rf_model = load_model_for_inference(args.model)
    model_name = os.path.basename(args.model).replace('.joblib', '')
    
    # 2. Carga del Extractor de Características MobileNetV2
    feature_extractor = crear_feature_extractor(input_shape=(224, 224, 3))
    
    # 3. Ejecutar Inferencia
    # El umbral no se usa en RF, pero se pasa por convención.
    results = run_inference_on_path(
        model=rf_model,
        feature_extractor_rf=feature_extractor,
        path=args.path,
        threshold=0.5,
        img_size=(224, 224),
        model_name=model_name,
        is_multiespectral=False,
        is_random_forest=True
    )

    # 4. Guardar Resultados
    if results:
        save_inference_results(results, BASE_DIR, 0.5, 'rf')
        
if __name__ == "__main__":
    main()