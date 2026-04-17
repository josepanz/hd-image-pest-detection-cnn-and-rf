import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional

# Extensiones y sufijos para bandas multiespectrales (Ajustar según tu dataset)
BAND_SUFFIXES = [
    "_blue.tif",
    "_green.tif",
    "_red.tif",
    "_red edge.tif",
    "_nir.tif"
]

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from src.utils.print_utils import print_time_and_step
import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def load_model_for_inference(model_path: str):
    """Carga un modelo de Keras para inferencia."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        raise e

def is_sample_folder(path, is_ms):
    """
    Determina si una carpeta es una muestra procesable.
    Si es MS: Debe contener las 5 bandas.
    Si es RGB: Debe contener al menos un archivo que termine en 'rgb.tif'.
    """
    files = [f.lower() for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if is_ms:
        # Verifica que estén los 5 sufijos
        return all(any(f.endswith(s) for f in files) for s in BAND_SUFFIXES)
    else:
        # Verifica que exista el archivo rgb.tif
        return any(f.endswith("rgb.tif") for f in files)

def get_all_sample_folders(root_path, is_ms):
    """
    Busca recursivamente todas las carpetas que cumplan con ser una muestra.
    """
    sample_folders = []
    
    # Si el path ya es una muestra, lo retornamos directamente
    if os.path.isdir(root_path) and is_sample_folder(root_path, is_ms):
        return [root_path]

    # Si no, caminamos por el árbol de directorios
    for root, dirs, files in os.walk(root_path):
        if is_sample_folder(root, is_ms):
            sample_folders.append(root)
            # Si encontramos una carpeta de muestra, no seguimos entrando en sus subcarpetas
            # para evitar duplicados o errores
            dirs[:] = [] 
            
    return sample_folders

def run_inference_on_path(model, feature_extractor_rf, path, threshold, img_size, model_name, is_multiespectral, is_random_forest):
    results = []
    
    # 1. Identificar todas las muestras (recursivo)
    print_time_and_step('riop 1', f"🔎 Buscando muestras en: {path}", timestamp=timestamp, start_time=start_time)
    sample_folders = get_all_sample_folders(path, is_multiespectral)
    print_time_and_step('riop 2', f"✅ Se encontraron {len(sample_folders)} muestras para procesar.", timestamp=timestamp, start_time=start_time)

    # 2. Extraer modelo del diccionario (RF)
    actual_model = model
    if is_random_forest and isinstance(model, dict):
        actual_model = model.get('model') or model.get('rf_model') or list(model.values())[0]

    # 3. Procesar cada carpeta encontrada
    for sample_path in sample_folders:
        print_time_and_step('riop 3', f"🚀 Procesando muestra: {os.path.basename(sample_path)}", timestamp=timestamp, start_time=start_time)
        
        # Cargamos la data (load_and_preprocess_image ya sabe manejar carpetas MS o RGB)
        img_data = load_and_preprocess_image(sample_path, img_size, is_multiespectral)
        
        if img_data is not None:
            x = np.expand_dims(img_data, axis=0)
            
            if is_random_forest:
                if feature_extractor_rf is not None:
                    features = feature_extractor_rf.predict(x, verbose=0)
                    x_input = features.reshape(1, -1)
                else:
                    x_input = x.reshape(1, -1)

                try:
                    probs = actual_model.predict_proba(x_input)[0]
                    prob_sana = float(probs[1]) 
                except ValueError as e:
                    print_time_and_step('riop 4', f"❌ Error de dimensiones en RF: {e}", timestamp=timestamp, start_time=start_time)
                    continue
            else:
                prediction = model.predict(x, verbose=0)
                prob_sana = float(prediction[0][0])
            
            prob_plaga = 1.0 - prob_sana
            prediccion = "Sana" if prob_sana >= threshold else "Plaga"

            results.append({
                "file_name": os.path.basename(sample_path), # Nombre de la carpeta (ej. 2021-05-25)
                "path": sample_path,
                "prob_sana": round(prob_sana, 4),
                "prob_plaga": round(prob_plaga, 4),
                "prediccion": prediccion,
                "umbral": threshold,
                "modelo": model_name
            })
            
    return results

def load_and_preprocess_image(path, img_size, is_ms):
    """
    Carga y preprocesa una imagen. 
    Soporta carpeta de bandas para MS y busca el archivo rgb.tif para RGB.
    """
    try:
        if is_ms:
            # Lógica Multiespectral: Cargar las 5 bandas desde la carpeta 'path'
            bands = []
            for suffix in BAND_SUFFIXES:
                band_path = next((os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(suffix)), None)
                
                if band_path is None:
                    raise FileNotFoundError(f"Falta banda {suffix} en {path}")
                
                band = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
                if band is None: raise ValueError(f"No leíble: {band_path}")
                
                band_resized = cv2.resize(band, img_size).astype('float32') / 255.0
                bands.append(band_resized)
            
            return np.stack(bands, axis=-1)
            
        else:
            # Lógica RGB: 'path' es la carpeta, necesitamos encontrar el archivo rgb.tif
            target_file = None
            if os.path.isdir(path):
                # Buscamos el archivo que termina en rgb.tif dentro de la carpeta
                target_file = next((os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith("rgb.tif")), None)
            elif os.path.isfile(path):
                target_file = path

            if target_file is None:
                raise FileNotFoundError(f"No se encontró un archivo RGB válido en {path}")

            img = cv2.imread(target_file)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen RGB: {target_file}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, img_size).astype('float32') / 255.0

    except Exception as e:
        print_time_and_step('riop err 1', f"❌ Error procesando {path}: {e}", timestamp=timestamp, start_time=start_time)
        return None

def save_inference_results(results, output_dir, threshold, img_type, model_type):
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results_{img_type}_{model_type}_t{threshold}_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print_time_and_step('riop 5', f"📂 JSON guardado en: {path}", timestamp=timestamp, start_time=start_time)