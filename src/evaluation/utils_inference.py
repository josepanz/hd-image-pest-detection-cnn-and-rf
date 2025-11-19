# src/evaluation/utils_inference.py

import os
import numpy as np
import tensorflow as tf
import joblib
from typing import Union, List, Tuple, Dict, Any
from src.models.function_losses import focal_loss # Necesario para cargar modelos Keras con Focal Loss
import json

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input # Para la extracción de features de RF
from datetime import datetime

CUSTOM_OBJECTS = {
    'focal_loss': focal_loss(), 
    # Añadir aquí cualquier otra clase o función personalizada que uses en tus modelos
}

def load_model_for_inference(model_path: str) -> Union[tf.keras.Model, joblib._io.NumpyArrayWrapper]:
    """
    Carga un modelo guardado (.keras o .joblib), manejando objetos personalizados.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no se encontró en: {model_path}")

    if model_path.endswith(('.keras', '.h5')):
        # Modelo Keras (CNN)
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=CUSTOM_OBJECTS, 
                compile=False # No necesitamos recompilar si solo vamos a predecir
            )
            print(f"Modelo Keras cargado exitosamente desde {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo Keras: {e}. Asegúrate de que las rutas y custom_objects son correctos.")

    elif model_path.endswith('.joblib'):
        # Modelo Scikit-learn (Random Forest)
        model = joblib.load(model_path)
        print(f"Modelo Scikit-learn (Joblib) cargado exitosamente desde {model_path}")
        return model
        
    else:
        raise ValueError("Formato de modelo no soportado. Use '.keras' o '.joblib'.")

def predict_cnn(model: tf.keras.Model, data_ds: tf.data.Dataset, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones con un modelo CNN en un tf.data.Dataset.
    Devuelve probabilidades (y_pred_proba) y etiquetas verdaderas (y_true).
    """
    print("Iniciando predicción del modelo CNN...")
    
    # Extraer etiquetas verdaderas del dataset
    # Usamos .flat_map para aplanar el dataset antes de extraer las etiquetas
    y_true_list = []
    # Nota: Es crucial usar la misma cantidad de pasos que en la predicción para asegurar que y_true sea del mismo tamaño
    for _, y_batch in data_ds.unbatch().as_numpy_iterator():
        y_true_list.append(y_batch)
    
    # Realizar predicciones
    # El modelo Keras ya está compilado para BCE/Focal Loss (salida sigmoid)
    y_pred_proba = model.predict(data_ds, steps=steps, verbose=1).flatten()
    
    # Concatenar las etiquetas verdaderas solo hasta el tamaño de las predicciones
    y_true_flat = np.concatenate(y_true_list).flatten()
    y_true = y_true_flat[:len(y_pred_proba)] # Asegurar tamaños iguales

    return y_pred_proba, y_true


def predict_rf(model: joblib._io.NumpyArrayWrapper, X_features: np.ndarray, Y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones con un modelo Random Forest (Scikit-learn).
    Devuelve etiquetas predichas (y_pred) y etiquetas verdaderas (y_true).
    """
    print("Iniciando predicción del modelo Random Forest...")
    # RF predice directamente las clases (0 o 1)
    y_pred = model.predict(X_features)
    
    return y_pred.flatten(), Y_true.flatten()

def preprocess_single_image(
    img_path: str,
    img_size: Tuple[int, int] = (224, 224),
    is_rf_feature_extraction: bool = False
) -> Union[np.ndarray, None]:
    """
    Carga y preprocesa una imagen para inferencia de CNN (Keras) o RF (Extracción de Features).
    """
    if not os.path.exists(img_path):
        print(f"❌ Archivo no encontrado: {img_path}")
        return None
        
    try:
        # 1. Cargar y Redimensionar
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de Batch

        # 2. Preprocesamiento específico
        if is_rf_feature_extraction:
            # Para RF, necesitamos preprocesar para el extractor MobileNetV2
            return preprocess_input(img_array)
        else:
            # Para CNNs, solo reescalar de 0-255 a 0-1 (usado en tu data_loaderbc.py)
            return img_array / 255.0
            
    except Exception as e:
        print(f"❌ Error al cargar/procesar imagen {img_path}: {e}")
        return None

def run_inference_on_path(
    model: Union[tf.keras.Model, joblib._io.NumpyArrayWrapper],
    feature_extractor_rf: Union[tf.keras.Model, None], # Solo necesario para RF
    path: str,
    threshold: float,
    img_size: Tuple[int, int],
    model_name: str,
    is_multiespectral: bool = False, # Bandera para manejar MS
    is_random_forest: bool = False
) -> List[Dict[str, Any]]:
    """
    Función genérica que itera sobre un archivo o carpeta y ejecuta la inferencia.
    """
    inference_results = []
    CLASSES = ["Plaga", "Sana"] 
    
    # 1. Determinar rutas a procesar
    if os.path.isfile(path):
        paths_to_process = [path]
        base_dir = os.path.dirname(path)
    elif os.path.isdir(path):
        paths_to_process = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        # Si es multiespectral, la ruta 'path' es la carpeta que contiene las bandas.
        if is_multiespectral:
             # Para MS, la carpeta *es* la unidad de inferencia.
            paths_to_process = [path] 
        else:
            paths_to_process = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        base_dir = path
    else:
        print(f"❌ Error: La ruta '{path}' no es un archivo ni una carpeta válida.")
        return []

    print(f"\nIniciando inferencia en {len(paths_to_process)} elementos...")
    
    for item_path in paths_to_process:
        try:
            # --- 2. Carga y Preprocesamiento ---
            if is_multiespectral:
                # Importación específica de MS (debe estar en el entorno)
                from src.data_management.multiespectral.loader_convolutional_neural_network_multiespectral import load_single_multispectral_image
                # Para MS, item_path es la carpeta de las 5 bandas
                X = load_single_multispectral_image(item_path, img_size=img_size)
                file_id = os.path.basename(item_path)
            else:
                # Para RGB/RF, item_path es el archivo de imagen
                X = preprocess_single_image(item_path, img_size, is_random_forest)
                file_id = os.path.basename(item_path)
            
            if X is None:
                continue
                
            X = np.expand_dims(X, axis=0) # Asegurar dimensión de batch (1, H, W, C)

            # --- 3. Extracción de Features (Solo RF) ---
            if is_random_forest and feature_extractor_rf:
                X_features = feature_extractor_rf.predict(X, verbose=0)
                # La predicción de RF espera (1, F) donde F son las features
                probs = model.predict_proba(X_features)[0] 
            
            # --- 4. Predicción (CNNs) ---
            elif not is_random_forest:
                # CNN predice directamente las probabilidades (salida Sigmoid)
                prob_sana = model.predict(X, verbose=0)[0][0]
                # [Prob_Plaga, Prob_Sana]
                probs = np.array([1.0 - prob_sana, prob_sana])
            
            # --- 5. Post-Procesamiento (Común) ---
            if not is_random_forest:
                # Para CNNs, la decisión binaria se hace con el umbral
                prediccion_clase_idx = 1 if prob_sana >= threshold else 0
                etiqueta = CLASSES[prediccion_clase_idx]
            else:
                # Para RF, predict_proba ya da la probabilidad de cada clase
                prediccion_clase_idx = np.argmax(probs) # O usar el umbral sobre probs[1]
                etiqueta = CLASSES[prediccion_clase_idx]

            # --- 6. Guardar Resultados ---
            result = {
                "file_name": file_id,
                "prob_plaga": round(probs[0], 4),
                "prob_sana": round(probs[1], 4),
                "prediccion": etiqueta,
                "umbral": threshold if not is_random_forest else 0.5, # RF es 0.5 por defecto
                "modelo": model_name
            }
            inference_results.append(result)

            # --- 7. Imprimir el resultado ---
            print(f"\n--- Resultados para: {file_id} ---")
            print(f"Prob. '{CLASSES[0]}': {probs[0]:.4f}")
            print(f"Prob. '{CLASSES[1]}': {probs[1]:.4f}")
            print(f"Predicción final: {etiqueta}")
            
        except Exception as e:
            print(f"❌ Error inesperado al procesar {file_id}: {e}")
            
    return inference_results

# --- Función Auxiliar para Guardar ---
def save_inference_results(results: List[Dict[str, Any]], base_dir: str, threshold: float, model_type: str) -> None:
    """Guarda la lista de resultados de inferencia en un archivo JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    umbral_str = f"t{int(threshold * 100):02d}" 

    OUTPUT_DIR = os.path.join(base_dir, f'inference-results-{model_type}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_filename = f"inference_results_{model_type}_{timestamp}_{umbral_str}.json"
    final_json_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(final_json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n\n✅ Resultados de la inferencia guardados en: {final_json_path}")