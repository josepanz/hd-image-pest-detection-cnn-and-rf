# pest_detection/evaluation/model_loading.py
#
# Carga de modelos ya entrenados (.keras / .joblib) para evaluate.py.
# Antes vivía en utils_inference.py junto con lógica de inferencia que ya no se usa
# (el motor de inferencia vigente es inference_utils.py, usado por infer.py);
# se dejó solo lo que evaluate.py realmente importa.

import os
import tensorflow as tf
import joblib

from pest_detection.focal_loss import focal_loss # misma implementación usada para entrenar/compilar en cnn_model.py
from pest_detection.print_utils import print_time_and_step

CUSTOM_OBJECTS = {
    # focal_loss() usa los valores por defecto de alpha/gamma (no los que se pasaron
    # en el CLI de train.py al entrenar cada modelo en particular, que no se persisten
    # junto al .keras). Esto es suficiente para poder cargar el modelo y predecir,
    # pero el valor de "loss" que se reporte en evaluate.py para un modelo entrenado
    # con otro alpha/gamma no será exactamente el de entrenamiento.
    'focal_loss': focal_loss(),
    # Añadir aquí cualquier otra clase o función personalizada que uses en tus modelos
}

def load_model_for_inference(model_path: str):
    """
    Carga un modelo guardado (.keras o .joblib), manejando objetos personalizados.
    """
    print_time_and_step('1', f"⏳ Cargando modelo desde: {model_path}")
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
            print_time_and_step('2', f"✅ Modelo Keras cargado exitosamente desde {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo Keras: {e}. Asegúrate de que las rutas y custom_objects son correctos.")

    elif model_path.endswith('.joblib'):
        # Modelo Scikit-learn (Random Forest)
        model = joblib.load(model_path)
        print_time_and_step('2', f"✅ Modelo Scikit-learn (Joblib) cargado exitosamente desde {model_path}")
        return model

    else:
        raise ValueError("Formato de modelo no soportado. Use '.keras' o '.joblib'.")
