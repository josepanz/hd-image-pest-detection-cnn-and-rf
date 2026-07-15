"""API pública de pest_detection para usar el motor de inferencia desde otro
proyecto Python, sin pasar por la CLI (cli/infer.py delega en este módulo, para no
mantener dos implementaciones de "cómo cargar un modelo y correr inferencia").

No incluye ningún modelo por defecto: siempre hay que apuntar a un .keras/.joblib
propio (entrenado con pest_detection.train o alguno de los que ya haya en
best_models/ de este repo).
"""

import os

import joblib
import tensorflow as tf

from pest_detection.evaluation.inference_utils import load_model_for_inference, run_inference_on_path
from pest_detection.print_utils import print_time_and_step

IMG_SIZE = (224, 224)


class PestDetector:
    """Carga un modelo (CNN .keras o Random Forest .joblib) una sola vez y permite
    predecir múltiples muestras sin recargarlo en cada llamada.

    Args:
        model_path: ruta al .keras (model_type='cnn') o .joblib (model_type='rf').
        model_type: 'cnn' o 'rf'.
        is_multiespectral: True si el modelo espera 5 bandas en vez de una imagen RGB.
        loss: 'fl' o 'bce'. Solo aplica a model_type='rf': indica qué CNN "hermana"
            (misma carpeta que model_path, mismo tipo RGB/MS) usar como extractor de
            features - un RF no clasifica directamente los píxeles, necesita el
            vector de features que produce esa CNN. Se ignora para model_type='cnn'.
    """

    def __init__(self, model_path: str, model_type: str = "cnn", is_multiespectral: bool = False, loss: str = "fl"):
        if model_type not in ("cnn", "rf"):
            raise ValueError(f"model_type debe ser 'cnn' o 'rf', no {model_type!r}")

        self.model_path = model_path
        self.model_type = model_type
        self.is_multiespectral = is_multiespectral
        self.loss = loss
        self.feature_extractor = None

        if model_type == "cnn":
            self.model = load_model_for_inference(model_path)
        else:
            self.model = joblib.load(model_path)
            self.feature_extractor = self._load_companion_cnn(model_path, is_multiespectral, loss)

    @staticmethod
    def _load_companion_cnn(model_path, is_multiespectral, loss):
        """Busca, en la misma carpeta que model_path, la CNN "hermana" (mismo tipo
        RGB/MS, mismo loss) y arma un extractor de features cortándola antes de su
        capa de salida. Devuelve None (con un aviso) si no la encuentra: el RF podría
        fallar al predecir si de verdad necesita esas features."""
        img_mode = "MULTIESPECTRAL" if is_multiespectral else "RGB"
        best_models_dir = os.path.dirname(model_path)
        cnn_suffix = "focal_loss.keras" if "fl" in loss else "binary_crossentropy.keras"
        cnn_for_rf_path = os.path.join(best_models_dir, f"best_model_final_{img_mode}_{cnn_suffix}")

        if not os.path.exists(cnn_for_rf_path):
            print_time_and_step('WARN', f"⚠️ No se encontró la CNN base en {cnn_for_rf_path}. El RF podría fallar si espera features en vez de píxeles crudos.")
            return None

        full_cnn = tf.keras.models.load_model(cnn_for_rf_path, compile=False)
        return tf.keras.Model(inputs=full_cnn.input, outputs=full_cnn.layers[-2].output)

    def predict(self, path: str, threshold: float = 0.5):
        """Corre inferencia sobre `path` (un archivo, o una carpeta - ver
        evaluation/inference_utils.py::run_inference_on_path para el contrato exacto
        de qué cuenta como una muestra válida) y devuelve la lista de resultados (uno
        por muestra encontrada bajo `path`), sin guardar nada a disco."""
        return run_inference_on_path(
            model=self.model,
            feature_extractor_rf=self.feature_extractor,
            path=path,
            threshold=threshold,
            img_size=IMG_SIZE,
            model_name=os.path.basename(self.model_path),
            is_multiespectral=self.is_multiespectral,
            is_random_forest=(self.model_type == "rf"),
        )


def predict(path: str, model_path: str, model_type: str = "cnn", is_multiespectral: bool = False, threshold: float = 0.5, loss: str = "fl"):
    """Atajo de un solo uso: crea un PestDetector, predice, y lo descarta.

    Para predecir muchas muestras con el mismo modelo, usar PestDetector
    directamente - evita recargar el modelo (y la CNN extractora, para RF) en
    cada llamada.
    """
    detector = PestDetector(model_path, model_type=model_type, is_multiespectral=is_multiespectral, loss=loss)
    return detector.predict(path, threshold=threshold)
