"""Detección de plagas en cultivos de papa (CNN y Random Forest, RGB y multiespectral).

Uso típico como librería (sin pasar por la CLI):

    from pest_detection import PestDetector

    detector = PestDetector("best_models/best_model_final_RGB_focal_loss.keras", model_type="cnn")
    resultados = detector.predict("ruta/a/una/imagen_rgb.tif")

`predict()` es un atajo de un solo uso si no hace falta reutilizar el modelo cargado.
`train`/`evaluate` reexportan los mismos scripts que corren `pest-train`/`pest-evaluate`.
Ninguna de las tres funciones/clases incluye un modelo por defecto: siempre hay que
apuntar a un .keras/.joblib propio.
"""

from pest_detection.api import PestDetector, predict
from pest_detection.cli.train import run_training as train
from pest_detection.cli.evaluate import run_evaluation as evaluate

__version__ = "0.1.0"

__all__ = ["PestDetector", "predict", "train", "evaluate"]
