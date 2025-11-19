# src/rf/modelrf_multiespectral.py o simplemente src/rf/model.py (reutilizado)

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

def crear_modelo_rf_ms(
    n_estimators: int = 100, 
    max_depth: int = 10,
    random_state: int = 123,
    class_weight: str | dict | None = 'balanced' 
) -> ClassifierMixin:
    """
    Crea un modelo de Random Forest.
    
    Este modelo es agnóstico al tipo de feature (RGB o Multiespectral),
    solo recibe el vector de características.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight, # Manejo de desbalance nativo de RF
        n_jobs=-1
    )
    return model