# src/rf/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

def crear_modelo(
    n_estimators: int = 100, 
    max_depth: int = 10,
    random_state: int = 123,
    class_weight: str | dict | None = 'balanced' 
) -> ClassifierMixin:
    """
    Crea un modelo de Random Forest.
    
    El parámetro class_weight='balanced' es el equivalente en sklearn a usar 
    class_weighting en Keras para manejar el desbalance.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight, # Manejo de desbalance nativo de RF
        n_jobs=-1
    )
    return model