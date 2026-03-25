from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin
from typing import Union, Dict

def entrenar_rf(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def model_random_forest(
    n_estimators: int = 200, 
    max_depth: int = 10,
    random_state: int = 42, #123
    class_weight: Union[str, Dict, None] = 'balanced' 
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
        n_jobs=-1 # Usa todos los núcleos disponibles
    )
    return model

def evaluar_rf(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]

    return y_pred, y_prob