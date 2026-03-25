import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from train import train
from evaluate import evaluate
from src.utils.metrics import evaluar_modelo
from src.utils.feature_extractor import extraer_features_cnn
from src.utils.models.random_forest import entrenar_rf, evaluar_rf, model_random_forest

from src.utils.data_management.extract_data_to_img import extract_data_to_img_for_train


def run_experiment(name, data_dir, isRgb, loss_type):

    print(f"\n🔥 Ejecutando: {name}")

    model = train(
        data_dir=data_dir,
        isRgb=isRgb,
        loss_type=loss_type
    )

    X_train, X_val, y_train, y_val, _ = extract_data_to_img_for_train(
        data_dir=data_dir,
        isRgb=isRgb,
        model_type='cnn'
    )

    # CNN evaluation
    y_pred_probs = model.predict(X_val)
    cm, report, auc, fpr, tpr = evaluar_modelo(y_val, y_pred_probs)

    return {
        "name": name,
        "auc": auc
    }, model, X_train, X_val, y_train, y_val


def run_rf_experiment(name, model, X_train, X_val, y_train, y_val):

    print(f"\n🌲 RF sobre: {name}")

    X_train_feat = extraer_features_cnn(model, X_train)
    X_val_feat = extraer_features_cnn(model, X_val)

    # RF training
    model_rf = model_random_forest(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf = entrenar_rf(model_rf, X_train_feat, y_train)
    y_pred, y_prob = evaluar_rf(rf, X_val_feat, y_val)

    from utils.metrics import evaluar_modelo
    cm, report, auc, _, _ = evaluar_modelo(y_val, y_prob)

    return {
        "name": name,
        "auc": auc
    }


def main():

    data_dir = "data/..."

    resultados = []

    # --- CNN MS ---
    res, model_ms_focal, X_train, X_val, y_train, y_val = run_experiment(
        "CNN_MS_FOCAL", data_dir, False, 'focal_loss'
    )
    resultados.append(res)

    # RF MS
    resultados.append(
        run_rf_experiment("RF_MS", model_ms_focal, X_train, X_val, y_train, y_val)
    )

    # --- CNN RGB ---
    res, model_rgb_focal, X_train, X_val, y_train, y_val = run_experiment(
        "CNN_RGB_FOCAL", data_dir, True, 'focal_loss'
    )
    resultados.append(res)

    # RF RGB
    resultados.append(
        run_rf_experiment("RF_RGB", model_rgb_focal, X_train, X_val, y_train, y_val)
    )

    # CNN MS BCE
    res, model_ms_bce, X_train, X_val, y_train, y_val = run_experiment(
        "CNN_MS_BCE", data_dir, False, 'binary_crossentropy'
    )
    resultados.append(res)

    # CNN RGB BCE
    res, model_rgb_bce, X_train, X_val, y_train, y_val = run_experiment(
        "CNN_RGB_BCE", data_dir, True, 'binary_crossentropy'
    )
    resultados.append(res)

    df = pd.DataFrame(resultados)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/comparacion_modelos.csv", index=False)

    print("\n📊 RESULTADOS FINALES")
    print(df)


if __name__ == "__main__":
    main()