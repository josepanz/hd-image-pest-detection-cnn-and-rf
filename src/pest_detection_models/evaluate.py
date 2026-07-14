"""Evaluación de un modelo ya entrenado (CNN o RF) contra un split de validación.

IMPORTANTE: reconstruye el split de validación llamando a extract_data_to_img_for_train
con la MISMA seed/val_split que usa train.py, es decir que asume que se lo está
llamando con el mismo data_dir (y el mismo dataset intacto) que se usó para entrenar
ese modelo - no carga un split guardado. El tipo de modelo (RGB/MS, focal/BCE) se
infiere del NOMBRE del archivo del modelo (substrings 'rgb'/'focal'), no de un flag
explícito; si se renombra el .keras/.joblib sin esas palabras, la detección falla.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# Utils propios
from src.utils.print_utils import print_time_and_step
from src.utils.evaluation.utils_metrics import (
    save_report_and_plot_cm,
    CLASSES,
    plot_roc_curve_and_auc,
)
from src.utils.evaluation.model_loading import load_model_for_inference
from src.utils.data_management.extract_data_to_img import extract_data_to_img_for_train

# Constantes
IMG_SIZE = (224, 224)
SEED = 42
VAL_SPLIT = 0.2


# =========================================================
# MAIN
# =========================================================
def run_evaluation(data_dir, model_path, threshold, model_type, base_dir, batch_size):
    if model_type == 'cnn':
        run_evaluation_cnn(data_dir, model_path, threshold, base_dir, batch_size)
    elif model_type == 'rf':
        run_evaluation_rf(data_dir, model_path, threshold, base_dir)
    else:
        raise ValueError("Modelo no soportado")


# =========================================================
# CNN EVALUATION (FIXED)
# =========================================================
def run_evaluation_cnn(data_dir, model_path, threshold, base_dir, batch_size):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    imgType = 'RGB' if 'rgb' in model_path.lower() else 'MULTIESPECTRAL'
    loss_type = 'focal_loss' if 'focal' in model_path.lower() else 'binary_crossentropy'
    isRgb = 'rgb' in model_path.lower()

    print_time_and_step('init',
        f'Evaluando CNN {imgType} - Loss: {loss_type} - Threshold: {threshold}',
        timestamp, start_time
    )

    # ✅ USAR EXACTAMENTE EL MISMO PIPELINE QUE TRAIN
    print_time_and_step('1', 'Extrayendo datos (MISMO pipeline que training)...', timestamp, start_time)

    X_train, X_val, y_train, y_val, _ = extract_data_to_img_for_train(
        data_dir=data_dir,
        isRgb=isRgb,
        model_type='cnn',
        batch_size=batch_size,
        img_size=IMG_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    print("Distribución y_val:", dict(zip(*np.unique(y_val, return_counts=True))))

    # Modelo
    print_time_and_step('2', 'Cargando modelo...', timestamp, start_time)
    model = load_model_for_inference(model_path)
    model_name = os.path.basename(model_path)

    # Predicción
    print_time_and_step('3', 'Prediciendo...', timestamp, start_time)
    y_prob = model.predict(X_val, verbose=1).flatten()

    # DEBUG CRÍTICO
    print("\n--- DEBUG PROBABILIDADES ---")
    print("MIN:", np.min(y_prob))
    print("MAX:", np.max(y_prob))
    print("MEAN:", np.mean(y_prob))
    print("PERCENTILES:", np.percentile(y_prob, [0, 25, 50, 75, 100]))

    print("X_val stats:")
    print("MIN:", np.min(X_val))
    print("MAX:", np.max(X_val))
    print("MEAN:", np.mean(X_val))

    print("STD:", np.std(y_prob))

    # Threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Resultados
    RESULTS_DIR = os.path.join(base_dir, f'evaluation_results/CNN/{imgType}/{loss_type}/{threshold}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_time_and_step('4', 'Generando métricas...', timestamp, start_time)

    save_report_and_plot_cm(
        y_val,
        y_pred,
        CLASSES,
        RESULTS_DIR,
        model_name,
        threshold
    )

    plot_roc_curve_and_auc(
        y_val,
        y_prob,
        RESULTS_DIR,
        model_name,
        threshold
    )


# =========================================================
# RF EVALUATION (FIXED)
# =========================================================
def run_evaluation_rf(data_dir, model_path, threshold, base_dir):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    imgType = 'RGB' if 'rgb' in model_path.lower() else 'MULTIESPECTRAL'
    isRgb = 'rgb' in model_path.lower()

    print_time_and_step('init',
        f'Evaluando RANDOM FOREST {imgType}',
        timestamp, start_time
    )

    # =========================================================
    # 1. DATA (MISMO PIPELINE QUE TRAINING)
    # =========================================================
    X_train, X_val, y_train, y_val, _ = extract_data_to_img_for_train(
        data_dir=data_dir,
        isRgb=isRgb,
        model_type='cnn',  # 🔥 IMPORTANTE: CNN porque necesitas imágenes
        img_size=IMG_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
        batch_size=32
    )

    print("Distribución y_val:", dict(zip(*np.unique(y_val, return_counts=True))))

    # =========================================================
    # 2. LOAD MODEL
    # =========================================================
    print_time_and_step('2', 'Cargando modelo...', timestamp, start_time)

    bundle = joblib.load(model_path)

    rf_model = bundle["rf_model"]
    scaler = bundle["scaler"]
    feature_extractor = bundle["feature_extractor"]

    print("\n--- DEBUG RF ---")
    print("X_val shape:", X_val.shape)
    print("X_val min:", np.min(X_val))
    print("X_val max:", np.max(X_val))

    # =========================================================
    # 3. FEATURE EXTRACTION (🔥 ESTO TE FALTABA)
    # =========================================================
    print_time_and_step('3', 'Extrayendo features...', timestamp, start_time)

    X_val_feat = feature_extractor.predict(X_val, verbose=1)

    print("Features shape:", X_val_feat.shape)

    # =========================================================
    # 4. SCALING (MISMO QUE TRAIN)
    # =========================================================
    X_val_feat = scaler.transform(X_val_feat)

    # =========================================================
    # 5. PREDICCIÓN
    # =========================================================
    print_time_and_step('4', 'Prediciendo...', timestamp, start_time)

    y_prob = rf_model.predict_proba(X_val_feat)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n--- DEBUG PROB ---")
    print("MIN:", np.min(y_prob))
    print("MAX:", np.max(y_prob))
    print("STD:", np.std(y_prob))

    # =========================================================
    # 6. MÉTRICAS (mismas funciones que usa la evaluación CNN, ver arriba)
    # =========================================================
    print_time_and_step('4', 'Evaluación y Reporte de Random Forest...', timestamp=timestamp, start_time=start_time)

    RESULTS_DIR = os.path.join(base_dir, f'evaluation_results/RANDOM_FOREST/{imgType}/{threshold}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_name = os.path.basename(model_path)

    save_report_and_plot_cm(y_val, y_pred, CLASSES, RESULTS_DIR, model_name, threshold)
    plot_roc_curve_and_auc(y_val, y_prob, RESULTS_DIR, model_name, threshold)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-mt", "--model_type", required=True, choices=["cnn", "rf"])
    parser.add_argument("-b", "--base_dir", default=BASE_DIR)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)

    args = parser.parse_args()

    run_evaluation(
        args.data_dir,
        args.model,
        args.threshold,
        args.model_type,
        args.base_dir,
        args.batch_size
    )