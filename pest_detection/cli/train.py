"""Entrenamiento de los modelos de detección de plaga/sana (CNN o Random Forest).

Dos modos, seleccionados con -mt/--model_type:
- cnn: entrena una CNN (MobileNet-simple definida en models/cnn_model.py) desde cero,
  con Focal Loss o Binary Crossentropy, sobre imágenes RGB o multiespectrales (5 bandas).
- rf: entrena un Random Forest sobre features extraídas por una CNN YA ENTRENADA.
  run_rf_training carga desde disco (best_models/best_model_final_{RGB|MULTIESPECTRAL}_
  {loss_type}.keras) la CNN correspondiente al mismo tipo de imagen/loss indicado por
  -lt; si ese archivo no existe todavía (no se corrió antes el modo cnn con esa
  combinación), esto falla con FileNotFoundError. El RF no entrena una CNN propia.

El umbral de decisión (0.5 por defecto) NO se fija acá: entrenar no lo usa para nada,
solo evaluate.py/inference_models.py lo aplican al momento de clasificar. train()
además calcula su propio "umbral óptimo" post-entrenamiento (ver encontrar_umbral_optimo)
solo a modo de referencia/diagnóstico, no como umbral que se guarda con el modelo.
"""

import argparse
import os

import keras
# Importaciones de Módulos
from pest_detection.datasets.extract_data_to_img import extract_data_to_img_for_train
from pest_detection.models.cnn_model import crear_modelo_cnn
from pest_detection.callbacks import get_callbacks
from pest_detection.utils_train import encontrar_umbral_optimo, save_history_and_plot
from pest_detection.datasets.class_weights import calculate_class_weights
from pest_detection.evaluation.utils_metrics import save_report_and_plot_cm, plot_roc_curve_and_auc, CLASSES
from pest_detection.metrics import evaluar_modelo
import random
import numpy as np
import tensorflow as tf

from pest_detection.print_utils import print_time_and_step
from pest_detection.models.random_forest import entrenar_rf, evaluar_rf, model_random_forest
import joblib
from sklearn.preprocessing import StandardScaler

import time
from datetime import datetime

# CONSTANTS
SEED = 42
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.2
BATCH_SIZE = 32

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Si usas operaciones muy específicas de GPU:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds(SEED) # Usa el número que quieras, pero mantenlo


def run_training(data_dir: str, epochs: int, loss_type: str, isRgb: bool, alpha: float, gamma: float, model_type: str, base_dir: str = None) -> None:
  """Despacha a train() (CNN) o run_rf_training() (Random Forest) según model_type."""
  base_dir = base_dir if base_dir is not None else os.getcwd()
  if model_type == 'cnn':
    print(f'Modelo cnn {loss_type} | RGB? {isRgb} | alpha {alpha} | gamma {gamma}')
    train(
      data_dir=data_dir,
      epochs=epochs,
      loss_type=loss_type,
      isRgb=isRgb,
      alpha=alpha,
      gamma=gamma,
      base_dir=base_dir
    )
  elif model_type == 'rf':
      # BUG CORREGIDO: -lt/--loss_type no tiene default en el CLI (None si se omite,
      # ver main() más abajo) - pasar ese None directo a run_rf_training rompía con
      # ValueError("File not found: ...best_model_final_MULTIESPECTRAL_None.keras"),
      # exactamente el comando documentado en EJECUCION.md/README ("-mt rf" sin
      # "-lt"). Default a 'focal_loss' (misma convención que infer.py/api.py usan
      # para RF cuando no se especifica -l/--loss) si no se pasó explícitamente.
      rf_loss_type = loss_type if loss_type is not None else 'focal_loss'
      print(f'Modelo Random Forest con extracción CNN (loss_type={rf_loss_type})')
      run_rf_training(
        data_dir=data_dir,
        isRgb=isRgb,
        base_dir=base_dir,
        model_loss_type=rf_loss_type
        )
  else:
      raise ValueError(f"Tipo de modelo '{model_type}' no soportado.")

def train(
    data_dir,
    isRgb=False,
    loss_type='focal_loss',
    alpha=0.25,
    gamma=2.0,
    epochs=20,
    base_dir=None
):
    """Entrena la CNN desde cero y la guarda en best_models/ vía el ModelCheckpoint
    de get_callbacks (que monitorea val_f2_plaga, no val_loss/val_accuracy: en este
    dominio un falso negativo -plaga clasificada como sana- es el error caro, ver
    pest_detection/models/plaga_metrics.py y callbacks.py para el detalle).

    Si loss_type == 'focal_loss' no se pasa class_weight a model.fit: alpha ya
    compensa el desbalance de clases dentro de la propia función de pérdida, así
    que aplicar además class_weight sería doblemente penalizar la clase minoritaria.
    class_weight solo se usa con binary_crossentropy, que no tiene ese mecanismo propio.
    """
    base_dir = base_dir if base_dir is not None else os.getcwd()
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    input_shape = (224, 224, 3) if isRgb else (224, 224, 5)
    
    print_time_and_step('init', f'Iniciando entrenamiento {"RGB" if isRgb else "MULTIESPECTRAL"} con perdida {"Focal Loss" if loss_type == "focal_loss" else "Binary Crossentropy"}', timestamp=timestamp, start_time=start_time)
    print_time_and_step('1', f"1. Extrayendo datos", timestamp=timestamp, start_time=start_time)
    X_train, X_val, y_train, y_val, _ = extract_data_to_img_for_train(
        data_dir=data_dir,
        isRgb=isRgb,
        model_type='cnn',
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    print(f"Shape train: {X_train.shape}")

    model = crear_modelo_cnn(
        input_shape=input_shape,
        loss_type=loss_type,
        alpha=alpha,
        gamma=gamma
    )

    # Calcular conteos para reportes (no para class_weight, ya que RF usa diferentes mecanismos)
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    class_weight = calculate_class_weights(train_counts[0], train_counts[1])

    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(isRgb=isRgb, loss_type=loss_type, base_dir=base_dir),
        class_weight=None if loss_type == 'focal_loss' else class_weight,
        verbose=1
    )

    # 5. Guardado y Ploteo (Usando utils_train)
    print_time_and_step('5', 'Guardado y Ploteo (Usando utils_train)', timestamp=timestamp, start_time=start_time)
    suffix = f"_Focal_a{alpha}_g{gamma}" if loss_type == 'focal_loss' else "_BCE"
    save_history_and_plot(history, base_dir, epochs, suffix=suffix, isRgb=isRgb, loss_type=loss_type)

    # BUG CORREGIDO: hasta acá 'model' tiene los pesos que restauró EarlyStopping
    # (restore_best_weights=True, criterio val_loss) - NO necesariamente los que
    # ModelCheckpoint grabó en disco (criterio val_f2_macro), pueden ser épocas
    # distintas. Recargamos el .keras recién guardado para que el reporte post-train
    # de acá abajo (y el umbral óptimo) reflejen el checkpoint real que va a usar
    # evaluate.py/infer.py después, no una época distinta que quedó solo en memoria.
    model_path = os.path.join(base_dir, 'best_models', f'best_model_final_{"RGB" if isRgb else "MULTIESPECTRAL"}_{loss_type}.keras')
    model = keras.models.load_model(model_path, compile=False)

    umbral_maestro = encontrar_umbral_optimo(model, X_train, y_train)
    print('Umbral mas optimo: ', umbral_maestro)

    # Reporte/matriz de confusión/ROC de esta validación post-entrenamiento, reutilizando
    # las mismas funciones que usa evaluate.py (antes esto se recalculaba de forma
    # separada e incompleta en post_train.py).
    imgType = 'RGB' if isRgb else 'MULTIESPECTRAL'
    val_results_dir = os.path.join(base_dir, 'evaluation_results', 'CNN', imgType, loss_type, 'post_train_val')
    os.makedirs(val_results_dir, exist_ok=True)
    model_label = f"post_train_{imgType}_{loss_type}"

    y_val_probs = model.predict(X_val).ravel()
    y_val_pred = (y_val_probs >= umbral_maestro).astype(int)
    save_report_and_plot_cm(y_val, y_val_pred, CLASSES, val_results_dir, model_label, umbral_maestro)
    plot_roc_curve_and_auc(y_val, y_val_probs, val_results_dir, model_label, umbral_maestro)

    return model

def run_rf_training(data_dir: str, isRgb: bool, base_dir: str, model_loss_type: str) -> None:
    """Entrena un Random Forest sobre features extraídas por una CNN ya entrenada.

    Requiere que best_models/best_model_final_{RGB|MULTIESPECTRAL}_{model_loss_type}.keras
    ya exista (entrenado antes con `train.py ... -mt cnn -lt <model_loss_type>` [-rgb]).
    El "feature extractor" es esa misma CNN cortada antes de su capa de salida
    (penúltima capa, ver cnn_model.crear_modelo_cnn), no un modelo nuevo.

    Guarda un único .joblib con un dict {rf_model, scaler, feature_extractor}: hace
    falta el feature_extractor completo (no solo el RF) porque para clasificar una
    imagen nueva primero hay que pasarla por la misma CNN para obtener el vector de
    features antes de dárselo al RF.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print_time_and_step('1', f'Iniciando entrenamiento Random Forest {"RGB" if isRgb else "MULTIESPECTRAL"}', timestamp=timestamp, start_time=start_time)

    # =========================================================
    # 1. CARGA DE DATOS
    # =========================================================
    print_time_and_step('1', "Cargando datos...", timestamp=timestamp, start_time=start_time)

    X_train, X_val, y_train, y_val, _ = extract_data_to_img_for_train(
        data_dir=data_dir,
        isRgb=isRgb,
        model_type='cnn',
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    print(f"Shape train: {X_train.shape}")

    # =========================================================
    # 2. CARGAR CNN PREENTRENADA (feature extractor)
    # =========================================================
    print_time_and_step('2', "Cargando modelo CNN para extracción de features...", timestamp=timestamp, start_time=start_time)

    cnn_model_path = os.path.join(base_dir, "best_models", f"best_model_final_{'RGB' if isRgb else 'MULTIESPECTRAL'}_{model_loss_type}.keras")

    cnn_model = tf.keras.models.load_model(cnn_model_path, compile=False)

    # Cortamos la última capa (clasificación)
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-2].output
    )

    print("Feature extractor listo ✔")

    # =========================================================
    # 3. EXTRAER FEATURES
    # =========================================================
    print_time_and_step('3', "Extrayendo features con CNN...", timestamp=timestamp, start_time=start_time)

    X_train_feat = feature_extractor.predict(X_train, verbose=1)
    X_val_feat = feature_extractor.predict(X_val, verbose=1)

    print(f"Shape features train: {X_train_feat.shape}")

    # =========================================================
    # 4. ESCALADO (MUY IMPORTANTE)
    # =========================================================
    print_time_and_step('4', "Aplicando StandardScaler...", timestamp=timestamp, start_time=start_time)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_val_feat = scaler.transform(X_val_feat)

    # =========================================================
    # 5. CLASS WEIGHT
    # =========================================================
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    class_weight = calculate_class_weights(train_counts[0], train_counts[1])

    # =========================================================
    # 6. ENTRENAR RANDOM FOREST
    # =========================================================
    print_time_and_step('5', "Entrenando Random Forest...", timestamp=timestamp, start_time=start_time)

    rf = model_random_forest(n_estimators=300, max_depth=15, random_state=42, class_weight=class_weight)
    rf = entrenar_rf(rf, X_train_feat, y_train)

    print_time_and_step('Finish', "Entrenamiento completado ✔", timestamp=timestamp, start_time=start_time)

    # =========================================================
    # 7. EVALUACIÓN
    # =========================================================
    print_time_and_step('6', "Evaluando modelo...", timestamp=timestamp, start_time=start_time)

    y_prob = rf.predict_proba(X_val_feat)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm, report, auc_score, _, _ = evaluar_modelo(y_val, y_prob)

    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(cm)

    print("\n--- REPORTE DE CLASIFICACIÓN ---")
    print(report)

    print(f"\nAUC: {auc_score:.4f}")

    # =========================================================
    # 8. GUARDADO COMPLETO (RF + SCALER)
    # =========================================================
    print_time_and_step('7', "Guardando modelo...", timestamp=timestamp, start_time=start_time)

    MODEL_DIR = os.path.join(base_dir, 'best_models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_bundle = {
        "rf_model": rf,
        "scaler": scaler,
        "feature_extractor": feature_extractor
    }

    model_file_name = f"best_model_final_random_forest_{'RGB' if isRgb else 'MULTIESPECTRAL'}_{timestamp}.joblib"
    final_save_path = os.path.join(MODEL_DIR, model_file_name)

    joblib.dump(model_bundle, final_save_path)

    print(f"\nModelo RF + Scaler guardado en: {final_save_path}")

def main():
  parser = argparse.ArgumentParser(description="Entrena el modelo RGB o MULTIESPECTRAL para detección de plagas")
  parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
  parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
  parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
  parser.add_argument("-g", "--gamma", type=float, default=3.0, help="Gamma")
  parser.add_argument("-lt", "--loss_type", type=str, choices=["focal_loss", "binary_crossentropy"], help="Tipo de funcion de perdida")
  parser.add_argument("-rgb", "--rgb", action='store_true', default=False, help="Es RGB?")
  parser.add_argument("-mt", "--model_type", type=str, required=True, default='cnn', choices=["cnn", "rf"], help="Tipo de modelo a entrenar (cnn o rf)")
  parser.add_argument("-b", "--base_dir", default=os.getcwd(), help="Directorio donde se crean best_models/, evaluation_results/, history/ (por defecto, el directorio actual).")
  args = parser.parse_args()
  run_training(
    data_dir=args.data_dir,
    epochs=args.epochs,
    loss_type=args.loss_type,
    isRgb=args.rgb,
    alpha=args.alpha,
    gamma=args.gamma,
    model_type=args.model_type,
    base_dir=args.base_dir
    )

if __name__ == "__main__":
  main()