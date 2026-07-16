import tensorflow as tf
import os

def get_callbacks(isRgb=False, loss_type="focal_loss", base_dir: str = ""):
    """Callbacks de entrenamiento usados por train.py::train.

    BUG CORREGIDO: monitoreaba val_recall (Keras Recall() sobre la salida sigmoide
    cruda, que con LabelEncoder Plaga=0/Sana=1 mide recall de SANA, no de Plaga).
    La tesis del proyecto es explícita en que el checkpoint debe maximizar el recall
    de PLAGA ("el costo de no detectar una plaga es mayor que el de una falsa
    alarma"). Además, el recall puro (de cualquiera de las dos clases) tiene un techo
    trivial: un modelo que predice siempre la misma clase ya sacaría recall=1.0 de
    esa clase sin ser útil, congelando el checkpoint en una época temprana/poco
    entrenada (confirmado en la práctica). Ahora se monitorea val_f2_plaga
    (F-beta, beta=2, ver pest_detection/models/plaga_metrics.py): sigue priorizando
    recall de Plaga (pesado 4x más que la precisión), pero penaliza ese colapso
    porque la precisión de un modelo así sería pésima. EarlyStopping/ReduceLROnPlateau
    siguen monitoreando val_loss (no cambia).
    """
    model_save_dir = os.path.join(base_dir, 'best_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Ruta donde se guardará el modelo con mejor precisión
    model_path = os.path.join(model_save_dir, f'best_model_final_{"RGB" if isRgb else "MULTIESPECTRAL"}_{loss_type}.keras')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode='min'
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=True,
        monitor="val_f2_plaga",
        verbose=1,
        mode='max'
    )

    return [early_stop, reduce_lr, model_checkpoint]
