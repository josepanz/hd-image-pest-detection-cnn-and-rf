import tensorflow as tf
import os

def get_callbacks(isRgb=False, loss_type="focal_loss", base_dir: str = ""):
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
        monitor="val_recall",
        verbose=1,
        mode='max'
    )

    return [early_stop, reduce_lr, model_checkpoint]
