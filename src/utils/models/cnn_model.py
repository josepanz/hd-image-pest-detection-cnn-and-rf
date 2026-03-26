import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from keras import layers, Model, Input
from keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from src.utils.focal_loss import focal_loss

def crear_modelo_cnn(input_shape, loss_type='focal_loss', alpha=0.25, gamma=2.0):

    inp = Input(shape=input_shape)

    x = layers.Rescaling(1./255)(inp)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=focal_loss(alpha, gamma) if loss_type == 'focal_loss' else 'binary_crossentropy',
        metrics=[
            BinaryAccuracy(name='accuracy'), 
            Precision(name='precision'), 
            Recall(name='recall'), 
            AUC(name='auc')
            ]
    )

    return model