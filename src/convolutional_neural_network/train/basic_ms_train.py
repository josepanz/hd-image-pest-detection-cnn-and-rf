from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from math import ceil

# Ajustar la ruta de importación si es necesario
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.extract_data_to_img import MultiespectralDataGenerator


# --- CONFIGURACIÓN DE RUTAS Y CONSTANTES (Deben coincidir con tu script principal) ---
# Define los sufijos de tus archivos para cada banda (Red, Red Edge, NIR)
TARGET_SIZE = (128, 128) 
# Ruta base de tus datos ráster (donde están las carpetas de fecha)
BASE_DIR_RASTER = r'data\multispectral_images\TTADDA_NARO_2023_F1\drone_data'
SHP_ID_COLUMN = 'PlotID' # Columna de match en el Shapefile

# --- SIMULACIÓN DE CARGA DE DATOS (Basado en tu código anterior) ---
# Necesitas: labels_df (con obsUnitId_num), parcels_gdf (con SHP_MATCH_ID), y los arrays de etiquetas.

# 0. Carga y Preparación (asumiendo que estos objetos ya existen)
labels_df = pd.read_csv(r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\measurements\generated_labels_unified.csv') # Carga tu CSV
parcels_gdf = gpd.read_file(r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\metadata\plot_shapefile.shp') # Carga tu SHP
# ... (Código para limpiar y filtrar dataframes y preparar GeoDataFrame)

# 1. Preparación de datos para el generador
df_train_full = labels_df[labels_df['Etiqueta_FINAL'].isin(['Plaga', 'Sana'])].copy()
df_train_full['obsUnitId_num'] = df_train_full['obsUnitId'].astype(str).str.extract(r'P(\d+)$').fillna('0').astype(str) # Simular la creación

# 2. Codificación de etiquetas (LabelEncoder)
le = LabelEncoder()
# Entrenar el encoder con TODAS las etiquetas elegibles para asegurar el mapeo (0, 1)
le.fit(df_train_full['Etiqueta_FINAL']) 
print(f"Clases codificadas: {le.classes_}") 

# 3. División de datos (Usamos el DF completo para hacer el split)
df_train, df_val = train_test_split(
    df_train_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_train_full['Etiqueta_FINAL']
)

print(f"\nMuestras totales: {len(df_train_full)}")
print(f"\nMuestras de Entrenamiento: {len(df_train)}")
print(f"Muestras de Validación: {len(df_val)}")

# 4. Inicialización de los Generadores
BATCH_SIZE = 32

train_generator = MultiespectralDataGenerator(
    df=df_train, 
    parcels_gdf=parcels_gdf,
    base_dir_raster=BASE_DIR_RASTER, 
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    le=le,
    shuffle=True
)

validation_generator = MultiespectralDataGenerator(
    df=df_val, 
    parcels_gdf=parcels_gdf,
    base_dir_raster=BASE_DIR_RASTER, 
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    le=le,
    shuffle=False # No se debe mezclar el conjunto de validación
)

# 5. Uso en un Modelo Keras (Ejemplo de CNN)

# Asumimos una CNN sencilla
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), # 3 canales de entrada
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(len(le.classes_), activation='softmax') # 2 clases (Plaga/Sana)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Usamos sparse porque las etiquetas son números enteros (0, 1)
              metrics=['accuracy'])

print("\nModelo listo para entrenar con Generadores.")

# ENTRENAMIENTO
# history = model.fit(
#     train_generator,
#     epochs=10, # Ajusta el número de épocas
#     validation_data=validation_generator
# )
