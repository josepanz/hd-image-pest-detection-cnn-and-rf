import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, accuracy_score
import cv2

from sklearn.preprocessing import LabelEncoder
from keras.utils import Sequence
from math import ceil

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.metrics import Precision, Recall, BinaryAccuracy

from utils_train import create_cnn_callbacks, save_history_and_plot
import argparse
# import inspeccionar_tif

# --- CONFIGURACIÓN DE RUTAS ---
# 1. Usar r-strings para evitar errores de barras invertidas en Windows.
LABELS_CSV = r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\measurements\generated_labels_unified.csv'
PARCELS_SHP = r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\metadata\plot_shapefile.shp'
BASE_DIR_RASTER = r'data\multispectral_images\TTADDA_NARO_2023_F1\drone_data' 
#TIF_SUFFIX = '.tif' # Define qué TIF usarás (puedes cambiarlo a '_WUR_transparent_reflectance_nir.tif')
BAND_SUFFIXES = ['red.tif', 'red edge.tif', 'nir.tif']

# 2. DEFINICIÓN DE LA COLUMNA DE UNIÓN DEL SHAPEFILE
# ¡AJUSTA ESTO! Debe ser el nombre exacto de la columna en tu SHP que tiene el número de parcela.
SHP_ID_COLUMN = 'PlotID' # <--- EJEMPLO: Revisa y ajusta este nombre.

# 3. DIMENSIÓN DE SALIDA PARA EL MODELO CNN (Necesario para el entrenamiento)
TARGET_SIZE = (128, 128) 

def extract_data_to_img_for_train():
  # --- 1. CARGA DE DATOS ---
  labels_df = pd.read_csv(LABELS_CSV)
  parcels_gdf = gpd.read_file(PARCELS_SHP)

  # Preparación del Shapefile para la unión: convertir la columna del SHP a string
  parcels_gdf['SHP_MATCH_ID'] = parcels_gdf[SHP_ID_COLUMN].astype(str) 
  # Preparación del CSV: aseguramos que el campo recién agregado también sea string
  labels_df['obs_unit_id_num'] = labels_df['obs_unit_id_num'].astype(str)


  # --- 2. FILTRADO Y EXTRACCIÓN ---
  df_train = labels_df[labels_df['Etiqueta_FINAL'].isin(['Plaga', 'Sana'])].copy()

  X_images = [] 
  y_labels = [] 

  print(f"Iniciando procesamiento de {len(df_train)} imágenes etiquetadas...")

  for index, row in df_train.iterrows():
      fecha = row['Fecha']
      # Usamos directamente la nueva columna del CSV
      obs_unit_id_num = row['obs_unit_id_num'] 
      etiqueta = row['Etiqueta_FINAL']

      # 3. CONSTRUCCIÓN DE RUTA TIF
      tif_folder = os.path.join(BASE_DIR_RASTER, fecha)

      # 4. BÚSQUEDA DE POLÍGONO (Unión directa por ID numérico)
      parcela = parcels_gdf[parcels_gdf['SHP_MATCH_ID'] == obs_unit_id_num]
      
      if parcela.empty:
          print(f"Polígono no encontrado en SHP para ID numérico: {obs_unit_id_num}")
          continue
      
      geometries = parcela.geometry.values
      # inspeccionar_tif(tif_path)

      all_bands_clipped = [] # Lista temporal para guardar los recortes de cada banda
      
      # 5. EXTRACCIÓN Y APILAMIENTO DE TRES BANDAS (Red, Red Edge, NIR)
      try:
          tif_date_prefix = fecha.replace('-', '') # Ej: '20230605'
          
          for suffix in BAND_SUFFIXES:
              # Construir la ruta al archivo de la banda específica
              tif_name = f"{tif_date_prefix}_{suffix}" # Ej: 20230605_red.tif
              tif_path = os.path.join(BASE_DIR_RASTER, fecha, tif_name)

              if fecha == "2023-05-18":
                tif_name = tif_date_prefix + '_WUR_' + 'transparent_reflectance_' + suffix
                tif_path = os.path.join(tif_folder, tif_name)
              else:
                tif_name = tif_date_prefix + '_transparent_reflectance_' + suffix
                tif_path = os.path.join(tif_folder, tif_name)

              if not os.path.exists(tif_path):
                  raise FileNotFoundError(f"Falta el archivo: {tif_name}")

              with rasterio.open(tif_path) as src:
                  # Recortar el ráster. out_band_clip tiene forma (1, H, W)
                  out_band_clip, out_transform = mask(src, geometries, crop=True)
                  
                  # Agregamos la banda recortada a la lista
                  all_bands_clipped.append(out_band_clip)

          # CRÍTICO: Apilar todas las bandas en una sola matriz
          # Usamos np.concatenate con axis=0 para apilar las 3 matrices (1, H, W) -> (3, H, W)
          stacked_image = np.concatenate(all_bands_clipped, axis=0)
              
          # 6. REORDENAMIENTO Y RESIZE
          # Reorganizar array: (Bandas, Alto, Ancho) -> (Alto, Ancho, Bandas)
          # Necesario para cv2.resize y TensorFlow/PyTorch
          out_image_reorder = np.transpose(stacked_image, (1, 2, 0))

          # Redimensionar la imagen apilada a TARGET_SIZE (ej: 128x128x3)
          resized_image = cv2.resize(
              out_image_reorder, 
              TARGET_SIZE, 
              interpolation=cv2.INTER_LINEAR
          )
          
          # Opcional: Asegurar que el array sea tipo float32 para el entrenamiento
          resized_image = resized_image.astype(np.float32)

          # print(f"Shape final apilado: {resized_image.shape}") # Debug
          
          X_images.append(resized_image)
          y_labels.append(etiqueta)

      except FileNotFoundError as e:
          print(f"Advertencia: {e}")
          continue
      except Exception as e:
          print(f"Error procesando {tif_path} (ID: {obs_unit_id_num}): {e}")
          continue

      # 5. RECORTE (CLIPPING), RESIZE Y EXTRACCIÓN
      # un archivo
      # try:
      #     with rasterio.open(tif_path) as src:
      #         # Recortar el ráster
      #         out_image, out_transform = mask(src, geometries, crop=True)
              
      #         # Reorganizar array: Rasterio usa (Bandas, Alto, Ancho). CV2 usa (Alto, Ancho, Bandas).
      #         # Para Resizing con CV2 (y la mayoría de CNNs), necesitamos (Alto, Ancho, Bandas)
      #         out_image_reorder = np.transpose(out_image, (1, 2, 0))

      #         # **CRUCIAL: RESIZE** para que todas las imágenes tengan el mismo tamaño
      #         resized_image = cv2.resize(out_image_reorder, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
              
      #         # NORMALIZACIÓN (Si el TIF es de reflectancia [0-1], puede que no sea necesaria
      #         # si el TIF ya está en el rango correcto. Si es uint16, normaliza a 0-1)
      #         # normalized_image = resized_image / resized_image.max() 
      #         # Si el resizing resultó en una imagen 2D (H, W) por ser mono-banda,
      #         # necesitamos reintroducir la dimensión de banda: (H, W) -> (H, W, 1)
      #         if resized_image.ndim == 2:
      #             resized_image = np.expand_dims(resized_image, axis=-1)
              
      #         X_images.append(resized_image)
      #         y_labels.append(etiqueta)

      except rasterio.RasterioIOError:
          print(f"Advertencia: Archivo TIF no encontrado o dañado para {tif_path}")
          continue
      except Exception as e:
          print(f"Error procesando {tif_path} (ID: {obs_unit_id_num}): {e}")
          continue

  print("Extracción de imágenes completada.")

  # Convertir las etiquetas de texto a números (ej: Plaga=0, Sana=1, Indeterminado=2)
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  y_encoded = le.fit_transform(y_labels)

  # --- 6. DIVISIÓN PARA ENTRENAMIENTO ---
  if len(X_images) == 0:
      print("Error: El conjunto de datos de imágenes está vacío después de la extracción.")
  else:
      # Convertir a array de NumPy
      X_images_array = np.array(X_images)
      # y_labels_array = np.array(y_labels) # codificado a 0, 1, 2
      y_labels_array = np.array(y_encoded) # codificado

      # División final
      X_train, X_test, y_train, y_test = train_test_split(
          X_images_array, 
          y_labels_array, 
          test_size=0.2, 
          random_state=42, 
          stratify=y_labels_array
      )

      print(f"\nDatos listos para entrenamiento. Total de muestras: {len(X_images)}")
      print(f"\nResumen de Datos Multiespectrales:")
      print(f"Total de parches extraídos: {len(X_images)}")
      print(f"X Train/Val Split: {len(X_train)} / {len(X_test)}")
      print(f"Y Train/Val Split: {len(y_train)} / {len(y_test)}")
      print(f"Forma X_train: {X_train.shape}")
      print(f"Forma Y_train: {y_train.shape}")
      
      return X_train, X_test, y_train, y_test, le
  
class MultiespectralDataGenerator(Sequence):
    """Generador de datos para cargar imágenes multiespectrales (parches de parcela) 
    a partir de un CSV de etiquetas y un Shapefile de parcelas."""

    def __init__(self, df, parcels_gdf, base_dir_raster, target_size, batch_size, le: LabelEncoder, shuffle=True):
        self.df = df.reset_index(drop=True)  # DataFrame de etiquetas (ya filtrado por Plaga/Sana)
        self.parcels_gdf = parcels_gdf.set_index(SHP_ID_COLUMN) # GeoDataFrame, indexado por el ID numérico
        self.base_dir_raster = base_dir_raster
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.le = le # LabelEncoder ya entrenado (con 'Plaga', 'Sana', etc.)
        
        # Codificar las etiquetas del DataFrame que usará este generador
        self.y_encoded = self.le.transform(self.df['Etiqueta_FINAL'])
        
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # Número de lotes por época
        return ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self):
        # Función llamada al final de cada época (opcional, pero buena práctica)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _get_image_data(self, row):
        """Implementa la lógica de recorte y apilamiento para una sola fila (muestra)."""
        fecha = row['Fecha']
        obs_unit_id_num = row['obsUnitId_num'] 
        
        # 1. BÚSQUEDA DE POLÍGONO (Match por el índice)
        try:
            # Obtener geometría directamente por el índice (SHP_MATCH_ID)
            parcela = self.parcels_gdf.loc[[obs_unit_id_num]]
        except KeyError:
            # Si el ID no existe en el índice del GeoDataFrame
            raise FileNotFoundError(f"Polígono no encontrado en SHP para ID: {obs_unit_id_num}")

        geometries = parcela.geometry.values
        all_bands_clipped = []
        tif_date_prefix = fecha.replace('-', '')

        # 2. APILAMIENTO DE BANDAS (Red, Red Edge, NIR)
        for suffix in BAND_SUFFIXES:
            tif_name = f"{tif_date_prefix}_{suffix}"
            tif_path = os.path.join(self.base_dir_raster, fecha, tif_name)

            if fecha == "2023-05-18":
              tif_name = tif_date_prefix + '_WUR_' + 'transparent_reflectance_' + suffix
              tif_path = os.path.join(self.base_dir_raster, tif_name)
            else:
              tif_name = tif_date_prefix + '_transparent_reflectance_' + suffix
              tif_path = os.path.join(self.base_dir_raster, tif_name)

            if not os.path.exists(tif_path):
                raise FileNotFoundError(f"Falta el archivo: {tif_name}")

            with rasterio.open(tif_path) as src:
                out_band_clip, _ = mask(src, geometries, crop=True)
                all_bands_clipped.append(out_band_clip)

        # 3. APILAMIENTO, REORDENAMIENTO Y RESIZE
        stacked_image = np.concatenate(all_bands_clipped, axis=0) # (3, H, W)
        out_image_reorder = np.transpose(stacked_image, (1, 2, 0)) # (H, W, 3)

        resized_image = cv2.resize(
            out_image_reorder, 
            self.target_size, 
            interpolation=cv2.INTER_NEAREST # Mantenemos nearest para valores de reflectancia
        )
        
        # Convertir a float32 para la CNN
        return resized_image.astype(np.float32)

    def __getitem__(self, index):
        """Genera un lote de datos."""
        # Obtener los índices de las muestras para este lote
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        
        # Inicializar arrays para el lote (X, y)
        # N: Batch Size, H: Altura, W: Ancho, C: Canales (3)
        X = np.empty((len(batch_indices), self.target_size[0], self.target_size[1], len(BAND_SUFFIXES)), dtype=np.float32)
        y = np.empty((len(batch_indices),), dtype=np.int32) 

        # Llenar el lote
        for i, data_index in enumerate(batch_indices):
            row = self.df.iloc[data_index]
            
            try:
                # Obtener y preprocesar la imagen
                image = self._get_image_data(row)
                X[i,] = image
                
                # Obtener la etiqueta codificada
                y[i] = self.y_encoded[data_index]
            except Exception as e:
                # En caso de error (archivo TIF faltante, geometría inválida), 
                # se puede optar por ignorar la muestra o llenar con ceros y manejarlo en la loss.
                # Aquí, simplemente llenamos con un array de ceros y una etiqueta inválida (-1) 
                # para evitar errores de forma. En un entorno de producción, es mejor
                # filtrar estos errores antes de crear el DataFrame.
                print(f"Error fatal al cargar muestra: {row['obsUnitId']} - {e}")
                # Usaremos un continue para intentar recuperar la siguiente muestra si la forma lo permite
                # Para Keras, la forma debe ser consistente, por lo que una muestra fallida
                # probablemente deba hacer fallar el lote si no se puede reemplazar.
                # Por simplicidad, en este ejemplo, aseguramos que la muestra esté vacía.
                X[i,] = np.zeros((self.target_size[0], self.target_size[1], len(BAND_SUFFIXES)))
                y[i] = -1 # Etiqueta inválida
                continue

        # Filtrar posibles muestras inválidas (-1) si se usó la lógica de error
        valid_indices = y != -1
        return X[valid_indices], y[valid_indices]

# ----------------------------------------------------------------------------------------------------

def test_extract_data_to_img_for_train(X_train, X_test, y_train, y_test, le):
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
                metrics=['accuracy'
                    # BinaryAccuracy(name='accuracy'),
                    # Precision(name='precision'),
                    # Recall(name='recall')
                    ])

  print("\nModelo listo para entrenar con Generadores.")

  # # ENTRENAMIENTO
  callbacks, _ = create_cnn_callbacks(os.path.dirname(__file__))
  history = model.fit(
      X_train, 
      y_train,
      epochs=10, # Ajusta el número de épocas
      validation_data=[X_test, y_test],
      callbacks=callbacks
  )

  y_pred = model.predict(X_test)   
  target_names = le.fit_transform(le.classes_)
  print(f'le.classes_: {le.classes_}')
  print(f'target_names: {target_names}')
  print("\n--- Reporte de Clasificación ---")
  # Usamos inverse_transform para mostrar las etiquetas reales en el reporte

  # print(classification_report(y_test, y_pred, target_names=np.array(target_names)))
  # print(f"Precisión General (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

  suffix = f"_MS"
  save_history_and_plot(history, os.path.dirname(__file__), 10, suffix=suffix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
    parser.add_argument("--test", "-t", action="store_true", default=False, help="Indica si quiere demostrar el entrenamiento con un modelo simple")

    # Ejecutar la carga para prueba
    # X = data/img
    # Y = labels/etiquetes
    X_train, X_test, y_train, y_test, le = extract_data_to_img_for_train()

    # Aquí puedes dividir los datos en train/test y entrenar tu modelo
    if parser.parse_args().test:
      test_extract_data_to_img_for_train(X_train, X_test, y_train, y_test, le)
    
    # Asegúrate de que Y_labels se conviertan a one-hot encoding para el entrenamiento CNN:
    # from tensorflow.keras.utils import to_categorical
    # Y_one_hot = to_categorical(Y_labels, num_classes=len(CLASSES))
