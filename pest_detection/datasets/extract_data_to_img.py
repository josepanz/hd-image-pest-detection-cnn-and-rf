import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import cv2

from sklearn.preprocessing import LabelEncoder

from pest_detection.print_utils import print_time_and_step

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# --- CONFIGURACIÓN DE RUTAS ---
# 1. Usar r-strings para evitar errores de barras invertidas en Windows.
LABELS_CSV = r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\measurements\generated_labels_unified.csv'
PARCELS_SHP = r'C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\metadata\plot_shapefile.shp'
BASE_DIR_RASTER = r'data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data'
BAND_SUFFIXES = ['red.tif', 'red edge.tif', 'nir.tif', 'RGB.tif', 'blue.tif', 'green.tif']

# 2. DEFINICIÓN DE LA COLUMNA DE UNIÓN DEL SHAPEFILE
# ¡AJUSTA ESTO! Debe ser el nombre exacto de la columna en tu SHP que tiene el número de parcela.
SHP_ID_COLUMN = 'PlotID' # <--- EJEMPLO: Revisa y ajusta este nombre.

# 3. DIMENSIÓN DE SALIDA PARA EL MODELO CNN (Necesario para el entrenamiento)
TARGET_SIZE = (224, 224)

def extract_data_to_img_for_train(data_dir: str = BASE_DIR_RASTER, labels_dir: str = LABELS_CSV, parcels_dir: str = PARCELS_SHP, isRgb: bool = False, img_size: tuple[int, int] = TARGET_SIZE, val_split: float = 0.2, seed: int = 42, batch_size: int = 32, model_type: str = 'cnn'):
  """Construye el dataset de entrenamiento/validación a partir del esquema TTADDA:
  un CSV de labels (columna 'Etiqueta_FINAL' filtrada a 'Plaga'/'Sana', el resto se
  descarta silenciosamente) + un shapefile de parcelas + carpetas de TIFF por fecha.

  Para cada fila del CSV: ubica los TIFF de esa fecha/parcela (RGB.tif si isRgb,
  si no las 5 bandas red/red edge/nir/blue/green), los recorta al polígono de la
  parcela (rasterio.mask), los apila y redimensiona a img_size.

  Normalización (importante, ver también la nota de cnn_model.py y el bug
  documentado en evaluation/inference_utils.py): RGB se normaliza /255; multiespectral
  se normaliza dividiendo por el máximo de esa imagen en particular (no un valor fijo
  global), por lo que la escala relativa entre imágenes distintas no es idéntica.

  model_type='cnn' devuelve imágenes 4D (N, H, W, C) listas para una CNN; 'rf' aplana
  a vectores 2D (N, H*W*C) - pero en la práctica train.py::run_rf_training NO usa esta
  rama: extrae features de una CNN en vez de vectorizar los píxeles crudos, así que
  model_type='rf' solo se ejercita si se llama a esta función directamente con ese valor.

  Retorna (X_train, X_val, y_train, y_val, label_encoder).
  """
  # --- 1. CARGA DE DATOS ---
  print_time_and_step('1', f'Carga de datos {model_type} {"RGB" if isRgb else "Multiespectral"}', timestamp=timestamp, start_time=start_time)
  labels_df = pd.read_csv(labels_dir)
  parcels_gdf = gpd.read_file(parcels_dir)

  # Preparación del Shapefile para la unión: convertir la columna del SHP a string
  parcels_gdf['SHP_MATCH_ID'] = parcels_gdf[SHP_ID_COLUMN].astype(str)
  # Preparación del CSV: aseguramos que el campo recién agregado también sea string
  labels_df['obs_unit_id_num'] = labels_df['obs_unit_id_num'].astype(str)


  # --- 2. FILTRADO Y EXTRACCIÓN ---
  print_time_and_step('2', "Filtrado y extracción de datos", timestamp=timestamp, start_time=start_time)
  df_train = labels_df[labels_df['Etiqueta_FINAL'].isin(['Plaga', 'Sana'])].copy()

  X_images = []
  y_labels = []

  print_time_and_step('2.1', f"Iniciando procesamiento de {len(df_train)} imágenes etiquetadas...", timestamp=timestamp, start_time=start_time)
  # --- A. SELECCIÓN DE ARCHIVOS Y RECORTE ---
  selected_suffixes = ['RGB.tif'] if isRgb else BAND_SUFFIXES # Asumo que BAND_SUFFIXES contiene Red, Red Edge, NIR, Blue, Green, etc.
  # Filtramos los sufijos si es RGB para que solo procese 'RGB.tif'
  suffixes_to_process = [s for s in selected_suffixes if (isRgb and s == 'RGB.tif') or (not isRgb and s != 'RGB.tif')]
  print_time_and_step('2.2', f'Archivos seleccionados: {suffixes_to_process}', timestamp=timestamp, start_time=start_time)

  for index, row in df_train.iterrows():
      fecha = row['Fecha']
      # Usamos directamente la nueva columna del CSV
      obs_unit_id_num = row['obs_unit_id_num']
      etiqueta = row['Etiqueta_FINAL']

      # 3. CONSTRUCCIÓN DE RUTA TIF
      tif_folder = os.path.join(data_dir, fecha)

      # 4. BÚSQUEDA DE POLÍGONO (Unión directa por ID numérico)
      parcela = parcels_gdf[parcels_gdf['SHP_MATCH_ID'] == obs_unit_id_num]

      if parcela.empty:
          print_time_and_step('error', f"Polígono no encontrado en SHP para ID numérico: {obs_unit_id_num}", timestamp=timestamp, start_time=start_time)
          continue

      geometries = parcela.geometry.values

      all_bands_clipped = [] # Lista temporal para guardar los recortes de cada banda

      # 5. EXTRACCIÓN Y APILAMIENTO DE TRES BANDAS (Red, Red Edge, NIR)
      try:
          tif_date_prefix = fecha.replace('-', '') # Ej: '20230605'

          for suffix in suffixes_to_process:

              if suffix == 'RGB.tif':
                # Construir la ruta al archivo de la banda específica
                tif_name = f"{tif_date_prefix}_{suffix}" # Ej: 20230605_red.tif
                tif_path = os.path.join(data_dir, fecha, tif_name)
              else:
                if fecha == "2023-05-18":
                  tif_name = tif_date_prefix + '_WUR_' + 'transparent_reflectance_' + suffix
                  tif_path = os.path.join(tif_folder, tif_name)
                else:
                  tif_name = tif_date_prefix + '_transparent_reflectance_' + suffix
                  tif_path = os.path.join(tif_folder, tif_name)

              if not os.path.exists(tif_path):
                  # Algunas carpetas de fecha traen los TIFF multiespectrales con un
                  # prefijo de fecha distinto al de la carpeta/RGB/DEM (dataset real:
                  # 2023-06-05/20230606_transparent_reflectance_*.tif). Antes de
                  # descartar la fila, buscamos por sufijo cualquier archivo de esa
                  # misma carpeta que calce, sin importar el prefijo de fecha.
                  candidatos = glob.glob(os.path.join(tif_folder, f"*transparent_reflectance_{suffix}"))
                  if candidatos:
                      tif_path = candidatos[0]
                  else:
                      raise FileNotFoundError(f"Falta el archivo: {tif_name}")

              with rasterio.open(tif_path) as src:
                  # Recortar el ráster. out_band_clip tiene forma (1, H, W)
                  out_band_clip, out_transform = mask(src, geometries, crop=True)

                  # Agregamos la banda recortada a la lista
                  all_bands_clipped.append(out_band_clip)

          if isRgb == False:
            # CRÍTICO: Apilar todas las bandas en una sola matriz
            # Usamos np.concatenate con axis=0 para apilar las N matrices (1, H, W) -> (N, H, W)
            stacked_image = np.concatenate(all_bands_clipped, axis=0)
          else:
            stacked_image = all_bands_clipped[0]
            if stacked_image.shape[0] > 3:
              # Algunos RGB.tif traen un canal alpha además de R/G/B (dataset real:
              # 2023-05-18/20230518_RGB.tif tiene 4 bandas) - nos quedamos solo con R/G/B.
              stacked_image = stacked_image[:3]

          # 6. REORDENAMIENTO Y RESIZE
          # Reorganizar array: (Bandas, Alto, Ancho) -> (Alto, Ancho, Bandas)
          # Necesario para cv2.resize y TensorFlow/PyTorch
          out_image_reorder = np.transpose(stacked_image, (1, 2, 0))

          # Redimensionar la imagen apilada a img_size (ej: 224x224xN)
          resized_image = cv2.resize(
              out_image_reorder,
              img_size,
              interpolation=cv2.INTER_LINEAR
          )

          # Normalización
          if not isRgb:
            max_val = np.max(resized_image)
            if max_val > 0:
              resized_image = resized_image / max_val
          else:
             resized_image = resized_image.astype(np.float32) / 255.0

          # Opcional: Asegurar que el array sea tipo float32 para el entrenamiento
          resized_image = resized_image.astype(np.float32)

          # *** VERIFICACIÓN CRÍTICA DE CANALES ***
          expected_channels = 3 if isRgb else len(suffixes_to_process)

          if resized_image.shape[-1] != expected_channels:
              print(f"Imagen inválida: {resized_image.shape[-1]} canales (esperado {expected_channels})")
              continue

          X_images.append(resized_image)
          y_labels.append(etiqueta)

      except FileNotFoundError as e:
          print_time_and_step('error', f"Advertencia: {e}", timestamp=timestamp, start_time=start_time)
          continue
      except Exception as e:
          print_time_and_step('Error', f"Error procesando {tif_path} (ID: {obs_unit_id_num}): {e}", timestamp=timestamp, start_time=start_time)
          continue

  print_time_and_step('7', "Extracción de imágenes completada.")

  # Convertir las etiquetas de texto a números (ej: Plaga=0, Sana=1, Indeterminado=2)
  le = LabelEncoder()
  y_encoded = le.fit_transform(y_labels)

  # --- 6. DIVISIÓN PARA ENTRENAMIENTO ---
  if len(X_images) == 0:
    raise ValueError("Dataset vacío: ninguna imagen válida fue procesada.")
  else:
      # Convertir a array de NumPy
      X_images_array = np.array(X_images)
      y_labels_array = np.array(y_encoded) # codificado

      if model_type=='cnn':
        # División final
        X_train, X_test, y_train, y_test = train_test_split(
            X_images_array,
            y_labels_array,
            test_size=val_split,
            random_state=seed,
            stratify=y_labels_array
        )
      else:
        # Obtenemos la forma requerida F (features totales)
        num_samples = X_images_array.shape[0]
        total_features = X_images_array.size // num_samples

        # Aplanar las dimensiones espaciales y de canales en una sola dimensión de características (F)
        X_vectorized = X_images_array.reshape(num_samples, total_features)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, # <--- Usar el array vectorizado
            y_labels_array,
            test_size=val_split,
            random_state=seed,
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

if __name__ == '__main__':
    # Smoke test manual: ejecuta la extracción con las rutas por defecto (dataset TTADDA_NARO_2023_F1)
    # y solo imprime el resumen de shapes; no entrena ningún modelo.
    extract_data_to_img_for_train()
