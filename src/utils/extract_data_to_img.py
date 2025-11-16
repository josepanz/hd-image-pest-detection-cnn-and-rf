import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2

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
      print(f"X_train shape: {X_train.shape}")
