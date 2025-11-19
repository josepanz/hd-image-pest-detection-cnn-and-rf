# src/data_management/loader_cnn_ms.py

import os
import numpy as np
import rasterio
import fiona
from shapely.geometry import shape
from typing import List, Tuple, Dict
from skimage.transform import resize
# Nota: Keras Sequence (data generator) se omite por simplicidad inicial
from keras.utils import Sequence
import cv2

from rasterio.mask import mask
from math import ceil
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURACIÓN Y UTILIDADES (Mantener) ---
# Necesitas una función para load_bands_and_mask, extract_labeled_patches, y load_single_multispectral_image
# Estas funciones provienen de tu data_loader_multiespectral.py y se asume que están aquí.

# [NOTA: INSERTAR AQUÍ las funciones: load_bands_and_mask, extract_labeled_patches, load_single_multispectral_image de tu archivo original]
# ... [Código de load_bands_and_mask] ...
# ... [Código de extract_labeled_patches] ...
# ... [Código de load_single_multispectral_image] ...
CLASSES = ["Plaga", "Sana"] 
BASE_DATA_DIR = os.path.join("data")

# --- FUNCIONES DE UTILIDAD PARA GEOPROCESAMIENTO ---
# (Se asume que estas funciones están definidas aquí arriba)

def load_bands_and_mask(tiff_folder: str, modified: bool = False) -> Tuple[np.ndarray, rasterio.io.DatasetReader]:
    """Carga y apila 5 bandas TIFF con normalización."""
    # ... (cuerpo de la función original) ...
    # Porque puede usar 
    #Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_blue_modified
    #20230518_WUR_transparent_reflectance_blue.tif
    bands = {
        'blue': f"transparent_reflectance_blue{'_modified' if modified else ''}.tif",
        'green': f"transparent_reflectance_green{'_modified' if modified else ''}.tif",
        'red': f"transparent_reflectance_red{'_modified' if modified else ''}.tif",
        'red_edge': f'transparent_reflectance_red edge{'_modified' if modified else ''}.tif',
        'nir': f'transparent_reflectance_nir{'_modified' if modified else ''}.tif',
        #'rgb': 'rgb.tif',
        #'dem': 'dem.tif',
    }
    
    # Lista para almacenar las matrices de las bandas
    band_data_list = []
    source = None
    
    # 1. Cargar las 5 bandas y asegurarse de que sean consistentes
    for band_name, filename_suffix in bands.items():
        # Busca el archivo TIF que contenga el sufijo
        full_filename = next((f for f in os.listdir(tiff_folder) if f.lower().endswith(filename_suffix)), None)
        
        if not full_filename:
            print(f"Error: No se encontró el archivo para la banda {band_name} en {tiff_folder}")
            continue

        band_path = os.path.join(tiff_folder, full_filename)
        with rasterio.open(band_path) as src:
            # Leer el primer (y único) canal de cada TIF, solo RGB.tif tiene 3
            band_data = src.read(1)
            band_data_list.append(band_data)
            
            # Guardar el metadato (CRS, transform) del primer archivo
            if source is None:
                source = src
                
    if not band_data_list:
        raise FileNotFoundError(f"No se pudieron cargar bandas de {tiff_folder}")
        
    # 2. Apilar las bandas para obtener una matriz (H, W, 5)
    stacked_image = np.stack(band_data_list, axis=-1)

    # 3. Normalización (Opcional, crucial para Deep Learning)
    # Convertir a float y normalizar al rango [0, 1] (asumiendo datos de reflectancia)
    stacked_image = stacked_image.astype(np.float32) / 10000.0
    
    return stacked_image, source

def extract_labeled_patches(
    stacked_image: np.ndarray, 
    source: rasterio.io.DatasetReader,
    shapefile_path: str,
    patch_size: int = 224, 
    class_label: int = 1 # 1 para 'Sana', 0 para 'Plaga'
) -> List[Tuple[np.ndarray, int]]:
    """Usa el shapefile para extraer parches etiquetados de la imagen multiespectral."""
    patches = []
    
    with fiona.open(shapefile_path) as shp:
        for feature in shp:
            geom = shape(feature['geometry'])
            
            # Asumimos que la etiqueta (Sana/Plaga) se define por el shapefile usado
            # Encontrar el centroide de la geometría (patch)
            centroid_x, centroid_y = geom.centroid.x, geom.centroid.y

            # Convertir coordenadas geográficas (X, Y) a coordenadas de píxeles (row, col)
            row, col = source.index(centroid_x, centroid_y)

            # Definir los límites del parche en coordenadas de píxeles
            half_size = patch_size // 2
            
            row_start = max(0, row - half_size)
            row_end = min(stacked_image.shape[0], row + half_size)
            col_start = max(0, col - half_size)
            col_end = min(stacked_image.shape[1], col + half_size)
            
            # Extraer el parche
            patch = stacked_image[row_start:row_end, col_start:col_end, :]

            # Asegurar que el parche tenga el tamaño correcto (manejo de bordes)
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append((patch, class_label))
            # Si el parche es más pequeño (borde), podemos reescalarlo o descartarlo
            elif patch.shape[0] > 0 and patch.shape[1] > 0:
                # Opción: Redimensionar (más sencillo)
                resized_patch = resize(patch, (patch_size, patch_size, stacked_image.shape[2]), anti_aliasing=True)
                patches.append((resized_patch, class_label))
                
    return patches

def load_single_multispectral_image(
    tiff_dir: str, 
    img_size: Tuple[int, int] = (224, 224),
    modified: bool = False
) -> np.ndarray:
    """Carga, apila y redimensiona las 5 bandas TIFF de una carpeta de imagen única."""
    bands = {
        'blue': f'transparent_reflectance_blue{"_modified" if modified else ""}.tif',
        'green': f'transparent_reflectance_green{"_modified" if modified else ""}.tif',
        'red': f'transparent_reflectance_red{"_modified" if modified else ""}.tif',
        'red_edge': f'transparent_reflectance_red edge{"_modified" if modified else ""}.tif',
        'nir': f'transparent_reflectance_nir{"_modified" if modified else ""}.tif',
        #'rgb': 'rgb.tif',
        #'dem': 'dem
    }
    band_data_list = []
    
    # Cargar y apilar las 5 bandas
    for band_name, filename_suffix in bands.items():
        # Encuentra el archivo TIF que contiene el sufijo en el directorio
        full_filename = next((f for f in os.listdir(tiff_dir) if f.lower().endswith(filename_suffix)), None)
        if not full_filename:
            raise FileNotFoundError(f"Falta el archivo para la banda {band_name} en {tiff_dir}")

        band_path = os.path.join(tiff_dir, full_filename)
        with rasterio.open(band_path) as src:
            band_data = src.read(1)
            band_data_list.append(band_data)

    stacked_image = np.stack(band_data_list, axis=-1)
    
    # Normalización (igual que en el entrenamiento)
    stacked_image = stacked_image.astype(np.float32) / 10000.0
    
    # Redimensionar al tamaño del modelo (ej. 224x224)
    # skimage.transform.resize usa anti_aliasing por defecto
    resized_image = resize(stacked_image, (*img_size, 5), anti_aliasing=True)
    
    return resized_image


# --- FUNCIÓN PRINCIPAL DE CARGA DE DATOS ---

def load_multiespectral_data(
    patch_size: int = 224, 
    test_split: float = 0.2, 
    seed: int = 123,
    dataset: str = "TTADDA" # "EKOKOVNV"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga, apila y extrae los datos multiespectrales de Plaga y Sana, y realiza el split.
    
    Returns: (X_train, X_val, Y_train, Y_val)
    """
    from sklearn.model_selection import train_test_split # Se importa aquí para mantener el archivo limpio
    
    all_patches = []
    
    # 1. Definir rutas a los archivos y etiquetas
    # ms_images_dir = os.path.join(BASE_DATA_DIR, r"multispectral_images")
    # sf_dir = os.path.join(BASE_DATA_DIR, r"shapefiles")
    ms_images_dir = os.path.join(BASE_DATA_DIR, r"C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\drone_data")
    sf_dir = os.path.join(BASE_DATA_DIR, r"C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\metadata")

    # Los nombres de las carpetas de fecha/campo para iterar
    tiff_folders = os.listdir(ms_images_dir)

    # Definir los Shapefiles para Plaga y Sana (Asumiendo que 'konv' es Plaga y 'eko' es Sana/Control, o viceversa)
    # NECESITAS CONFIRMAR QUÉ SHAPEFILE CORRESPONDE A QUÉ CLASE
    SHAPEFILES_MAP = {
        # 0: os.path.join(sf_dir, "potato_locations_konv.shp"),  # Plaga (Clase 0)
        # 1: os.path.join(sf_dir, "potato_locations_eko.shp"),   # Sana (Clase 1)
        0: os.path.join(sf_dir, "plot_shapefile.shp"),  # Plaga (Clase 0)
        1: os.path.join(sf_dir, "plot_shapefile.shp"),   # Sana (Clase 1)
    }

    # 2. Extracción de Parches
    for tiff_folder_name in tiff_folders:
        tiff_folder_path = os.path.join(ms_images_dir, tiff_folder_name)
        if not os.path.isdir(tiff_folder_path):
            continue
            
        print(f"-> Procesando imágenes en: {tiff_folder_name}")
        
        try:
            # Cargar la imagen multibanda apilada (H, W, 5)
            stacked_image, source = load_bands_and_mask(tiff_folder_path)
            
            # 3. Extraer parches etiquetados usando los Shapefiles
            for class_label, shapefile_path in SHAPEFILES_MAP.items():
                if os.path.exists(shapefile_path):
                    print(f"   Extrayendo parches para Clase {class_label} ({CLASSES[class_label]})")
                    patches = extract_labeled_patches(
                        stacked_image, source, shapefile_path, patch_size=patch_size, class_label=class_label
                    )
                    all_patches.extend(patches)
                else:
                    print(f"   Advertencia: Shapefile no encontrado en {shapefile_path}")

        except Exception as e:
            print(f"Error al procesar la carpeta {tiff_folder_name}: {e}")
            continue

    if not all_patches:
        raise ValueError("No se extrajeron parches. Revise rutas de archivos y shapefiles.")

    # 4. Final: Separar datos y etiquetas
    X = np.array([patch[0] for patch in all_patches])
    Y = np.array([patch[1] for patch in all_patches])
    
    # 2. Split de Datos (Agregado para unificación)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_split, random_state=seed, stratify=Y
    )
    
    print(f"\nResumen de Datos Multiespectrales:")
    print(f"Total de parches extraídos: {len(X)}")
    print(f"X Train/Val Split: {len(X_train)} / {len(X_val)}")
    print(f"Y Train/Val Split: {len(Y_train)} / {len(Y_val)}")
    print(f"Forma X_train: {X_train.shape}")
    print(f"Forma Y_train: {Y_train.shape}")
    
    return X_train, X_val, Y_train, Y_val


# --- CONFIGURACIÓN DE RUTAS Y CONSTANTES (Deben coincidir con tu script principal) ---
# Define los sufijos de tus archivos para cada banda (Red, Red Edge, NIR)
BAND_SUFFIXES = ['red.tif', 'red_edge.tif', 'nir.tif'] 
class MultiespectralDataGenerator(Sequence):
    """Generador de datos para cargar imágenes multiespectrales (parches de parcela) 
    a partir de un CSV de etiquetas y un Shapefile de parcelas."""

    def __init__(self, df, parcels_gdf, base_dir_raster, target_size, batch_size, le: LabelEncoder, shuffle=True):
        self.df = df.reset_index(drop=True)  # DataFrame de etiquetas (ya filtrado por Plaga/Sana)
        self.parcels_gdf = parcels_gdf.set_index('SHP_MATCH_ID') # GeoDataFrame, indexado por el ID numérico
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

if __name__ == '__main__':
    # Ejecutar la carga para prueba
    # X = data/img
    # Y = labels/etiquetes
    X_train, X_val, Y_train, Y_val = load_multiespectral_data(patch_size=224)
    # Aquí puedes dividir los datos en train/test y entrenar tu modelo
    
    # Asegúrate de que Y_labels se conviertan a one-hot encoding para el entrenamiento CNN:
    # from tensorflow.keras.utils import to_categorical
    # Y_one_hot = to_categorical(Y_labels, num_classes=len(CLASSES))