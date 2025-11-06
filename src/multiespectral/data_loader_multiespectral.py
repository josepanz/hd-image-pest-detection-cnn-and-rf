import os
import numpy as np
import rasterio
import fiona
from shapely.geometry import shape, box
from typing import List, Tuple, Dict
from skimage.transform import resize
from keras.utils import Sequence

# --- CONFIGURACIÓN DE RUTAS Y CLASES ---

# CLASES: 0 = Plaga, 1 = Sana
CLASSES = ["Plaga", "Sana"] 
BASE_DATA_DIR = os.path.join("data")

# --- FUNCIONES DE UTILIDAD PARA GEOPROCESAMIENTO ---

def load_bands_and_mask(tiff_folder: str) -> Tuple[np.ndarray, rasterio.io.DatasetReader]:
    """
    Carga las 5 bandas TIFF separadas, las apila y devuelve el stack y el metadato (source).
    """
    bands = {
        'blue': 'transparent_reflectance_blue_modified.tif',
        'green': 'transparent_reflectance_green_modified.tif',
        'red': 'transparent_reflectance_red_modified.tif',
        'red_edge': 'transparent_reflectance_red edge_modified.tif',
        'nir': 'transparent_reflectance_nir_modified.tif',
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
            # Leer el primer (y único) canal de cada TIF
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
    stacked_image = stacked_image.astype(np.float32) / 10000.0 # Ajusta si tu reflectancia está en otro rango
    
    return stacked_image, source


def extract_labeled_patches(
    stacked_image: np.ndarray, 
    source: rasterio.io.DatasetReader,
    shapefile_path: str,
    patch_size: int = 224, 
    class_label: int = 1 # 1 para 'Sana', 0 para 'Plaga'
) -> List[Tuple[np.ndarray, int]]:
    """
    Usa el shapefile para extraer parches etiquetados de la imagen multiespectral.
    """
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
                resized_patch = resize(patch, (patch_size, patch_size), anti_aliasing=True)
                patches.append((resized_patch, class_label))
                
    return patches


def load_multiespectral_data(patch_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga, apila y extrae los datos multiespectrales de Plaga y Sana.
    """
    all_patches = []
    
    # 1. Definir rutas a los archivos y etiquetas
    ms_images_dir = os.path.join(BASE_DATA_DIR, "multispectral_images")
    sf_dir = os.path.join(BASE_DATA_DIR, "shapefiles")

    # Los nombres de las carpetas de fecha/campo para iterar
    tiff_folders = os.listdir(ms_images_dir)

    # Definir los Shapefiles para Plaga y Sana (Asumiendo que 'konv' es Plaga y 'eko' es Sana/Control, o viceversa)
    # NECESITAS CONFIRMAR QUÉ SHAPEFILE CORRESPONDE A QUÉ CLASE
    SHAPEFILES_MAP = {
        0: os.path.join(sf_dir, "potato_locations_konv.shp"),  # Ejemplo: 'konv' -> Plaga (Clase 0)
        1: os.path.join(sf_dir, "potato_locations_eko.shp"),   # Ejemplo: 'eko' -> Sana (Clase 1)
    }

    # 2. Iterar sobre las carpetas de fecha/campo
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
                    
                    # Llamar a la función de extracción
                    patches = extract_labeled_patches(
                        stacked_image, 
                        source, 
                        shapefile_path, 
                        patch_size=patch_size, 
                        class_label=class_label
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
    
    print(f"\nResumen de Datos Multiespectrales:")
    print(f"Total de parches extraídos: {len(X)}")
    print(f"Forma de entrada X (Parches, Alto, Ancho, Canales): {X.shape}")
    print(f"Forma de etiquetas Y: {Y.shape}")
    
    return X, Y

def load_single_multispectral_image(
    tiff_dir: str, 
    img_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Carga, apila y redimensiona las 5 bandas TIFF de una carpeta de imagen única.
    """
    bands = {
        'blue': 'transparent_reflectance_blue_modified.tif',
        'green': 'transparent_reflectance_green_modified.tif',
        'red': 'transparent_reflectance_red_modified.tif',
        'red_edge': 'transparent_reflectance_red edge_modified.tif',
        'nir': 'transparent_reflectance_nir_modified.tif',
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
    
    # Retorna la imagen lista para la predicción (H, W, 5)
    return resized_image

# --- GENERADOR DE DATOS PARA ENTRENAMIENTO (OPCIONAL, PERO RECOMENDADO PARA GRANDES DATASETS) ---

class MultiespectralDataGenerator(Sequence):
    # Puedes implementar esta clase para generar los parches en tiempo real, 
    # pero para el primer prototipo, usar load_multiespectral_data() es más simple.
    pass


if __name__ == '__main__':
    # Ejecutar la carga para prueba
    X_data, Y_labels = load_multiespectral_data(patch_size=224)
    # Aquí puedes dividir los datos en train/test y entrenar tu modelo
    
    # Asegúrate de que Y_labels se conviertan a one-hot encoding para el entrenamiento CNN:
    # from tensorflow.keras.utils import to_categorical
    # Y_one_hot = to_categorical(Y_labels, num_classes=len(CLASSES))