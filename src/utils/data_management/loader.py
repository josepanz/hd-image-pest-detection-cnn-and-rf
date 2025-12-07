import os
import numpy as np
import rasterio
from shapely.geometry import shape
from typing import Tuple
from skimage.transform import resize

def load_single_multispectral_image(
    tiff_dir: str, 
    img_size: Tuple[int, int] = (224, 224),
    modified: bool = False
) -> np.ndarray:
    """Carga, apila y redimensiona las 5 bandas TIFF de una carpeta de imagen única."""
    bands = {
        #'blue': f'transparent_reflectance_blue{"_modified" if modified else ""}.tif',
        #'green': f'transparent_reflectance_green{"_modified" if modified else ""}.tif',
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
        print('full_filename: ', full_filename)
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

if __name__ == '__main__':
    pass
    