# src/data_management/utils_tiff_converter.py

import os
import rasterio
import numpy as np
from skimage.io import imsave
from typing import Tuple

# Definición de Canales Estándar (Adaptar si el orden de tus TIFF es diferente)
# Asumimos que el TIFF de 5 canales tiene el siguiente orden de bandas (índices Python 0-based):
BLUE_BAND_INDEX = 0  
GREEN_BAND_INDEX = 1
RED_BAND_INDEX = 2
# RED_EDGE_BAND_INDEX = 3
# NIR_BAND_INDEX = 4


def convert_tiff_to_rgb_png(
    tiff_path: str,
    output_dir: str,
    max_reflectance: int = 10000,
    rgb_band_indices: Tuple[int, int, int] = (RED_BAND_INDEX, GREEN_BAND_INDEX, BLUE_BAND_INDEX)
) -> str | None:
    """
    Carga un archivo TIFF multiespectral, extrae los canales RGB, normaliza 
    la reflectancia a 0-255 y lo guarda como un archivo PNG.

    Args:
        tiff_path: Ruta al archivo TIFF de entrada.
        output_dir: Directorio donde se guardará el archivo PNG resultante.
        max_reflectance: Valor máximo de reflectancia (e.g., 10000 para 16-bit).
        rgb_band_indices: Índices de las bandas [R, G, B] en el archivo TIFF.
        
    Returns:
        Ruta del archivo PNG guardado o None si hay un error.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Procesando TIFF: {tiff_path}")
    
    try:
        with rasterio.open(tiff_path) as src:
            # 1. Leer las bandas R, G, B (usando índices 1-based para rasterio)
            # rasterio usa índices 1-based, no 0-based como Python
            print(f'src.count: {src.count}')
            R = src.read(rgb_band_indices[0] + 1).astype(np.float32) 
            G = src.read(rgb_band_indices[1] + 1).astype(np.float32)
            B = src.read(rgb_band_indices[2] + 1).astype(np.float32)
            
            # 2. Apilar las bandas
            rgb_img = np.stack([R, G, B], axis=-1)

            # 3. Normalización (Reflectancia a 0-1)
            # Clip para asegurar que los valores atípicos no afecten la normalización
            rgb_img = np.clip(rgb_img, 0, max_reflectance)
            rgb_normalized = rgb_img / max_reflectance 
            
            # 4. Escalar a 0-255 (formato de imagen estándar, 8-bit)
            # Convertir a UINT8 (necesario para PNG/JPG)
            rgb_8bit = (rgb_normalized * 255).astype(np.uint8)

            # 5. Guardar el archivo PNG
            base_filename = os.path.basename(tiff_path).replace('.tif', '_rgb.png').replace('.tiff', '_rgb.png')
            output_path = os.path.join(output_dir, base_filename)
            
            # Usar imsave de skimage o PIL/OpenCV para guardar el array RGB
            imsave(output_path, rgb_8bit) 

            print(f"✅ TIFF convertido y guardado como PNG en: {output_path}")
            return output_path

    except Exception as e:
        print(f"❌ Error al procesar el archivo {tiff_path}: {e}")
        return None

# --- Ejemplo de Script de Ejecución (Para tu uso) ---

def main():
    # EJEMPLO: Reemplaza estas rutas con tus carpetas reales
    TIFF_ROOT_DIR = 'data/multispectral_images/TTADDA_NARO_2023_F1/drone_data/2023-05-18'
    OUTPUT_ROOT_DIR = 'data/multispectral_images/TTADDA_NARO_2023_F1/drone_data/2023-05-18'

    if not os.path.exists(TIFF_ROOT_DIR):
        print(f"❌ Directorio de TIFFs no encontrado: {TIFF_ROOT_DIR}")
        return

    # Iterar sobre todos los TIFFs en la carpeta y subcarpetas
    for root, _, files in os.walk(TIFF_ROOT_DIR):
        for file in files:
            if file.lower().endswith(('rgb.tif', 'rgb.tiff')):
                tiff_full_path = os.path.join(root, file)
                
                # Crear la estructura de carpetas de salida basada en el archivo de entrada
                relative_path = os.path.relpath(root, TIFF_ROOT_DIR)
                target_output_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path)
                
                convert_tiff_to_rgb_png(
                    tiff_full_path, 
                    target_output_dir, 
                    max_reflectance=10000 
                )

if __name__ == '__main__':
    # Este script DEBE ejecutarse UNA SOLA VEZ antes del entrenamiento RGB
    main()