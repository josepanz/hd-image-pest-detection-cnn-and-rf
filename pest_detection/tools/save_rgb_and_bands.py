# src/multiespectral/save_rgb_and_bands.py

import os
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# --- CONFIGURACIÓN DE RUTAS Y BANDAS ---
# Definimos el mapeo de bandas que están en tus archivos TIFF
BANDS_MAPPING = {
    'blue': 'transparent_reflectance_blue_modified.tif',
    'green': 'transparent_reflectance_green_modified.tif',
    'red': 'transparent_reflectance_red_modified.tif',
    'red_edge': 'transparent_reflectance_red edge_modified.tif',
    'nir': 'transparent_reflectance_nir_modified.tif',
}

# --- FUNCIONES DE UTILIDAD ---

def load_bands(tiff_folder: str) -> Dict[str, np.ndarray]:
    """
    Carga las 5 bandas TIFF en un diccionario.
    """
    band_data: Dict[str, np.ndarray] = {}
    
    for band_name, filename_suffix in BANDS_MAPPING.items():
        # Busca el archivo TIF que contenga el sufijo
        try:
            full_filename = next(
                (f for f in os.listdir(tiff_folder) if f.endswith(filename_suffix)), None
            )
            if not full_filename:
                print(f"❌ Error: Falta el archivo para la banda {band_name} en {tiff_folder}")
                continue
        except FileNotFoundError:
            print(f"❌ Error: El directorio {tiff_folder} no existe.")
            return {}
        
        band_path = os.path.join(tiff_folder, full_filename)
        with rasterio.open(band_path) as src:
            # Leemos la banda, la normalizamos y la almacenamos
            data = src.read(1).astype(np.float32) / 10000.0 # Normalización a 0-1
            band_data[band_name] = data

    return band_data

def normalize_stretch(band: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> np.ndarray:
    """
    Aplica un estiramiento de contraste (percentile clipping) a una banda.
    Esto ajusta los valores dentro del rango [0, 1] para mejorar la visualización.
    """
    # Recorta los valores extremos (outliers)
    low_val = np.percentile(band, lower_percentile)
    high_val = np.percentile(band, upper_percentile)
    
    # Aplica el recorte
    stretched_band = np.clip(band, low_val, high_val)
    
    # Re-normaliza la banda estirada al rango 0-1
    # Evita la división por cero si el rango es 0
    if (high_val - low_val) > 0:
        stretched_band = (stretched_band - low_val) / (high_val - low_val)
    else:
        stretched_band = np.zeros_like(stretched_band) # Si no hay rango, es negro
        
    return stretched_band

def save_rgb_and_individual_bands(tiff_folder: str, output_dir: str):
    """
    Carga las bandas, crea la imagen RGB y guarda el RGB y las bandas individuales como PNG.
    """
    # 1. Cargar datos
    print(f"Cargando bandas desde: {tiff_folder}")
    bands = load_bands(tiff_folder)

    if not bands:
        print("Saliendo. No se pudieron cargar las bandas.")
        return

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Diccionario para almacenar las bandas que ya tienen el stretch aplicado
    stretched_rgb_bands = {}

    # 2. Guardar bandas individuales (PNG)
    print("\nGuardando bandas individuales como PNG...")
    for band_name, data in bands.items():
        data_stretched = normalize_stretch(data, lower_percentile=0.1, upper_percentile=99.9)
        # El nombre del archivo se obtiene del nombre de la carpeta + nombre de la banda
        output_filename = f"{os.path.basename(tiff_folder)}_{band_name}.png"
        save_path = os.path.join(output_dir, output_filename)
        
        # Usamos matplotlib.pyplot para guardar la matriz (data) como imagen.
        # cmap='gray' es adecuado para bandas monocromáticas.
        # Los datos están normalizados (0-1), lo que es ideal para guardar.
        #plt.imsave(save_path, data_stretched, cmap='gray')
        
        # --- NUEVO CÓDIGO PARA GUARDAR CON PSEUDO-COLOR Y COLORBAR ---
        fig, ax = plt.subplots(figsize=(8, 8)) # Creamos una figura
        
        # Usamos 'viridis' u otro colormap que se vea bien
        im = ax.imshow(data_stretched, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Banda: {band_name.upper()}')
        ax.axis('off') # Ocultar ejes
        
        # Añadir la barra de color (colorbar)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Reflectancia Normalizada (0-1)')
        
        # Guardar la figura
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Cerramos la figura para liberar memoria

        print(f"  - {band_name.upper()} guardada en: {output_filename}")

        # Guardamos solo las RGB estiradas para el paso 3
        if band_name in ['red', 'green', 'blue']:
            # Usaremos un stretch ligeramente más agresivo para la imagen RGB final
            stretched_rgb_bands[band_name] = normalize_stretch(data, lower_percentile=5.0, upper_percentile=95.0)

    # 3. Crear y guardar RGB (bandas Rojo, Verde, Azul)
    if 'red' in stretched_rgb_bands and 'green' in stretched_rgb_bands and 'blue' in stretched_rgb_bands:
        # Apilamos las bandas que ya están estiradas y normalizadas a 0-1
        # rgb_image = np.stack(
        #     [stretched_rgb_bands['red'], stretched_rgb_bands['green'], stretched_rgb_bands['blue']], 
        #     axis=-1
        # )

        r_stretched = normalize_stretch(bands['red'],   lower_percentile=0.5, upper_percentile=99.5)
        g_stretched = normalize_stretch(bands['green'], lower_percentile=0.5, upper_percentile=99.5)
        b_stretched = normalize_stretch(bands['blue'],  lower_percentile=0.5, upper_percentile=99.5)
        
        # Apilamos las bandas que ya están estiradas y normalizadas a 0-1
        rgb_image = np.stack(
            [r_stretched, g_stretched, b_stretched], 
            axis=-1
        )

        # Guardar la imagen RGB
        rgb_output_filename = f"{os.path.basename(tiff_folder)}_RGB_stretched.png"
        rgb_save_path = os.path.join(output_dir, rgb_output_filename)
        
        # 🎉 AHORA sí funcionará y se verá bien, ya que todos los valores están en [0, 1]
        plt.imsave(rgb_save_path, rgb_image)
        print(f"\n✅ RGB (RVA) con contraste ajustado guardada en: {rgb_output_filename}")
    else:
        print("❌ Error: Faltan las bandas 'red', 'green' o 'blue' para crear la imagen RGB.")

    # # 3. Crear y guardar RGB (bandas Rojo, Verde, Azul)
    # if 'red' in bands and 'green' in bands and 'blue' in bands:
    #     # Apilamos las bandas en el orden R, G, B
    #     # Las bandas están normalizadas a 0-1 (Float32)
    #     rgb_image = np.stack([bands['red'], bands['green'], bands['blue']], axis=-1)
        
    #     # Guardar la imagen RGB
    #     rgb_output_filename = f"{os.path.basename(tiff_folder)}_RGB.png"
    #     rgb_save_path = os.path.join(output_dir, rgb_output_filename)
        
    #     # imsave es ideal para guardar arrays RGB normalizados
    #     plt.imsave(rgb_save_path, rgb_image)
    #     print(f"\n✅ RGB (RVA) guardada en: {rgb_output_filename}")
    # else:
    #     print("❌ Error: Faltan las bandas 'red', 'green' o 'blue' para crear la imagen RGB.")

# --- FUNCIÓN PRINCIPAL ---

def main():
    parser = argparse.ArgumentParser(
        description="Crea imágenes RGB y guarda bandas individuales en formato PNG desde TIFF multiespectrales."
    )
    # Argumento posicional para el directorio de entrada (donde están los TIFFs)
    parser.add_argument(
        "tiff_dir",
        help="Ruta al directorio que contiene los archivos TIFF de las bandas (ej: data/multispectral_images/2022_06_15__eko_ecobreed)"
    )
    # Argumento opcional para el directorio de salida
    parser.add_argument(
        "-o", "--output_dir",
        default="multispectral_png_output",
        help="Directorio donde se guardarán los resultados PNG."
    )
    
    args = parser.parse_args()
    
    save_rgb_and_individual_bands(args.tiff_dir, args.output_dir)
    print("\nProceso completado.")


if __name__ == "__main__":
    main()