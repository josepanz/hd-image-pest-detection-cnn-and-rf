import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
import argparse
# Para prevenir errores de división por cero/invalidos en cálculos
np.seterr(divide='ignore', invalid='ignore') 

# Algoritmo basado en https://colab.research.google.com/drive/1xfFKP2BlFLBk8pykN59Ma7-MuoAx4ZM5?usp=sharing#scrollTo=qIQGDW2o8gCu de Nunez y Evelio

def getCalculations(savesavi: bool = False, plot: bool = False):
  # --- 1. CONFIGURACIÓN DE RUTAS CORREGIDA ---
  # La ruta base de la carpeta que contiene los archivos TIF
  TIF_FOLDER = 'data/multispectral_images/TTADDA_NARO_2023_F1/drone_data/2023-05-18' 
  # La ruta completa del archivo de salida
  PATH_SAVI = os.path.join(TIF_FOLDER, 'savi_calculated_test_nunez.tif')

  print(f"Debug: {TIF_FOLDER}")

  # Rutas completas a las bandas (usando TIF_FOLDER)
  tifdirgb = os.path.join(TIF_FOLDER, '20230518_RGB.tif')
  tifdirred = os.path.join(TIF_FOLDER, '20230518_WUR_transparent_reflectance_red.tif')
  tifdirblue = os.path.join(TIF_FOLDER, '20230518_WUR_transparent_reflectance_blue.tif')
  tifdirgreen = os.path.join(TIF_FOLDER, '20230518_WUR_transparent_reflectance_green.tif')
  tifdirrededge = os.path.join(TIF_FOLDER, '20230518_WUR_transparent_reflectance_red edge.tif')
  tifdirnir = os.path.join(TIF_FOLDER, '20230518_WUR_transparent_reflectance_nir.tif')


  # --- 2. CARGA DE BANDAS Y CREACIÓN DE RGB ---
  rgb = rasterio.open(tifdirgb) # Abrimos el RGB para obtener metadata

  # Cargar las bandas como arrays de NumPy
  azul = rasterio.open(tifdirblue).read(1)
  verde = rasterio.open(tifdirgreen).read(1)
  rojo = rasterio.open(tifdirred).read(1)
  nir = rasterio.open(tifdirnir).read(1)
  reg = rasterio.open(tifdirrededge).read(1)

  # Crear imagen merge (para visualización)
  img_merge = cv2.merge((rojo, verde, azul))

  if plot:
    # Visualizar RGB
    plt.figure(figsize=(7, 7))
    # Multiplicar por 255 y convertir a uint8 es común para visualización si las reflectancias son 0-1
    plt.imshow(((img_merge * 255).astype(np.uint8)), interpolation='nearest') 
    plt.title("Imagen RGB (Visualización)")
    plt.savefig('rgb.png')
    plt.close()
  else:
    print(f'No se plotea RGB...')


  # --- 3. CÁLCULO DE ÍNDICES DE VEGETACIÓN ---

  # 3.1 CÁLCULO SAVI 
  # Usamos un L=0.5 (Soil Adjusted Vegetation Index)
  savi = np.where(
      (nir + rojo) == 0.0,
      -1, # Valor para evitar división por cero
      ((nir - rojo) / (nir + rojo + 0.5)) * (1 + 0.5)
  )

  # 3.2 CÁLCULO NDVI
  ndvi = np.where(
      (nir + rojo) == 0.,
      0,
      (nir - rojo) / (nir + rojo)
  )

  # 3.3 CÁLCULO CIg (Se ve incorrecto en tu fórmula, se asume NIR)
  # Tu fórmula original era: CIg = np.where((nir+green)==0., -0, ((nir)))
  # Esto no es un índice. Asumo que es el valor de NIR para tu cálculo.
  CIg = nir 
  np.max(CIg), np.min(CIg)


  # 3.4 CÁLCULO NDWI (Normalized Difference Water Index, usa Green y NIR)
  ndwi = np.where(
      (verde + nir) == 0, 
      -1,
      (verde - nir) / (verde + nir)
  )

  # 3.5 CÁLCULO NDRE (Normalized Difference Red Edge)
  ndre = np.where(
      (nir + reg) == 0, 
      -1, # Se necesita un valor si el divisor es cero
      (nir - reg) / (nir + reg)
  )


  # --- 4. EXPORTAR SAVI CORREGIDO (¡CRÍTICO!) ---
  if savesavi:
    print(f"Guardando SAVI en: {PATH_SAVI}")
    # Usamos 'with' para garantizar el cierre del archivo
    with rasterio.open(
        PATH_SAVI, 
        'w', 
        driver='GTiff',
        width=rgb.width, 
        height=rgb.height,
        count=1,
        crs=rgb.crs,
        transform=rgb.transform,
        dtype='float64' 
    ) as savi_src:
        # --- LA CORRECCIÓN CLAVE ---
        savi_src.write(savi, 1) # Escribir el array SAVI, no el objeto rgb
        # --------------------------
    print("Exportación SAVI completada.")
  else:
    print("No se exporta SAVI.")


  # --- 5. VISUALIZACIÓN DE ÍNDICES ---
  if plot:
    # Visualizar SAVI
    print(f'Ploteando SAVI...')
    plt.figure(figsize=(7, 7))
    plt.imshow(savi, cmap='RdYlGn') 
    plt.title("SAVI")
    plt.colorbar()
    plt.savefig('savi.png')
    plt.close()
    print(f'SAVI Ploteado...')

    # Visualizar NDVI
    print(f'Ploteando NDVI...')
    plt.figure(figsize=(7, 7))
    plt.imshow(ndvi, cmap='RdYlGn') 
    plt.title("NDVI")
    plt.colorbar()
    plt.savefig('ndvi.png')
    plt.close()
    print(f'NDVI Ploteado...')

    # Visualizar CIg (NIR)
    print(f'Ploteando NIR...')
    plt.figure(figsize=(7, 7))
    plt.imshow(CIg, cmap='Spectral') 
    plt.title("NIR (CIg - Asumed NIR)")
    plt.colorbar()
    plt.savefig('nir.png')
    plt.close()
    print(f'NIR Ploteado...')

    # Visualizar NDWI
    print(f'Ploteando NDWI...')
    plt.figure(figsize=(7, 7))
    plt.imshow(ndwi, cmap='Spectral', vmin=-1, vmax=1) 
    plt.title("NDWI")
    plt.colorbar()
    plt.savefig('ndwi.png')
    plt.close()
    print(f'NDWI Ploteado...')

    # Visualizar NDRE
    print(f'Ploteando NDRE...')
    plt.figure(figsize=(7, 7))
    plt.imshow(ndre, cmap='Spectral', vmin=-1, vmax=1) 
    plt.title("NDRE")
    plt.colorbar()
    plt.savefig('ndre.png')
    plt.close()
    print(f'NDRE Ploteado...')
  else:
    print(f'No se plotea...')

  print(f'Proceso culminado con exito...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtiene datos del satelite Sentinel-2 y calcula el NDVI"
    )
    parser.add_argument(
        "--savesavi", "-ss",
        action="store_true",
        default=False,
        help="Indica si debe guardar el SAVI.tif."
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        default=False,
        help="Indica si debe plotear la imagen .tif."
    )
    args = parser.parse_args()
    getCalculations(args.savesavi, args.plot)
