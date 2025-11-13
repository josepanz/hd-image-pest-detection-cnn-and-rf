from pystac_client import Client
import stackstac
import rioxarray
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
from shapely.geometry import shape
import sys # solo si se quiere computar antes para valores minimos y maximos y sum nvdi
import time
import argparse
import os
from datetime import datetime


# Variables globales
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'satelital-results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# 0. Registrar el tiempo de inicio
start_time = time.time()

def print_time_and_step(step_number, message):
    """Calcula y imprime el tiempo transcurrido desde el inicio."""
    elapsed = time.time() - start_time
    # Usar f-string para formatear el tiempo a segundos con 2 decimales
    print(f"\n--- [T={elapsed:.2f}s] ---")
    print(f"{step_number}. {message}")

def get_satellital_image(plot: bool = False, compute: bool = False, saveraster: bool = True, daterange: str = "2024-06-01/2024-09-30", imagetype: str = "box"):
  # === 1. Definir el AOI (IPTA Hernando Bertoni - Caacupé) ===
  print_time_and_step("1", f"Definiendo el AOI ({imagetype}) para (IPTA Hernando Bertoni - Caacupé)...")
  # # poligono, como feature de featureCollection
  if imagetype == "polygon":
    aoi = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "name": "IPTA Hernando Bertoni - Caacupé"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [
                  -57.193370500217284,
                  -25.38823273121325
                ],
                [
                  -57.19334380918299,
                  -25.389751855549605
                ],
                [
                  -57.19398439218915,
                  -25.389751855549605
                ],
                [
                  -57.19366410068632,
                  -25.391053946869732
                ],
                [
                  -57.19297013576259,
                  -25.391005721516336
                ],
                [
                  -57.19107507770187,
                  -25.394067993262894
                ],
                [
                  -57.19107507770187,
                  -25.394574345672453
                ],
                [
                  -57.186671069533034,
                  -25.394212665596825
                ],
                [
                  -57.18893980101416,
                  -25.391608537056158
                ],
                [
                  -57.189180019176746,
                  -25.39098160914135
                ],
                [
                  -57.18812815597191,
                  -25.391152863226438
                ],
                [
                  -57.187994700747794,
                  -25.390405368902634
                ],
                [
                  -57.18746088157573,
                  -25.390212466334177
                ],
                [
                  -57.18892888429849,
                  -25.38864512154244
                ],
                [
                  -57.193370500217284,
                  -25.38823273121325
                ]
              ]
            ]
          }
        }
      ]
    }
  elif imagetype == "box":
    #aoi como caja/cuadro box del mapa
    aoi = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "name": "IPTA Hernando Bertoni - Caacupé"
          },
          "geometry": {
            "coordinates": [
              [
                [
                  -57.194198571371174,
                  -25.388042396442614
                ],
                [
                  -57.194198571371174,
                  -25.394619456295416
                ],
                [
                  -57.18633520694212,
                  -25.394619456295416
                ],
                [
                  -57.18633520694212,
                  -25.388042396442614
                ],
                [
                  -57.194198571371174,
                  -25.388042396442614
                ]
              ]
            ],
            "type": "Polygon"
          }
        }
      ]
    }
  else:
    print("❌ AOI ERROR: El AOI debe ser 'polygon' o 'box'.")
    sys.exit(1)

  geometry = (
      aoi["features"][0]["geometry"]
      if aoi.get("type") == "FeatureCollection"
      else aoi
  )

  # === 2. Conectarse al catálogo de Sentinel-2 del Planetary Computer ===
  print_time_and_step("2", "Conectandose al catálogo del satelite Sentinel-2 del Planetary Computer...")
  catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

  search = catalog.search(
      collections=["sentinel-2-l2a"], # también se puede usar como collections el landsat-8-c2-l2
      intersects=geometry,
      datetime=daterange,
      query={"eo:cloud_cover": {"lt": 80}}, # Relajado a 80% (o incluso 100%) para incluir más imágenes para la mediana, 30% para mejor enmask
      limit=20 # Se puede aumentar el límite para tener más opciones para la mediana a 20
  )

  items = list(search.items())
  items = [planetary_computer.sign(item) for item in items]
  print(f"Encontradas {len(items)} imágenes firmadas y listas para descarga.")

  # 1. Convierte a objeto Shapely el geometry de tu aoi
  poly = shape(geometry)
  # 2. Obtiene el Bounding Box (xmin, ymin, xmax, ymax)
  aoi_bbox = poly.bounds 

  # === 3. Cargar las imágenes como un stack de arrays ===
  print_time_and_step("3", "Cargando imagenes en el stack (Preparando grafo de tareas)...")
  stack = stackstac.stack(
      items,
      assets=["B04", "B08", "SCL"],  # rojo y NIR se puede agregar si se quiere SCL (máscara de nubes)
      # bounds=aoi_bbox, # si se quiere delimitar
      resolution=20, # 10 o 20
      epsg=32721,
      chunksize=2048
  )

  # # === 4. Limpieza de Nubes y Cálculo de NDVI ===
  print_time_and_step("4", "Realizando limpieza de nubes y cálculo de NDVI...")
  # 4.1 Filtrado con SCL (Scene Classification Layer)
  # Códigos de SCL para nubes, sombra de nubes y agua (generalmente se enmascaran)
  # 3: Sombra de Nubes, 8: Nube media/alta, 9: Nube Cirro, 10: Nube opaca
  cloud_mask_values = [3, 8, 9, 10] # tambien se puede configurar [8, 9, 10] # Relajado: solo nubes densas
  print_time_and_step("4.1", "Limpiando nubes...")
  scl = stack.sel(band="SCL")

  # Crear una máscara booleana: True donde hay nubes/sombras/agua
  # cloud_mask_with_band = np.isin(scl, cloud_mask_values)
  cloud_mask_with_band = scl.isin(cloud_mask_values)
  cloud_mask = cloud_mask_with_band.squeeze() # Ahora forma (time, y, x)

  # stacked_data: Datos con 2 bandas (B04, B08)
  stacked_data = stack.sel(band=["B04", "B08"])

  # Aplicar máscara: los píxeles nublados se convierten en NaN
  # stack.where(condición) mantiene los valores donde la condición es True
  # stacked_masked = stack.sel(band=["B04", "B08"]).where(~cloud_mask)
  stacked_masked = stacked_data.where(~cloud_mask, other=np.nan)

  # === 4.2 Calcular NDVI ===
  print_time_and_step("4.2", "Calculando NDV1...")
  #red = stack.sel(band="B04").mean(dim="time")
  #nir = stack.sel(band="B08").mean(dim="time")

  red = stack.sel(band="B04").median(dim="time", skipna=True)
  nir = stack.sel(band="B08").median(dim="time", skipna=True)

  ndvi = (nir - red) / (nir + red)

  # El CRS de los datos de Sentinel-2 en stackstac (por defecto) es EPSG:4326
  ndvi = ndvi.rio.write_crs("EPSG:32721")

  # TODO: DESCOMENTAR O MANDAR -c COMO ARGUMENTO, SI ES QUE SE TIENE LA POTENCIA COMPUTACIONAL PARA OBTENER TODOS LOS CHUNKS
  # =============================================================================================
  if compute:
    print(f"4.3 Realizando validaciones...")
    print(f"DEBUG: Valor mínimo de NDVI antes del clip: {ndvi.min().compute().item():.4f}") # agregar ndvi.min().compute().item() si se quiere tener todos los datos del enorme array y traiga todos los chunks
    print(f"DEBUG: Valor máximo de NDVI antes del clip: {ndvi.min().compute().item():.4f}") # agregar ndvi.min().compute().item() si se quiere tener todos los datos del enorme array y traiga todos los chunks
    # Si ambos valores son NaN, el problema es que la máscara eliminó todos los datos.

    # Comprobar la cantidad de píxeles válidos
    total_pixels = ndvi.size
    valid_pixels = ndvi.notnull().sum() # agregar ndvi.notnull().sum().compute().item() si quiere todos los datos y no por chunks, alto costo computacional -> RAM
    print(f"DEBUG: Píxeles válidos encontrados: {valid_pixels} de {total_pixels}")

    if valid_pixels == 0:
        print("❌ ERROR DE DIAGNÓSTICO: La mediana temporal resultó en CERO píxeles válidos. Ajuste la máscara de nubes.")
        # Si esto ocurre, detener
        sys.exit(1)
  # =============================================================================================

  # # Recorta al AOI real antes de guardar
  ndvi = ndvi.rio.clip([geometry], crs="EPSG:4326")

  # === 5. Visualizar ===
  print_time_and_step("5", f"¿Se debe plotear (Fuerza la COMPUTE de Dask)? '{plot}'")
  if plot:
    print_time_and_step("5.1", "Ploteando imagen (Fuerza la COMPUTE de Dask)...")
    plt.figure(figsize=(8, 6))
    ndvi.plot(cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("NDVI - IPTA Caacupé (Sentinel-2, 2024)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plot_path = os.path.join(OUTPUT_DIR, f"NDVI_IPTA_Caacupe_{timestamp}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
  else:
    print_time_and_step("5.1", f"Omitiendo paso 5.1 Ploteando imagen...")

  # guarda el archivo 
  if saveraster:
    print_time_and_step("6.1", "Guardando archivo raster...")
    raster_path = os.path.join(OUTPUT_DIR, f"NDVI_IPTA_Caacupe_{daterange.replace("/", "_")}.tif")
    ndvi.rio.to_raster(raster_path)
    print_time_and_step("6.2", "El archivo NDVI ha sido guardado.")
  else:
    print_time_and_step("6.1", "Omitiendo paso 6.1 Guardando archivo raster...")
    
  print_time_and_step("✅ FINALIZADO", "\nProceso finalizado.")

def main():
    parser = argparse.ArgumentParser(
        description="Obtiene datos del satelite Sentinel-2 y calcula el NDVI"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        default=False,
        help="Indica si debe plotear la imagen .tif."
    )
    parser.add_argument(
        "--compute", "-c",
        action="store_true",
        default=False,
        help="Indica si debe computar antes para obtener los calculos."
    )
    parser.add_argument(
        "--saveraster", "-sv",
        action="store_false",
        default=True,
        help="Indica si debe guardar el archivo raster."
    )
    parser.add_argument(
        "--daterange", "-dr",
        default="2024-06-01/2024-09-30",
        help="Indica rango de fechas a consultar."
    )
    parser.add_argument(
        "--imagetype", "-it",
        default="box",
        help="Indica rango de fechas a consultar."
    )
    args = parser.parse_args()

    get_satellital_image(args.plot, args.compute, args.saveraster, args.daterange, args.imagetype)

if __name__ == "__main__":
    main()