import rasterio
import os
import argparse

def inspeccionar_tif(ruta_archivo_tif):
    """Abre un GeoTIFF e imprime información clave sobre sus bandas."""
    if not os.path.exists(ruta_archivo_tif):
        print(f"ERROR: Archivo no encontrado en la ruta: {ruta_archivo_tif}")
        return

    try:
        with rasterio.open(ruta_archivo_tif) as src:
            num_bandas = src.count
            print(f"--------------------------------------------------")
            print(f"Archivo: {os.path.basename(ruta_archivo_tif)}")
            print(f"Número de bandas: {num_bandas}")
            print(f"Dimensiones (Alto x Ancho): {src.height} x {src.width}")
            print(f"Sistema de Coordenadas (CRS): {src.crs}")
            
            # Intenta obtener las descripciones de las bandas (si están disponibles)
            if src.descriptions:
                print(f"Descripciones de bandas: {src.descriptions}")
            else:
                print("Descripciones de bandas no disponibles.")
                
            # Muestra los metadatos más detallados de las bandas
            print("Metadatos de Bandas (Longitud de Onda o Descripción):")
            for i in range(1, num_bandas + 1):
                # Usar tags para buscar descripciones o longitudes de onda
                tags = src.tags(i)
                wavelength = tags.get('WAVELENGTH') or tags.get('wavelength')
                description = tags.get('BandName') or src.descriptions[i-1] if src.descriptions else f"Banda {i}"

                info = f"  Banda {i}: {description}"
                if wavelength:
                    info += f" | Longitud de onda: {wavelength} nm"
                
                print(info)

    except rasterio.RasterioIOError as e:
        print(f"Error al intentar abrir el archivo TIF: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspecciona el archivo .tif verificar atributos y metadatos.")
    parser.add_argument("--tiff_path", "-tp", default="data/multispectral_images/TTADDA_NARO_2023_F1/drone_data/2023-06-05/20230605_DEM.tif", help="Ruta al archivo .tif")
    args = parser.parse_args()
    inspeccionar_tif(args.tiff_path)