import os
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import numpy as np
import argparse
from typing import Tuple, List

# --- CONFIGURACIÓN ---
# CLASES que nos interesan (debes verificar qué campo del shapefile contiene 'Healthy' y 'Diseased')
# Asumimos que el Shapefile tiene un campo llamado 'clase' o 'etiqueta' con valores 0 y 1.
# Modifica 'FIELD_NAME' y 'LABEL_MAP' si tus datos son diferentes.

FIELD_NAME = 'PlotID' # Placeholder. DEBES REVISAR EL NOMBRE REAL del campo de etiqueta en el DBF.
LABEL_MAP = { 
    'Healthy': 'Sana',      # Reemplaza 'Healthy' con el valor exacto del shapefile
    'Diseased': 'Plaga'     # Reemplaza 'Diseased' con el valor exacto del shapefile
}
CLASSES = ["Plaga", "Sana"] 

def extract_patches_from_shapefile(
    tiff_path: str, 
    shapefile_path: str, 
    output_root_dir: str, 
    patch_size: int = 224,
    label_field: str = FIELD_NAME,
    labels_to_extract: dict = LABEL_MAP
) -> None:
    """
    Carga un TIFF multiespectral y un Shapefile de polígonos, recorta parches 
    (patches) dentro de esos polígonos y los guarda en carpetas de etiquetas.

    Args:
        tiff_path: Ruta a la imagen TIFF multiespectral (ej. con 5 bandas).
        shapefile_path: Ruta al archivo .shp (sin extensión).
        output_root_dir: Directorio base donde se guardarán los parches (e.g., data/train_patches).
        patch_size: Tamaño de los parches cuadrados (ej. 224x224).
        label_field: Nombre del campo en el Shapefile que contiene la etiqueta (e.g., 'clase').
        labels_to_extract: Diccionario para mapear las etiquetas del Shapefile a tus clases.
    """
    if not os.path.exists(tiff_path) or not os.path.exists(shapefile_path):
        print("❌ Error: Rutas TIFF o Shapefile no encontradas.")
        return

    # 1. Crear directorios de salida
    for label in labels_to_extract.values():
        os.makedirs(os.path.join(output_root_dir, label), exist_ok=True)
    
    # Contadores
    patch_count = 0
    
    # 2. Abrir el Shapefile (fiona) y el TIFF (rasterio)
    try:
        with fiona.open(shapefile_path, "r") as shp:
            with rasterio.open(tiff_path) as src:
                
                # Iterar sobre cada polígono/registro en el Shapefile
                for feature in shp:
                    # El campo de la etiqueta real
                    true_label_shp = feature['properties'].get(label_field)
                    
                    if true_label_shp not in labels_to_extract:
                        continue # Saltar etiquetas que no son Plaga o Sana (e.g., Soil, Weed)

                    # Mapear a tu etiqueta de tesis (Plaga/Sana)
                    target_label = labels_to_extract[true_label_shp]
                    output_dir = os.path.join(output_root_dir, target_label)
                    
                    # 3. Recortar la imagen con el polígono (rasterio.mask)
                    # El resultado es un recorte del TIFF que contiene solo el área del polígono
                    try:
                        geom = [feature['geometry']]
                        out_image, out_transform = mask(src, geom, crop=True, filled=True)
                        out_meta = src.meta.copy()

                        # Si el recorte falla o tiene dimensiones cero, saltar
                        if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                            continue

                        # 4. Iterar y guardar los parches (patches) dentro del recorte
                        # Se hace un recorte deslizable (sliding window) sobre la imagen recortada
                        # Asegúrate de que el recorte tenga 5 bandas (Canales, Altura, Ancho)
                        if out_image.shape[0] != 5:
                          print(f"⚠️ Aviso: Recorte {true_label_shp} tiene {out_image.shape[0]} bandas, se esperaba 5. Revisar TIFF.")
                          continue
                          
                        for r in range(0, out_image.shape[1] - patch_size + 1, patch_size):
                            for c in range(0, out_image.shape[2] - patch_size + 1, patch_size):
                                
                                # Definir la ventana del parche
                                window = Window(c, r, patch_size, patch_size)
                                patch = out_image[:, window.row_off:window.row_off + patch_size, window.col_off:window.col_off + patch_size]
                                
                                # Si el parche no tiene el tamaño correcto o tiene datos nulos (e.g., -9999)
                                if patch.shape[1] != patch_size or patch.shape[2] != patch_size or np.any(patch <= 0):
                                    continue
                                
                                # 5. Guardar el parche
                                # Adaptar metadatos para guardar el nuevo recorte
                                patch_meta = out_meta.copy()
                                patch_meta.update({
                                    "driver": "GTiff",
                                    "height": patch_size,
                                    "width": patch_size,
                                    "transform": src.window_transform(window)
                                })
                                
                                filename = f"{target_label}_{patch_count}_{os.path.basename(tiff_path).replace('.tif', '')}.tif"
                                output_path = os.path.join(output_dir, filename)
                                
                                with rasterio.open(output_path, "w", **patch_meta) as dst:
                                    dst.write(patch)
                                    
                                patch_count += 1
                                
                    except Exception as e:
                        print(f"❌ Error al enmascarar/recortar polígono {true_label_shp}: {e}")
                        continue
                        
    except Exception as e:
        print(f"❌ Error general al abrir archivos: {e}")
        return

    print("\n--- Resumen ---")
    print(f"✅ Extracción completada. {patch_count} parches generados en: {output_root_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extrae parches de TIFFs usando polígonos de un Shapefile para etiquetar datos.")
    parser.add_argument("tiff_path", help="Ruta al archivo TIFF multiespectral de 5 bandas.")
    parser.add_argument("shapefile_path", help="Ruta al archivo plot_shapefile.shp.")
    parser.add_argument("output_dir", help="Directorio raíz donde se crearán las subcarpetas 'Plaga' y 'Sana'.")
    parser.add_argument("-s", "--size", type=int, default=224, help="Tamaño de los parches cuadrados (ej. 224).")
    parser.add_argument("-l", "--label_field", default=FIELD_NAME, help="Nombre de la columna en el Shapefile con las etiquetas.")

    args = parser.parse_args()
    
    # --- PASO CRÍTICO: REVISIÓN DE ETIQUETAS ---
    # DEBES ASEGURARTE DE QUE LOS VALORES EN EL DICCIONARIO COINCIDAN CON LOS DEL Shapefile
    # Si la columna 'PlotID' contiene los valores 'A1', 'A2', etc., NO es la columna de etiqueta.
    # Debes abrir el archivo .dbf (ej. con QGIS o una herramienta GIS) para ver qué columna tiene 'Healthy'/'Diseased' (o 0/1).
    
    print("\n⚠️ VERIFICACIÓN CRÍTICA: ¿Estás seguro de que 'PlotID' es la columna de etiqueta?")
    print("Si no lo es, edita el script o usa la opción -l con el nombre correcto de la columna (ej. 'Clase').")
    
    extract_patches_from_shapefile(
        tiff_path=args.tiff_path,
        shapefile_path=args.shapefile_path,
        output_root_dir=args.output_dir,
        patch_size=args.size,
        label_field=args.label_field,
        labels_to_extract=LABEL_MAP # Usa el mapeo definido arriba
    )

if __name__ == "__main__":
    main()