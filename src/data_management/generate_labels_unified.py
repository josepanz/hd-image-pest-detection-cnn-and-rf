import argparse
import os
import numpy as np
import rasterio
import rasterio.mask
import fiona
from shapely.geometry import shape
import pandas as pd
from typing import List, Dict, Any, Tuple
import sys
# Importa iqr para el cálculo de umbrales robustos si es necesario, aunque aquí usamos .quantile()
# from scipy.stats import iqr 


# --- CONFIGURACIÓN DE RUTAS Y UMBRALES ---
BASE_DATA_DIR = os.path.join("data") 

# Directorios de imágenes y shapefiles (adaptados de tu código)
MULTISPECTRAL_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "multispectral_images", "TTADDA_NARO_2023_F1", "drone_data")
SHAPEFILES_DIR = os.path.join(BASE_DATA_DIR, "multispectral_images", "TTADDA_NARO_2023_F1", "metadata")

MULTISPECTRAL_IMAGES_DIR_E = os.path.join(BASE_DATA_DIR, "multispectral_images")
SHAPEFILES_DIR_E = os.path.join(BASE_DATA_DIR, "shapefiles")

# Rutas a los CSV de Ground Truth
YIELD_CSV_FILENAME = "NARO_field_2023_GT_yield.csv"
ADDITIONAL_GT_CSV_FILENAME = "NARO_field_2023_GT_additional.csv"

# Se asume que los CSV de GT están en la carpeta 'data/measurements'
YIELD_CSV_PATH_FULL = os.path.join(BASE_DATA_DIR,  "multispectral_images", "TTADDA_NARO_2023_F1", "measurements", YIELD_CSV_FILENAME)
ADDITIONAL_GT_CSV_PATH_FULL = os.path.join(BASE_DATA_DIR, "multispectral_images", "TTADDA_NARO_2023_F1", "measurements", ADDITIONAL_GT_CSV_FILENAME)

# Ruta base para el CSV de salida (se le añadirá el sufijo de la etiqueta final)
OUTPUT_CSV_BASE = os.path.join(BASE_DATA_DIR, "multispectral_images", "TTADDA_NARO_2023_F1", "measurements", "generated_labels.csv")


# --- UMBRALES DE CLASIFICACIÓN (AJUSTABLES) ---

# Umbrales de NDVI para clasificación de Dron
NDVI_HEALTHY_THRESHOLD = 0.5   
NDVI_DISEASED_THRESHOLD = 0.3  

# Umbrales basados en SPAD (clorofila, bajos valores indican estrés/deficiencia)
SPAD_HEALTHY_THRESHOLD = 45.0 
SPAD_DISEASED_THRESHOLD = 30.0 


# --- FUNCIONES DE CARGA Y ETIQUETADO DE GROUND TRUTH ---

def get_yield_label_mapping(yield_csv_path: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Carga el archivo de rendimiento, calcula cuartiles (Q1, Q3) y genera un mapeo 
    de etiquetas de rendimiento ('Plaga'/'Sana'/'Indeterminado') basado en tubwght_total_kgm-2.
    """
    try:
        # Intenta cargar los datos de rendimiento 
        df_yield = pd.read_csv(yield_csv_path)
        
        # Agrupar por obsUnitId para asegurar un solo rendimiento por parcela
        df_yield = df_yield.groupby('obsUnitId').first().reset_index()

        yield_col = 'tubwght_total_kgm-2'
        
        # Filtrar valores no positivos para el cálculo estadístico
        valid_yield = df_yield[df_yield[yield_col] > 0][yield_col]
        
        if valid_yield.empty:
            print("❌ ADVERTENCIA: No hay datos de rendimiento válidos para etiquetar.")
            return {}, {}

        # Calcular cuartiles (Q1 y Q3)
        Q1 = valid_yield.quantile(0.25)
        Q3 = valid_yield.quantile(0.75)
        
        # Umbrales
        YIELD_DISEASED_THRESHOLD = Q1 # 25% inferior -> Plaga
        YIELD_HEALTHY_THRESHOLD = Q3  # 25% superior -> Sana

        print(f"INFO Rendimiento (tubwght_total_kgm-2): Q1={Q1:.4f}, Q3={Q3:.4f}.")
        print(f"   Etiqueta 'Plaga' < {Q1:.4f} | Etiqueta 'Sana' > {Q3:.4f}")
        
        label_mapping = {}
        yield_mapping = {}
        for index, row in df_yield.iterrows():
            obs_id = row['obsUnitId']
            # Convertir a float, manejar NaN
            current_yield = row[yield_col] if pd.notna(row[yield_col]) else 0.0 
            
            if current_yield <= 0:
                label = "No_Cosecha" 
            elif current_yield < YIELD_DISEASED_THRESHOLD:
                label = "Plaga"
            elif current_yield > YIELD_HEALTHY_THRESHOLD:
                label = "Sana"
            else: # Entre Q1 y Q3
                label = "Indeterminado"

            label_mapping[obs_id] = label
            yield_mapping[obs_id] = current_yield

        return label_mapping, yield_mapping
    
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo de rendimiento no encontrado en {yield_csv_path}. La etiqueta de rendimiento no estará disponible.")
        return {}, {}
    except Exception as e:
        print(f"❌ ERROR al procesar datos de rendimiento: {e}")
        return {}, {}


def get_additional_gt_data(additional_csv_path: str) -> pd.DataFrame:
    """
    Carga los datos adicionales de GT (SPAD, NDVI de punto, etc.) y los retorna como DataFrame.
    """
    try:
        df_additional = pd.read_csv(additional_csv_path)
        # Asegurarse de que la columna de fecha sea tipo datetime 
        df_additional['collectionDate'] = pd.to_datetime(df_additional['collectionDate'])
        return df_additional
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo de datos adicionales no encontrado en {additional_csv_path}. Datos de SPAD/NDVI de punto no disponibles.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR al cargar datos adicionales: {e}")
        return pd.DataFrame()


# --- FUNCIONES DE CLASIFICACIÓN ---

def classify_by_ndvi(ndvi_value: float) -> str:
    """
    Clasifica una parcela como 'Sana', 'Plaga' o 'Indeterminado' basado en el valor de NDVI del dron.
    """
    if np.isnan(ndvi_value):
        return "Indeterminado_NDVI"
    
    if ndvi_value >= NDVI_HEALTHY_THRESHOLD:
        return "Sana"
    elif ndvi_value < NDVI_DISEASED_THRESHOLD and ndvi_value > 0: 
        return "Plaga"
    elif ndvi_value <= 0: 
        return "No_Vegetacion"
    else: 
        return "Indeterminado"

def classify_by_spad(spad_value: float) -> str:
    """
    Clasifica una parcela basado en el valor de SPAD (clorofila).
    """
    if np.isnan(spad_value) or spad_value <= 0:
        return "Indeterminado_SPAD"
    elif spad_value >= SPAD_HEALTHY_THRESHOLD:
        return "Sana"
    elif spad_value < SPAD_DISEASED_THRESHOLD: 
        return "Plaga" 
    else: 
        return "Indeterminado"


# --- FUNCIONES DE UTILIDAD DE ARCHIVOS ---

def get_band_path(tiff_folder: str, band_name: str) -> str:
    """
    Busca y devuelve la ruta completa del archivo para la banda dada de forma flexible.
    """
    # Componentes clave del nombre de archivo para las bandas
    BANDS_COMPONENTS = {
        'red': 'reflectance_red',
        'nir': 'reflectance_nir',
    }
    component = BANDS_COMPONENTS.get(band_name)
    if not component:
        raise ValueError(f"Componente clave no definido para la banda {band_name}.")

    # Busca el archivo en la carpeta que contenga el componente clave y termine en .tif
    # Excluye 'edge' de la búsqueda de 'red' para evitar mezclar bandas
    full_filename = next(
        (f for f in os.listdir(tiff_folder) if component in f.lower() and f.endswith('.tif') and ('edge' not in f.lower() if band_name == 'red' else True)), None
    )
    
    if not full_filename:
        raise FileNotFoundError(f"Falta el archivo para la banda {band_name} (componente: '{component}') en {tiff_folder}. Archivos encontrados: {os.listdir(tiff_folder)}")
        
    return os.path.join(tiff_folder, full_filename)


# --- FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---

def generate_labels_unified(
    multispectral_root_dir: str, 
    shapefiles_dir: str, 
    output_csv_path: str,
    label_source: str,
    yield_csv_path: str,
    additional_gt_csv_path: str
) -> None:
    """
    Calcula NDVI desde imágenes de dron e incorpora datos de Yield y SPAD/NDVI de punto.
    """
    all_processed_data = []

    # --- 0. Pre-carga de datos de Ground Truth ---
    yield_labels, actual_yields = get_yield_label_mapping(yield_csv_path)
    df_additional = get_additional_gt_data(additional_gt_csv_path)

    # Definir los shapefiles a procesar (incluyendo el de tu ejemplo)
    shapefile_names = [
        'plot_shapefile.shp', 
        # 'potato_measured_locations_konv.shp',
        # 'potato_measured_locations_eko.shp',
        # 'potato_locations_eko.shp',
        # 'potato_locations_konv.shp'
    ]

    # Iterar sobre cada subdirectorio de imágenes multiespectrales (e.g., 2023-05-18)
    for date_folder in os.listdir(multispectral_root_dir):
        
        tiff_folder_path = os.path.join(multispectral_root_dir, date_folder)
        if not os.path.isdir(tiff_folder_path) or 'ttadda_naro' in date_folder.lower():
            continue

        print(f"\nProcesando fecha: {date_folder}")
        
        # --- 1. Obtener y Abrir Archivos TIFF ---
        try:
            red_path = get_band_path(tiff_folder_path, 'red')
            nir_path = get_band_path(tiff_folder_path, 'nir')
            
            print(f"  DEBUG: Bandas encontradas - RED: {os.path.basename(red_path)}, NIR: {os.path.basename(nir_path)}")
        except (ValueError, FileNotFoundError) as e:
            print(f"  ❌ Saltando {date_folder} debido a error en la búsqueda de archivos: {e}")
            continue

        # 2. Abrir los archivos TIFF como objetos de origen
        try:
            with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
                
                # --- 3. Iterar sobre los Shapefiles ---
                for shp_name in shapefile_names:
                    shp_path = os.path.join(shapefiles_dir, shp_name)
                    
                    if not os.path.exists(shp_path):
                        continue

                    print(f"  Analizando Shapefile: {shp_name}")
                    
                    try:
                        with fiona.open(shp_path, 'r') as source_shp:
                            raster_crs = red_src.crs
                            nir_raster_crs = nir_src.crs
                            print(f'DEBUG: CRS del raster red: {raster_crs}')
                            print(f'DEBUG: CRS del raster nir: {nir_raster_crs}')
                            print(f'DEBUG: CRS del Shapefile: {source_shp.crs}')

                            for feature in source_shp:
                                # print(f'feature: {feature}')
                                geom = shape(feature['geometry'])
                                props = feature['properties']

                                # --- 3.1 Extracción de IDs y Metadata ---
                                try:
                                    # print(f'props: {props}')
                                    obs_unit_id = props.get('obsUnitId', props.get('PlotID'))
                                    obs_unit_id = str(obs_unit_id) if obs_unit_id is not None else ""
                                    obs_unit_id = f"TTADDA_NARO_2023_F1P{obs_unit_id}"
                                    
                                    # Intentar construir el ID si obsUnitId no es directamente el ID del CSV (TTADDA_NARO_2023_F1P1)
                                    if obs_unit_id and not obs_unit_id.startswith('TTADDA'):
                                        tipo_cultivo = 'eko' if 'eko' in shp_name.lower() else 'konv'
                                        blok = props.get('Blok')
                                        rastlina = props.get('Rastlina')
                                        sorta = props.get('Sorta')
                                        
                                        if blok is not None and rastlina is not None and sorta is not None:
                                            plot_id = f"{tipo_cultivo}-{int(blok)}-{sorta}-{int(rastlina)}" 
                                        else:
                                            plot_id = obs_unit_id
                                    elif obs_unit_id:
                                        plot_id = obs_unit_id
                                    else:
                                        # No hay ID válido para unión
                                        continue

                                    # Obtener el rendimiento final y etiqueta GT
                                    yield_label = yield_labels.get(obs_unit_id, yield_labels.get(plot_id, "Sin_GT_Rendimiento"))
                                    actual_yield = actual_yields.get(obs_unit_id, actual_yields.get(plot_id, 0.0))
                                    
                                    # Obtener SPAD y NDVI de punto del CSV adicional
                                    current_date = pd.to_datetime(date_folder)
                                    spad_mean, ndvi_gt, spad_label = np.nan, np.nan, "Sin_GT_SPAD"

                                    if not df_additional.empty:
                                        # Buscar la medición más cercana a la fecha de la imagen
                                        gt_data = df_additional[df_additional['obsUnitId'] == obs_unit_id].copy()
                                        if not gt_data.empty:
                                            # Calcular diferencia de días
                                            gt_data['date_diff'] = abs(gt_data['collectionDate'] - current_date).dt.days
                                            closest_measurement = gt_data.loc[gt_data['date_diff'].idxmin()]
                                            
                                            spad_mean = closest_measurement['SPAD'] if pd.notna(closest_measurement['SPAD']) else np.nan
                                            ndvi_gt = closest_measurement['NDVI'] if pd.notna(closest_measurement['NDVI']) else np.nan
                                            spad_label = classify_by_spad(spad_mean)
                                    
                                except Exception as e:
                                    print(f'Omitiendo parcela {obs_unit_id}. Error: {e}')
                                    continue # Omitir parcelas sin datos de identificación/GT

                                # --- 3.2 Enmascarar y Calcular NDVI (Dron) ---
                                try:
                                    red_patch_arr, _ = rasterio.mask.mask(red_src, [geom], crop=True, filled=False, all_touched=True)
                                    nir_patch_arr, _ = rasterio.mask.mask(nir_src, [geom], crop=True, filled=False, all_touched=True)
                                    
                                    red_values = red_patch_arr[0].flatten()
                                    nir_values = nir_patch_arr[0].flatten()
                                    
                                    red_valid = red_values[(red_values > 0) & (red_values <= 65535)]
                                    nir_valid = nir_values[(nir_values > 0) & (nir_values <= 65535)]

                                    # print(f"DEBUG (VALIDACIÓN): PlotID: {obs_unit_id} - Píxeles RED válidos: {len(red_valid)}, Píxeles NIR válidos: {len(nir_valid)}")

                                    if len(red_valid) < 5 or len(nir_valid) < 5: 
                                        print(f"❌ DEBUG (DESCARTADA): Parcela {obs_unit_id} descartada por insuficientes píxeles válidos.")
                                        continue

                                    # Normalización a Reflectancia (0-1) antes de NDVI (Asumiendo 10000 como factor de escala)
                                    red_mean = np.mean(red_valid).astype(np.float64) / 10000.0
                                    nir_mean = np.mean(nir_valid).astype(np.float64) / 10000.0
                                    
                                    denominator = (nir_mean + red_mean)
                                    ndvi_drone = (nir_mean - red_mean) / denominator if denominator != 0 else 0.0

                                    ndvi_drone_label = classify_by_ndvi(ndvi_drone)

                                    # print(f"✅ DEBUG (RESULTADO): {obs_unit_id} | NDVI Dron: {ndvi_drone:.4f} | Etiqueta Final (Yield): {yield_label}")

                                except Exception:
                                    ndvi_drone = np.nan
                                    ndvi_drone_label = "No_Data_Dron"
                                    red_mean = np.nan
                                    nir_mean = np.nan
                                
                                # --- 3.3 Definir la Etiqueta Final ---
                                # final_label = ""
                                # if label_source == 'yield':
                                #     final_label = yield_label
                                # elif label_source == 'ndvi':
                                #     final_label = ndvi_drone_label
                                # elif label_source == 'spad':
                                #     final_label = spad_label
                                # else:
                                #     final_label = yield_label # Default

                                # --- 3.3 Definir la Etiqueta Final (UNIFICADA) ---
                                # Ahora, la etiqueta final se calcula mediante la función de consenso
                                final_label = unify_labels(
                                    yield_label=yield_label,
                                    spad_label=spad_label,
                                    ndvi_drone_label=ndvi_drone_label
                                )
                                
                                # --- 4. Guardar los datos ---
                                all_processed_data.append({
                                    'Fecha': date_folder,
                                    'PlotID': plot_id,
                                    'obsUnitId': obs_unit_id, # El ID de unión
                                    'Yield_kgm2': actual_yield,
                                    'Etiqueta_Rendimiento': yield_label,
                                    'SPAD_Punto': spad_mean,
                                    'Etiqueta_SPAD': spad_label,
                                    'NDVI_Punto': ndvi_gt,
                                    'Mean_Red_Reflectance_Dron': red_mean,
                                    'Mean_NIR_Reflectance_Dron': nir_mean,
                                    'NDVI_Dron': ndvi_drone,
                                    'Etiqueta_NDVI_Dron': ndvi_drone_label,
                                    'Etiqueta_FINAL': final_label 
                                })

                    except fiona.errors.DriverError as e:
                        print(f"  ❌ ERROR: No se pudo abrir Shapefile {shp_path}: {e}")
                        continue
        
        except rasterio.RasterioIOError as e:
            print(f"  ❌ Error al abrir archivos raster en {tiff_folder_path}: {e}")
            continue
        except Exception as e:
            print(f"  ❌ ERROR DESCONOCIDO en fecha {date_folder}: {e}")
            continue

    
    # --- 5. Escribir a CSV ---
    if all_processed_data:
        df_output = pd.DataFrame(all_processed_data)
        df_output = df_output.sort_values(by=['obsUnitId', 'Fecha'])
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Renombrar el archivo de salida para reflejar la fuente de la etiqueta
        base_name, ext = os.path.splitext(os.path.basename(output_csv_path))
        new_output_csv_path = os.path.join(os.path.dirname(output_csv_path), f"{base_name}_{label_source}{ext}")

        df_output.to_csv(new_output_csv_path, index=False)
        print(f"\n✅ Etiquetas generadas y guardadas en: {new_output_csv_path}")
        print(f"La 'Etiqueta_FINAL' se basa en: {label_source.upper()}")
        print(f"Total de registros procesados: {len(df_output)}")
    else:
        print("\n❌ No se procesaron datos y no se generó el archivo CSV.")

def unify_labels(yield_label: str, spad_label: str, ndvi_drone_label: str) -> str:
    """
    Genera una etiqueta final de consenso ('Plaga', 'Sana', 'Indeterminado') 
    aplicando una lógica de riesgo/jerarquía a los tres indicadores.

    Jerarquía: Plaga > Indeterminado > Sana (dada la naturaleza de los datos)
    (A menos que haya datos fuertes de 'Sana' que superen a 'Indeterminado').
    """
    
    # 1. Recolectar todas las etiquetas
    etiquetas = [yield_label, spad_label, ndvi_drone_label]

    # 2. Manejar etiquetas de "Sin Datos" o "No Vegetación"
    
    # Excluir etiquetas de Sin_GT y No_Data, centrándonos en las etiquetas de clasificación
    # Las etiquetas "Indeterminado" son un resultado de clasificación válido (Q1-Q3, o umbrales intermedios)
    etiquetas_validas = [
        e for e in etiquetas 
        if e not in ["Sin_GT_Rendimiento", "Sin_GT_SPAD", "Indeterminado_NDVI", "Indeterminado_SPAD", "No_Vegetacion", "No_Cosecha"]
    ]

    # Si no hay ninguna etiqueta de clasificación válida
    if not etiquetas_validas:
        # Se prioriza la falta de datos por Rendimiento, que es el objetivo final
        if "Sin_GT_Rendimiento" in etiquetas:
            return "Sin_GT_Rendimiento"
        return "Incierto (No hay datos)"
    
    # 3. Aplicar Regla de Consenso Estricto (Jerarquía)
    
    # Prioridad 1: Riesgo (Plaga)
    # Si al menos un indicador fuerte sugiere 'Plaga', clasificamos como 'Plaga' por precaución.
    if 'Plaga' in etiquetas_validas:
        return 'Plaga'

    # Prioridad 2: Salud (Sana)
    # Si al menos dos indicadores válidos sugieren 'Sana', o todos los válidos son 'Sana'
    sana_count = etiquetas_validas.count('Sana')
    if sana_count >= 2 or (sana_count == 1 and len(etiquetas_validas) == 1):
        return 'Sana'
    
    # Prioridad 3: Indeterminado
    # Si las etiquetas restantes son 'Indeterminado' o una mezcla no concluyente
    return 'Indeterminado'

# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print("Iniciando generación de etiquetas de rendimiento, SPAD y NDVI...")
    
    # Asegurar que los directorios necesarios existan para evitar FileNotFoundError de os.listdir()
    os.makedirs(MULTISPECTRAL_IMAGES_DIR_E, exist_ok=True) 
    os.makedirs(SHAPEFILES_DIR_E, exist_ok=True) 
    os.makedirs(os.path.dirname(OUTPUT_CSV_BASE), exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Genera etiquetas unificadas (Yield, NDVI, SPAD) para parcelas agrícolas."
    )
    
    # Argumento para la fuente de la etiqueta final
    parser.add_argument(
        "--label_source", "-l",
        default="unified",
        choices=["unified", "yield", "ndvi", "spad"],
        help="Fuente principal para la columna 'Etiqueta_FINAL': 'unified' (por defecto), 'yield' (rendimiento), 'ndvi' (dron), o 'spad' (punto de campo)."
    )
    
    # Mantenemos el argumento -tft para compatibilidad, aunque ya no es tan necesario
    parser.add_argument(
        "--tifftype", "-tft",
        default="eko", # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )

    parser.add_argument(
        "--shapedir", "-spd",
        default=SHAPEFILES_DIR, # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )

    parser.add_argument(
        "--multidir", "-msd",
        default=MULTISPECTRAL_IMAGES_DIR, # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )

    parser.add_argument(
        "--yielddir", "-yd",
        default=YIELD_CSV_PATH_FULL, # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )

    parser.add_argument(
        "--additionaldir", "-addir",
        default=ADDITIONAL_GT_CSV_PATH_FULL, # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )

    parser.add_argument(
        "--outcsv", "-ocsv",
        default=OUTPUT_CSV_BASE, # Se ignora, pero se mantiene la estructura
        help="Para saber cual data usar (ahora se usa el directorio principal 'multispectral_images')."
    )
    
    args = parser.parse_args()

    # Se invoca la función unificada
    generate_labels_unified(
        multispectral_root_dir=args.multidir,
        shapefiles_dir=args.shapedir,
        output_csv_path=args.outcsv,
        label_source=args.label_source,
        yield_csv_path=args.yielddir,
        additional_gt_csv_path=args.additionaldir
    )
    
    print("Proceso de generación de etiquetas completado.")