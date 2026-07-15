"""Motor de inferencia vigente (usado por pest_detection/cli/infer.py):
descubre carpetas de muestra (RGB o multiespectral) bajo una ruta dada, las carga y
preprocesa, y corre la predicción con un modelo CNN o Random Forest ya cargado.
"""

import os
import cv2
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from typing import List, Dict, Any, Optional, Tuple

# Extensiones y sufijos para bandas multiespectrales (Ajustar según tu dataset)
BAND_SUFFIXES = [
    "_blue.tif",
    "_green.tif",
    "_red.tif",
    "_red edge.tif",
    "_nir.tif"
]

# Columna del shapefile de parcelas que identifica cada polígono (ver también
# SHP_ID_COLUMN en pest_detection/datasets/extract_data_to_img.py, mismo esquema).
SHP_ID_COLUMN = 'PlotID'

from pest_detection.print_utils import print_time_and_step
# Reexportado por compatibilidad con infer.py: la carga de modelo vive en
# model_loading.py (evaluate.py también la usa) para no tener dos implementaciones de
# "cargar modelo" a la vez.
from pest_detection.evaluation.model_loading import load_model_for_inference
import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def is_sample_folder(path, is_ms):
    """
    Determina si una carpeta es una muestra procesable.
    Si es MS: Debe contener las 5 bandas.
    Si es RGB: Debe contener al menos un archivo que termine en 'rgb.tif'.
    """
    files = [f.lower() for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if is_ms:
        # Verifica que estén los 5 sufijos
        return all(any(f.endswith(s) for f in files) for s in BAND_SUFFIXES)
    else:
        # Verifica que exista el archivo rgb.tif
        return any(f.endswith("rgb.tif") for f in files)

def get_all_sample_folders(root_path, is_ms):
    """
    Busca recursivamente todas las carpetas que cumplan con ser una muestra.

    Para RGB, root_path también puede ser directamente un archivo *rgb.tif suelto
    (no solo una carpeta que lo contenga): load_and_preprocess_image ya sabía manejar
    ese caso, pero antes este descubridor no lo reconocía como muestra y devolvía una
    lista vacía sin avisar (0 muestras encontradas, sin error).
    """
    sample_folders = []

    if not is_ms and os.path.isfile(root_path) and root_path.lower().endswith("rgb.tif"):
        return [root_path]

    # Si el path ya es una muestra, lo retornamos directamente
    if os.path.isdir(root_path) and is_sample_folder(root_path, is_ms):
        return [root_path]

    # Si no, caminamos por el árbol de directorios
    for root, dirs, files in os.walk(root_path):
        if is_sample_folder(root, is_ms):
            sample_folders.append(root)
            # Si encontramos una carpeta de muestra, no seguimos entrando en sus subcarpetas
            # para evitar duplicados o errores
            dirs[:] = [] 
            
    return sample_folders

def run_inference_on_path(model, feature_extractor_rf, path, threshold, img_size, model_name, is_multiespectral, is_random_forest):
    """Corre inferencia sobre todas las muestras encontradas bajo `path` (una imagen
    RGB, o recursivamente carpetas de bandas MS/subcarpetas RGB).

    feature_extractor_rf: solo aplica si is_random_forest=True; es la CNN cortada que
    convierte la imagen en el vector de features que espera el RF (ver
    inference_models.py::run_unified_inference, que arma este extractor cargando la
    CNN "hermana" del mismo tipo RGB/MS y loss). Si es None y is_random_forest=True,
    se le pasan los píxeles crudos aplanados al RF (probablemente incorrecto salvo que
    el RF se haya entrenado así a propósito).

    umbral (threshold) solo decide la etiqueta final para CNN (Sana si prob_sana >=
    threshold); para RF se usa directamente el argmax de predict_proba.
    """
    results = []
    
    # 1. Identificar todas las muestras (recursivo)
    print_time_and_step('riop 1', f"🔎 Buscando muestras en: {path}", timestamp=timestamp, start_time=start_time)
    sample_folders = get_all_sample_folders(path, is_multiespectral)
    print_time_and_step('riop 2', f"✅ Se encontraron {len(sample_folders)} muestras para procesar.", timestamp=timestamp, start_time=start_time)

    # 2. Extraer modelo (y scaler, si viene) del diccionario (RF)
    actual_model = model
    scaler = None
    if is_random_forest and isinstance(model, dict):
        actual_model = model.get('model') or model.get('rf_model') or list(model.values())[0]
        # BUG CORREGIDO: train.py::run_rf_training guarda el StandardScaler ajustado
        # sobre las features de la CNN junto con el RF (mismo bundle .joblib) y
        # evaluate.py::run_evaluation_rf sí lo aplica antes de predecir, pero acá nunca
        # se extraía ni se aplicaba: se le pasaban al RF features sin escalar, distintas
        # a las que vio en entrenamiento. Un árbol de decisión no es invariante a esto:
        # los umbrales de corte que aprendió son numéricamente los de la escala de
        # entrenamiento, así que aplicarlos a features sin escalar produce splits
        # arbitrarios.
        scaler = model.get('scaler')

    # 3. Procesar cada carpeta encontrada
    for sample_path in sample_folders:
        print_time_and_step('riop 3', f"🚀 Procesando muestra: {os.path.basename(sample_path)}", timestamp=timestamp, start_time=start_time)

        # Si hay un shapefile de parcelas junto a la muestra (mismo esquema que usó el
        # entrenamiento: <root>/drone_data/<fecha>/*.tif + <root>/metadata/plot_shapefile.shp),
        # recortamos y clasificamos una imagen por parcela, igual que en entrenamiento
        # (antes se clasificaba la imagen completa redimensionada a 224x224, una escala
        # visual muy distinta a la de un parche de una sola parcela). Si no se encuentra
        # shapefile (datasets sintéticos de test, u otra estructura), se cae al
        # comportamiento anterior de imagen completa.
        shapefile_path = find_parcels_shapefile(sample_path)
        if shapefile_path is not None:
            items = load_and_preprocess_parcels(sample_path, shapefile_path, img_size, is_multiespectral)
            fecha_label = os.path.basename(sample_path if os.path.isdir(sample_path) else os.path.dirname(sample_path))
            items = [(f"{fecha_label}_parcela_{parcel_id}", img) for parcel_id, img in items]
        else:
            img_data = load_and_preprocess_image(sample_path, img_size, is_multiespectral)
            items = [(os.path.basename(sample_path), img_data)] if img_data is not None else []

        for file_name, img_data in items:
            x = np.expand_dims(img_data, axis=0)

            if is_random_forest:
                if feature_extractor_rf is not None:
                    features = feature_extractor_rf.predict(x, verbose=0)
                    x_input = features.reshape(1, -1)
                    if scaler is not None:
                        x_input = scaler.transform(x_input)
                else:
                    x_input = x.reshape(1, -1)

                try:
                    probs = actual_model.predict_proba(x_input)[0]
                    prob_sana = float(probs[1])
                except ValueError as e:
                    print_time_and_step('riop 4', f"❌ Error de dimensiones en RF: {e}", timestamp=timestamp, start_time=start_time)
                    continue
            else:
                prediction = model.predict(x, verbose=0)
                prob_sana = float(prediction[0][0])

            prob_plaga = 1.0 - prob_sana
            prediccion = "Sana" if prob_sana >= threshold else "Plaga"

            results.append({
                "file_name": file_name, # Nombre de la carpeta (ej. 2021-05-25), o "<fecha>_parcela_<id>" si se recortó por parcela
                "path": sample_path,
                "prob_sana": round(prob_sana, 4),
                "prob_plaga": round(prob_plaga, 4),
                "prediccion": prediccion,
                "umbral": threshold,
                "modelo": model_name
            })

    return results

def find_parcels_shapefile(sample_path: str) -> Optional[str]:
    """Busca `metadata/plot_shapefile.shp` subiendo desde la muestra hacia la raíz del
    dataset, con el mismo esquema que usa el entrenamiento
    (`<root>/drone_data/<fecha>/*.tif` y `<root>/metadata/plot_shapefile.shp` como
    carpetas hermanas, ver extract_data_to_img.py). `sample_path` puede ser la carpeta
    de fecha (MS, o RGB pasado como carpeta) o un archivo *rgb.tif suelto.

    Devuelve None si no se encuentra (datasets sintéticos de test, u otra estructura)
    - el caller cae entonces al comportamiento anterior de imagen completa.
    """
    date_folder = sample_path if os.path.isdir(sample_path) else os.path.dirname(sample_path)
    dataset_root = os.path.dirname(os.path.dirname(date_folder))
    candidate = os.path.join(dataset_root, 'metadata', 'plot_shapefile.shp')
    return candidate if os.path.exists(candidate) else None

def _find_band_paths(date_folder: str, is_ms: bool) -> Optional[List[str]]:
    """Ubica, dentro de `date_folder`, el/los TIFF de banda a usar (5 sufijos MS, o el
    *rgb.tif), sin importar el prefijo de fecha del nombre de archivo (mismo criterio
    tolerante ya usado por `is_sample_folder`/`get_all_sample_folders`)."""
    files = [f for f in os.listdir(date_folder) if os.path.isfile(os.path.join(date_folder, f))]
    if is_ms:
        band_paths = []
        for suffix in BAND_SUFFIXES:
            match = next((f for f in files if f.lower().endswith(suffix)), None)
            if match is None:
                return None
            band_paths.append(os.path.join(date_folder, match))
        return band_paths
    else:
        match = next((f for f in files if f.lower().endswith("rgb.tif")), None)
        return [os.path.join(date_folder, match)] if match else None

def load_and_preprocess_parcels(sample_path: str, shapefile_path: str, img_size, is_ms: bool) -> List[Tuple[str, np.ndarray]]:
    """Recorta cada parcela de `shapefile_path` de las bandas/RGB de la muestra,
    replicando el mismo preprocesamiento que usó el entrenamiento
    (extract_data_to_img_for_train): recorte por polígono vía rasterio.mask, resize a
    `img_size`, y la misma normalización (`/max` de la imagen recortada en
    multiespectral, `/255.0` fijo en RGB - la escala 0-255 sí es correcta para RGB
    uint8, a diferencia de las bandas de reflectancia MS que vienen en escala 0-1).
    RGB con más de 3 bandas (alpha) se recorta a las primeras 3, igual que en
    extract_data_to_img.py.

    Devuelve una lista de (parcel_id, array (H, W, C)); las parcelas cuyo recorte
    falla (fuera de rango, polígono sin intersección, etc.) se omiten.
    """
    date_folder = sample_path if os.path.isdir(sample_path) else os.path.dirname(sample_path)

    if is_ms:
        band_paths = _find_band_paths(date_folder, True)
    else:
        band_paths = [sample_path] if os.path.isfile(sample_path) else _find_band_paths(date_folder, False)

    if not band_paths:
        return []

    parcels_gdf = gpd.read_file(shapefile_path)
    expected_channels = len(band_paths) if is_ms else 3

    items = []
    for _, row in parcels_gdf.iterrows():
        try:
            bands_clipped = []
            for band_path in band_paths:
                with rasterio.open(band_path) as src:
                    out_band, _ = rio_mask(src, [row.geometry], crop=True)
                    # Mismo fix que extract_data_to_img.py: nodata (-10000 en las bandas
                    # de reflectancia reales) se pone en 0 antes de normalizar, si no
                    # contamina el /max con valores del orden de -millones.
                    if src.nodata is not None:
                        out_band = np.where(out_band == src.nodata, 0, out_band)
                    bands_clipped.append(out_band)

            if is_ms:
                stacked = np.concatenate(bands_clipped, axis=0)
            else:
                stacked = bands_clipped[0]
                if stacked.shape[0] > 3:
                    stacked = stacked[:3]

            img = np.transpose(stacked, (1, 2, 0))
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

            if is_ms:
                max_val = np.max(img)
                if max_val > 0:
                    img = img / max_val
            else:
                img = img.astype(np.float32) / 255.0

            img = img.astype(np.float32)
            if img.shape[-1] != expected_channels:
                continue

            items.append((str(row[SHP_ID_COLUMN]), img))
        except Exception as e:
            print_time_and_step('riop parcela err', f"❌ Error recortando parcela {row.get(SHP_ID_COLUMN, '?')}: {e}", timestamp=timestamp, start_time=start_time)
            continue

    return items

def load_and_preprocess_image(path, img_size, is_ms):
    """
    Carga y preprocesa una imagen. 
    Soporta carpeta de bandas para MS y busca el archivo rgb.tif para RGB.
    """
    try:
        if is_ms:
            # Lógica Multiespectral: Cargar las 5 bandas desde la carpeta 'path'
            #
            # BUG CONFIRMADO Y CORREGIDO contra TIFFs reales (WUR_transparent_reflectance_*,
            # ver EJECUCION.md): son reflectancias float32 en escala ~0-0.01 (nodata=-10000),
            # no valores 0-255. El entrenamiento (extract_data_to_img_for_train) normaliza
            # cada imagen dividiendo por su propio máximo (resized_image / max_val); acá
            # antes se dividía siempre por un /255.0 fijo, que sobre esa escala aplasta
            # todo a ~1e-7 (prácticamente cero) - coincide con las probabilidades ~0.50
            # vistas en BITACORA_INFERENCE*.md para el modelo MULTIESPECTRAL. Ahora se
            # normaliza igual que en entrenamiento: por el máximo de la imagen apilada.
            bands = []
            for suffix in BAND_SUFFIXES:
                band_path = next((os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(suffix)), None)

                if band_path is None:
                    raise FileNotFoundError(f"Falta banda {suffix} en {path}")

                band = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
                if band is None: raise ValueError(f"No leíble: {band_path}")

                # Mismo fix de nodata que extract_data_to_img.py/load_and_preprocess_parcels:
                # cv2 no preserva el nodata de GDAL, así que se usa el sentinel real
                # (-10000, ver TIFFs WUR_transparent_reflectance_*) directamente.
                band = np.where(band <= -9999, 0, band).astype('float32')
                band_resized = cv2.resize(band, img_size)
                bands.append(band_resized)

            stacked = np.stack(bands, axis=-1)
            max_val = np.max(stacked)
            if max_val > 0:
                stacked = stacked / max_val
            return stacked.astype(np.float32)
            
        else:
            # Lógica RGB: 'path' es la carpeta, necesitamos encontrar el archivo rgb.tif
            target_file = None
            if os.path.isdir(path):
                # Buscamos el archivo que termina en rgb.tif dentro de la carpeta
                target_file = next((os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith("rgb.tif")), None)
            elif os.path.isfile(path):
                target_file = path

            if target_file is None:
                raise FileNotFoundError(f"No se encontró un archivo RGB válido en {path}")

            img = cv2.imread(target_file)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen RGB: {target_file}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, img_size).astype('float32') / 255.0

    except Exception as e:
        print_time_and_step('riop err 1', f"❌ Error procesando {path}: {e}", timestamp=timestamp, start_time=start_time)
        return None

def save_inference_results(results, output_dir, threshold, img_type, model_type):
    """Guarda la lista de resultados (una por muestra) como JSON en output_dir."""
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results_{img_type}_{model_type}_t{threshold}_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print_time_and_step('riop 5', f"📂 JSON guardado en: {path}", timestamp=timestamp, start_time=start_time)