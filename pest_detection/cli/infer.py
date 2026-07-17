"""Script CLI de inferencia vigente (reemplaza al viejo inference.py, ya eliminado).

Soporta CNN (.keras) y Random Forest (.joblib) sobre RGB o multiespectral, sobre una
imagen/carpeta de muestra individual o una carpeta que contenga varias. Para RF,
además del .joblib del RF necesita encontrar en el mismo best_models/ la CNN "hermana"
(mismo tipo RGB/MS, misma -l/--loss) para extraer las features; si no la encuentra,
avisa y sigue igual pasando los píxeles crudos aplanados (ver run_unified_inference).
"""

import argparse
import os
import time
from datetime import datetime

from pest_detection.api import PestDetector
from pest_detection.evaluation.inference_utils import save_inference_results
from pest_detection.print_utils import print_time_and_step
from pest_detection.print_utils import plot_inference_results

def main():
    parser = argparse.ArgumentParser(description="Inferencia unificada para modelos CNN y Random Forest (RGB/MS).")
    parser.add_argument("path", help="Ruta a imagen (RGB) o carpeta de muestra (MS).")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo (.keras o .joblib).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (t).")
    parser.add_argument("-mt", "--model_type", type=str, required=True, choices=["cnn", "rf"], help="Tipo de arquitectura.")
    parser.add_argument("-l", "--loss", type=str, required=False, choices=["fl", "bce"], help="Tipo de perdida.")
    parser.add_argument("-b", "--base_dir", default=os.getcwd(), help="Directorio donde se crea inference_results/ (por defecto, el directorio actual).")
    args = parser.parse_args()

    run_unified_inference(args.path, args.model, args.threshold, args.model_type, args.loss, args.base_dir)

def run_unified_inference(path, model_path, threshold, arch_type, loss: str = 'fl', base_dir: str = None):
    base_dir = base_dir if base_dir is not None else os.getcwd()
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Identificación automática de configuración
    model_path_lower = model_path.lower()
    is_ms = "multiespectral" in model_path_lower
    img_mode = "MULTIESPECTRAL" if is_ms else "RGB"
    # Mismos nombres que usa evaluate.py para sus propias carpetas de resultados
    # (CNN/RANDOM_FOREST, focal_loss/binary_crossentropy), para no tener dos
    # convenciones de nombres distintas entre evaluation_results/ e inference_results/.
    arch_folder = "CNN" if arch_type == "cnn" else "RANDOM_FOREST"

    if arch_type == "rf":
        loss_type = "focal_loss" if "fl" in loss else "binary_crossentropy"
    else:
        loss_type = "focal_loss" if "focal" in model_path_lower else "binary_crossentropy"

    print_time_and_step('init', f'🚀 Modo: {img_mode} | Arq: {arch_folder} | Config: {loss_type}', timestamp=timestamp, start_time=start_time)

    # 2. Carga del Modelo (y, para RF, de su CNN "hermana" extractora de features)
    print_time_and_step('1', f"⏳ Cargando modelo: {os.path.basename(model_path)}", timestamp=timestamp, start_time=start_time)
    detector = PestDetector(model_path, model_type=arch_type, is_multiespectral=is_ms, loss=loss)

    # 3. Ejecución de Inferencia
    print_time_and_step('2', "🔎 Procesando imágenes y realizando predicción...", timestamp=timestamp, start_time=start_time)

    try:
        results = detector.predict(path, threshold=threshold)
    except Exception as e:
        print_time_and_step('ERROR', f"Fallo crítico en inferencia: {e}", timestamp=timestamp, start_time=start_time)
        return

    # 4. Post-procesamiento y Guardado
    if not results:
        print_time_and_step('WARN', "No se encontraron resultados válidos en la ruta proporcionada.", timestamp=timestamp, start_time=start_time)
        return

    print_time_and_step('3', f"✅ Inferencia completada. Procesados: {len(results)} items.", timestamp=timestamp, start_time=start_time)

    # Crear directorios de salida
    output_base = os.path.join(base_dir, f'inference_results/{arch_folder}/{loss_type}/{img_mode}/{threshold}')
    os.makedirs(output_base, exist_ok=True)

    # Guardar JSON y Gráfico
    save_inference_results(results, output_base, threshold, img_mode, loss_type)
    plot_inference_results(results, output_base, timestamp, is_ms, arch_folder)
    
    print_time_and_step('END', f"✨ Proceso finalizado. Resultados en: {output_base}", timestamp=timestamp, start_time=start_time)

if __name__ == "__main__":
    main()