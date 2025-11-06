import argparse
import numpy as np
import tensorflow as tf
import os
import sys
from keras.preprocessing.image import load_img, img_to_array

from model import focal_loss # Importa focal_loss para cargar el modelo

import json
from datetime import datetime
import matplotlib.pyplot as plt

# Lista de nombres de clases; debe coincidir con el orden del entrenamiento.
# Clase 0: Plaga, Clase 1: Sana
CLASSES = ["Plaga", "Sana"]

def predict_single_image(
    model_obj, # Ahora pasamos el objeto del modelo ya cargado
    img_path: str,
    img_size: tuple[int, int] = (224, 224),
    threshold: float = 0.5
) -> tuple[str, np.ndarray]:
    """
    Carga y procesa una imagen, luego usa el modelo provisto para clasificarla.

    Args:
        model_obj: El objeto del modelo Keras ya cargado.
        img_path:  Ruta a la imagen a clasificar.
        img_size:  Tamaño (alto, ancho) para redimensionar la imagen.
        threshold: Umbral de decisión para clasificador binario (0-1).

    Returns:
        etiqueta:    Clase predicha (“Plaga” o “Sana”).
        probs_array: Array de probabilidades [prob_clase_0, prob_clase_1].
    """
    if not os.path.exists(img_path):
        print(f"Error: La imagen no se encontró en {img_path}")
        return "Error", np.array([0.0, 0.0])

    # Carga y procesa la imagen
    img = load_img(img_path, target_size=img_size)
    x = img_to_array(img) / 255.0         # Normaliza a [0,1]
    x = np.expand_dims(x, axis=0)         # Añade dimensión batch

    # Predicción: El modelo Dense(1, activation="sigmoid") devuelve una probabilidad única para la clase 1 (Sana)
    prob_clase_1 = model_obj.predict(x, verbose=0)[0][0] # verbose=0 para no imprimir cada barra de progreso de predicción

    # Lógica de decisión basada en el umbral
    if prob_clase_1 >= threshold: # Usa >= para incluir el umbral en la clase Sana
        idx = 1 # Corresponde a la clase 'Sana'
    else:
        idx = 0 # Corresponde a la clase 'Plaga'
    
    etiqueta = CLASSES[idx]

    # Para 'probs_array', queremos [prob_clase_0, prob_clase_1]
    probs_array = np.array([1 - prob_clase_1, prob_clase_1])

    return etiqueta, probs_array

def plot_inference_results(results: list, output_dir: str, timestamp: str = datetime.now().strftime("%Y%m%d_%H%M"), threshold: float = 0.65):
    """
    Crea un gráfico de confianza de predicción y lo guarda.
    """
    file_names = [res['file_name'] for res in results]
    prob_sana = np.array([res['prob_sana'] for res in results])
    predictions = [res['prediccion'] for res in results]
    
    # Asignar color basado en la predicción final
    colors = ['red' if pred == 'Plaga' else 'green' for pred in predictions]
    
    #plt.figure(figsize=(20, 8)) # Usar el nuevo tamaño ancho
    plt.figure(figsize=(15, 6))
    
    # Crear un índice numérico para las imágenes
    x_pos = np.arange(len(file_names))
    
    # Dibujar barras o puntos (Barras son mejores si hay pocas imágenes)
    plt.bar(x_pos, prob_sana, color=colors)
    
    # Dibujar la línea de umbral (referencia de decisión)
    umbral = results[0]['umbral'] 
    plt.axhline(umbral, color='gray', linestyle='--', linewidth=1, label=f'Umbral ({umbral:.2f})')

    # Usar los nombres acortados en el eje X:
    short_names = [name[0:6] + "..." + name[-15:] for name in file_names]
    
    plt.ylabel('Probabilidad de ser "Sana"')
    plt.title(f'Confianza de la Predicción por Imagen (Umbral: {umbral:.2f})')
    #plt.xticks(x_pos, file_names, rotation=90) # Rotar nombres para que quepan
    plt.xticks(x_pos, short_names, rotation=45, ha='right')
    plt.ylim(0, 1) # Escala de probabilidad de 0 a 1
    
    plt.legend()
    plt.tight_layout() # Ajusta automáticamente los parámetros de la subtrama
    
    # Guardar el gráfico
    plot_path = os.path.join(output_dir, f"inference_confidence_plot_{timestamp}_t{threshold}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f"📈 Gráfico de confianza guardado en: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Clasifica una imagen de hoja o todas las imágenes en una carpeta como Sana o Plaga"
    )
    parser.add_argument(
        "path", # Cambiado de "model" e "image" a un solo "path"
        help="Ruta al modelo .keras entrenado (si se usa --model) O ruta a la imagen/carpeta a clasificar."
    )
    parser.add_argument(
        "--model",
        type=str,
        default='best_model.keras', # Nombre por defecto del modelo
        help="Ruta al archivo del modelo .keras (por defecto: best_model.keras en la raíz del proyecto)."
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=(224, 224),
        metavar=('alto', 'ancho'),
        help="Tamaño para redimensionar la imagen (por defecto: 224 224)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        #default=0.75,
        default=0.65, # Con el umbral mas optimo para F1-Score por defecto
        help="Umbral de decisión para focal loss (por defecto: 0.65 Con el umbral mas optimo para F1-Score por defecto)"
    )
    args = parser.parse_args()

    # Construye la ruta completa al modelo (asumiendo que best_model.keras está en la raíz del proyecto)
    model_full_path = os.path.join(os.path.dirname(__file__), args.model)

    if not os.path.exists(model_full_path):
        print(f"Error: El modelo '{args.model}' no se encontró en '{model_full_path}'.")
        print("Asegúrate de que el modelo esté en la raíz de tu proyecto o ajusta la ruta.")
        sys.exit(1)

    # Cargar el modelo una sola vez para eficiencia
    try:
        model = tf.keras.models.load_model(model_full_path, custom_objects={'loss_fn': focal_loss()})
        print(f"Modelo '{args.model}' cargado exitosamente de '{model_full_path}'.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)
        
    # 1. Inicializar la lista de resultados
    inference_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Determinar si la 'path' proporcionada es un archivo o una carpeta
    if os.path.isfile(args.path):
        # Es un archivo individual
        etiqueta, probs = predict_single_image(model, args.path, tuple(args.size), args.threshold)

        prob_plaga_np = probs[0] # probs[0] es la probabilidad de Plaga (Clase 0)
        prob_sana_np = probs[1]  # probs[1] es la probabilidad de Sana (Clase 1)
        
        # Convertir a float nativo de Python para serialización JSON
        prob_plaga = float(prob_plaga_np)
        prob_sana = float(prob_sana_np)
        # 2. Guardar el resultado en la lista
        result = {
            "file_name": os.path.basename(full_image_path),
            "prob_sana": round(prob_sana, 4),  # Probabilidad de la clase 1 (Sana)
            "prob_plaga": round(prob_plaga, 4),  # Probabilidad de la clase 1 (Sana)
            "prediccion": etiqueta,
            "umbral": args.threshold,
            "modelo": os.path.basename(args.model) 
        }
        inference_results.append(result)
        
        print(f"\n--- Resultados para: {os.path.basename(args.path)} ---")
        print(f"Probabilidad de ser '{CLASSES[1]}': {probs[1]:.4f}") # Mostrar prob de Sana
        print(f"Predicción final (con umbral {args.threshold:.2f}): {etiqueta}")
    elif os.path.isdir(args.path):
        # Es una carpeta
        print(f"\nProcesando todas las imágenes en la carpeta: {args.path}")
        image_files = [f for f in os.listdir(args.path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"No se encontraron imágenes en la carpeta: {args.path}")
            sys.exit(0)
        
        # Ordenar los archivos para una salida consistente
        for image_file in sorted(image_files):
            full_image_path = os.path.join(args.path, image_file)
            etiqueta, probs = predict_single_image(model, full_image_path, tuple(args.size), args.threshold)

            prob_plaga_np = probs[0] # probs[0] es la probabilidad de Plaga (Clase 0)
            prob_sana_np = probs[1]  # probs[1] es la probabilidad de Sana (Clase 1)
            
            # Convertir a float nativo de Python para serialización JSON
            prob_plaga = float(prob_plaga_np)
            prob_sana = float(prob_sana_np)
            # 2. Guardar el resultado en la lista
            result = {
                "file_name": os.path.basename(full_image_path),
                "prob_sana": round(prob_sana, 4),  # Probabilidad de la clase 1 (Sana)
                "prob_plaga": round(prob_plaga, 4),  # Probabilidad de la clase 1 (Sana)
                "prediccion": etiqueta,
                "umbral": args.threshold,
                "modelo": os.path.basename(args.model) 
            }
            inference_results.append(result)

            print(f"\n--- Resultados para: {os.path.basename(full_image_path)} ---")
            print(f"Probabilidad de ser '{CLASSES[1]}': {probs[1]:.4f}") # Mostrar prob de Sana
            print(f"Predicción final (con umbral {args.threshold:.2f}): {etiqueta}")
    else:
        print(f"Error: La ruta '{args.path}' no es un archivo ni una carpeta válidos.")
        parser.print_help()
        return "Error"

    # 3. Guardar el archivo JSON
    output_filename = f"inference_results_{timestamp}_t{args.threshold}.json"
    
    # Asegúrate de definir un directorio para guardar los resultados
    # Usaremos una carpeta 'inference_results' dentro del directorio del script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'inference-results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    final_json_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(final_json_path, 'w') as f:
        json.dump(inference_results, f, indent=4)
        
    print(f"\n\n✅ Resultados de la inferencia guardados en: {final_json_path}")

    plot_inference_results(inference_results, OUTPUT_DIR, timestamp, args.threshold)

if __name__ == "__main__":
    main()