import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt # Necesario para plot_inference_results
import os

CLASSES = ["Plaga", "Sana"] 

def print_time_and_step(step_number, message, timestamp=None, start_time=None):
    """Calcula y imprime el tiempo transcurrido desde el inicio.

    timestamp/start_time se calculan en cada llamada si no se pasan
    explícitamente (evita el bug de "default arg" evaluado una sola vez
    al importar el módulo, que congelaba el reloj al momento del import).
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if start_time is None:
        start_time = time.time()
    elapsed = time.time() - start_time
    print(f"\n--- [Fecha/hora inicio={timestamp}] ---")
    # Usar f-string para formatear el tiempo a segundos con 2 decimales
    print(f"\n--- [T={elapsed:.2f}s] ---")
    print(f"{step_number}. {message}")

def plot_inference_results(results: List[Dict[str, Any]], output_dir: str, timestamp: str, is_multiespectral: bool, model_type: str):
  """Crea y guarda un gráfico de confianza de predicción."""
  if not results:
      print("No hay resultados para plotear.")
      return

  file_names = [res['file_name'] for res in results]
  # Asumimos que la probabilidad de "Sana" está en 'prob_sana'
  prob_sana = np.array([res['prob_sana'] for res in results])
  predictions = [res['prediccion'] for res in results]
  
  # Crear la figura
  plt.figure(figsize=(15, 6))
  x_pos = np.arange(len(file_names))
  
  # Colores: Rojo para 'Plaga', Verde para 'Sana'
  colors = ['red' if pred == 'Plaga' else 'green' for pred in predictions]
  
  plt.bar(x_pos, prob_sana, color=colors)
  
  # Usar el umbral del primer resultado (asumiendo que es constante)
  umbral = results[0].get('umbral', 0.5) 
  plt.axhline(umbral, color='gray', linestyle='--', linewidth=1, label=f'Umbral ({umbral:.2f})')
  
  # Nombres de eje X acortados para mejor visualización
  short_names = [name[0:6] + "..." + name[-15:] if len(name) > 25 else name for name in file_names]
  
  plt.ylabel(f'Probabilidad de ser "{CLASSES[1]}"')
  model_name = "MULTIESPECTRAL" if is_multiespectral else "RGB"
  plt.title(f'Confianza de la Predicción {model_type} ({model_name}) (Umbral: {umbral:.2f})')
  plt.xticks(x_pos, short_names, rotation=45, ha='right')
  plt.ylim(0, 1)
  plt.legend()
  plt.tight_layout()
  
  plot_path = os.path.join(output_dir, f"inference_confidence_plot_{model_type}_{model_name}_{timestamp}.png")
  plt.savefig(plot_path)
  # Se muestra sin bloquear (show(block=False)+pause) y no se cierra: queda visible
  # en pantalla igual que los gráficos de train.py/evaluate.py (ver utils_train.py y
  # evaluation/utils_metrics.py), sin frenar una corrida de infer.py desatendida.
  plt.show(block=False)
  plt.pause(0.001)

  print(f"📈 Gráfico de confianza guardado en: {plot_path}")