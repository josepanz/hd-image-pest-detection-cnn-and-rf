import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def plot_thesis_evolution_cm(data_dict, model_name):
    """
    Genera la comparativa de matrices de confusión para la tesis.
    """
    # Configuramos el estilo visual
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Etiquetas basadas en tu proyecto
    class_names = ['Plaga', 'Sana'] 
    thresholds = sorted(data_dict.keys())

    for i, t in enumerate(thresholds):
        # Convertimos la lista en array de numpy
        cm = np.array(data_dict[t])
        
        # Creamos el mapa de calor
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 16, "weight": "bold"}, cbar=False,
                    linewidths=1, linecolor='black')
        
        # Títulos y etiquetas
        axes[i].set_title(f'Umbral t = {t}', fontsize=15, pad=10)
        axes[i].set_xlabel('Predicción del Modelo', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Clase Real (Ground Truth)', fontsize=12)
        else:
            axes[i].set_ylabel('')

    # Título principal de la tesis
    plt.suptitle(f'Evolución de la matriz de confusión según umbral\n({model_name})', 
      fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    # Guardar en alta resolución para el documento final
    output_name = f"figura_14_evolucion_{model_name.replace(' ', '_')}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Imagen guardada como: {output_name}")
    plt.show()

# --- DATOS EXTRAÍDOS DE TUS LOGS (CNN RGB Binary Crossentropy) ---
# Formato: [[TP, FN], [FP, TN]] según el orden de tus clases
data_logs = {
    "0.45": [[42, 41], [11, 3]],
    "0.50": [[54, 29], [13, 1]],
    "0.70": [[80, 3], [14, 0]]
}

import matplotlib.pyplot as plt

# --- DATOS EXTRAÍDOS DE TUS LOGS (Modelo CNN RGB BCE) ---
# Umbrales procesados
thresholds = [0.45, 0.50, 0.70]

# Mapeo de tus matrices [[TP, FN], [FP, TN]] 
# (Asumiendo Plaga como clase positiva según el comportamiento del umbral en tus logs)
tp = [42, 54, 80]  # Aciertos Plaga
fn = [41, 29, 3]   # Plaga que el modelo no vio
fp = [11, 13, 14]  # Sana que el modelo marcó como Plaga
tn = [3, 1, 0]     # Aciertos Sana

def plot_metrics_evolution():
    plt.figure(figsize=(10, 6))
    
    # Dibujar las líneas de tendencia
    plt.plot(thresholds, tp, marker='o', label='Verdaderos Positivos (Plaga)', color='green', linewidth=2)
    plt.plot(thresholds, tn, marker='s', label='Verdaderos Negativos (Sana)', color='blue', linewidth=2)
    plt.plot(thresholds, fp, marker='^', label='Falsos Positivos (Error Plaga)', color='orange', linestyle='--')
    plt.plot(thresholds, fn, marker='v', label='Falsos Negativos (Error Sana)', color='red', linestyle='--')

    # Configuración de ejes y estética
    plt.title('Evolución de Clasificación según Umbral de Decisión', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Umbral de Clasificación ($t$)', fontsize=12)
    plt.ylabel('Cantidad de Muestras (n=97)', fontsize=12)
    plt.xticks(thresholds)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Añadir anotaciones para resaltar el comportamiento
    plt.annotate('Mayor sensibilidad a Plaga', xy=(0.70, 80), xytext=(0.55, 85),
      arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.tight_layout()
    
    # Guardar para la tesis
    plt.savefig('evolucion_metricas_tesis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_conteo_umbrales():
    plt.figure(figsize=(11, 6))
    
    # Graficar cada línea con su marcador y valor
    metriz_list = [
        (tp, 'Verdaderos Positivos (TP)', '#2ecc71', 'o'),
        (tn, 'Verdaderos Negativos (TN)', '#3498db', 's'),
        (fp, 'Falsos Positivos (FP)', '#f1c40f', '^'),
        (fn, 'Falsos Negativos (FN)', '#e74c3c', 'v')
    ]

    for data, label, color, marker in metriz_list:
        plt.plot(thresholds, data, label=label, color=color, marker=marker, linewidth=2.5, markersize=8)
        # Añadir las etiquetas de valor sobre cada punto
        for x, y in zip(thresholds, data):
            plt.text(x, y + 1.5, str(y), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Configuración de estética académica
    plt.title('Evolución de la matriz de confusión según umbral.', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Umbral de Decisión ($t$)', fontsize=12)
    plt.ylabel('Cantidad de Casos (Conteo)', fontsize=12)
    
    plt.xticks(thresholds)
    plt.ylim(-5, 95) # Ajustado al total de tu muestra
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Guardar para la tesis
    plt.savefig('C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\evolucion_conteo.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stacked_evolution():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Definir colores académicos
    colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b'] # Verdes y Rojos
    
    # Crear las barras apiladas
    # Las barras se construyen una sobre otra usando el parámetro 'bottom'
    ax.bar(thresholds, tp, label='Verdaderos Positivos (TP)', color='#2ecc71')
    ax.bar(thresholds, tn, bottom=tp, label='Verdaderos Negativos (TN)', color='#a9dfbf')
    ax.bar(thresholds, fp, bottom=tp+tn, label='Falsos Positivos (FP)', color='#f5b7b1')
    ax.bar(thresholds, fn, bottom=tp+tn+fp, label='Falsos Negativos (FN)', color='#e74c3c')

    # Añadir los números dentro de las barras para mayor claridad
    for i in range(len(thresholds)):
        # Texto para TP
        ax.text(i, tp[i]/2, str(tp[i]), ha='center', color='black', fontweight='bold')
        # Texto para TN (si es > 0)
        if tn[i] > 0:
            ax.text(i, tp[i] + tn[i]/2, str(tn[i]), ha='center', color='black', fontweight='bold')
        # Texto para FP
        ax.text(i, tp[i] + tn[i] + fp[i]/2, str(fp[i]), ha='center', color='black', fontweight='bold')
        # Texto para FN
        ax.text(i, tp[i] + tn[i] + fp[i] + fn[i]/2, str(fn[i]), ha='center', color='white', fontweight='bold')

    # Configuración de estética de tesis
    ax.set_title('Evolución de la Composición de la Matriz de Confusión\n(Proyección por Umbrales)', 
              fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Total de Muestras (n=97)', fontsize=12)
    ax.set_xlabel('Umbral de Decisión ($t$)', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Categorías")
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Guardar para el documento final
    plt.savefig('proyeccion_evolucion_umbrales.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
  parser = argparse.ArgumentParser(description="Evalúa el modelo CNN con BCE/Focal RGB/MS")
  parser.add_argument("-e", "--evolution", type=str, required=True, choices=["cm", "metrics", "stacked", "conteo"], default="metrics", help="Tipo de evolución a plotear (cm o metrics)")
  args = parser.parse_args()
  if args.evolution == "cm":
    plot_thesis_evolution_cm(data_logs, "CNN RGB Binary Crossentropy")
  elif args.evolution == "metrics":
    plot_metrics_evolution()
  elif args.evolution == "stacked":
    plot_stacked_evolution()
  elif args.evolution == "conteo":
    plot_conteo_umbrales()

if __name__ == "__main__":
  main()
  