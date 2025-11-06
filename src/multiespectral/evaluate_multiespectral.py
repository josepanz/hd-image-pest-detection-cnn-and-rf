import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.models import load_model
from keras.utils import to_categorical
from datetime import datetime
import matplotlib.pyplot as plt
import json

# --- IMPORTACIONES CLAVE ---
# Debes tener estas funciones en archivos separados (data_loader.py, data_loader_multiespectral.py)
# Adaptar según la estructura de tu proyecto
try:
    #from data_loader import load_rgb_data 
    from data_loader_multiespectral import load_multiespectral_data 
    from model_multiespectral import focal_loss
    # Asegúrate de importar tu BinaryFocalLoss si tu modelo lo usa
    # from focal_loss import BinaryFocalLoss
except ImportError as e:
    print(f"Error de importación: {e}. Asegúrate de que los data_loaders están en el PATH.")
    sys.exit(1)

# --- CONFIGURACIÓN ---
CLASSES = ["Plaga", "Sana"] 
THRESHOLD_DEFAULT = 0.5 # Umbral por defecto para DL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'results_multispectral')

# --- FUNCIÓN DE PLOTEO (Matriz de Confusión) ---
def plot_confusion_matrix(cm, classes, title='Matriz de Confusión', cmap=plt.cm.viridis, output_dir=None):
    """Plotea la matriz de confusión y la guarda."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Añadir números al centro de los cuadrados
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Verdadera')
    plt.xlabel('Predicha')
    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        loss_name = title.split('(')[0].strip().replace(' ', '_')
        plot_path = os.path.join(output_dir, f"confusion_matrix_{loss_name}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        print(f"📈 Matriz de Confusión guardada en: {plot_path}")


# --- FUNCIÓN PRINCIPAL DE EVALUACIÓN ---
def main():
    parser = argparse.ArgumentParser(description="Evalúa un modelo CNN (RGB o Multiespectral)")
    parser.add_argument("-m", "--model", default="best_model_ms_base.keras", help="Ruta al archivo del modelo Keras (.keras)")
    parser.add_argument("-f", "--loss_function", choices=['focal_loss', 'binary_crossentropy'], default="binary_crossentropy", help="Función de pérdida")
    parser.add_argument("-t", "--threshold", type=float, default=THRESHOLD_DEFAULT, 
                        help="Umbral de clasificación (solo para modelos DL). Por defecto: 0.5.")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=[224, 224], 
                        help="Tamaño de la imagen (alto ancho).")
    args = parser.parse_args()

    # --- 1. CARGAR EL MODELO ---
    custom_objects = {}
    # Añadir tu clase de Focal Loss si es necesaria para cargar el modelo
    if 'focal_loss' in args.loss_function:
      custom_objects['focal_loss'] = focal_loss() 

    try:
        model = load_model(args.model, custom_objects=custom_objects)
        print(f"✅ Modelo cargado: {os.path.basename(args.model)}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

    # --- 2. DETERMINAR CANALES Y CARGAR DATOS DE PRUEBA ---
    try:
        input_shape = model.input_shape[1:]
        channels = input_shape[-1]
        img_size = input_shape[:2]
    except AttributeError:
        print("❌ Error: No se pudo obtener la forma de entrada del modelo.")
        sys.exit(1)

    X_test, Y_test = None, None

    print("Cargando datos Multiespectrales (5 canales)...")
    # El data_loader_multiespectral.py ya está configurado para devolver todos los parches (X, Y)
    X_data, Y_data = load_multiespectral_data(patch_size=img_size[0]) 
    # Aquí puedes implementar el split final si tu load_multiespectral_data no lo hace internamente
    # Para este ejemplo, usaremos un 80/20 simple si devuelve todos los datos
    _, X_test, _, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=42, stratify=Y_data
    )
        
    if X_test is None: X_test = X_data # Asumimos que si no hay split, se usa todo como prueba
    if Y_test is None: Y_test = Y_data
    
    Y_true_labels = Y_test # Etiquetas verdaderas como (0, 1)

    # 3. PREDICCIÓN Y CLASIFICACIÓN
    print(f"\nRealizando predicción en {X_test.shape[0]} muestras...")
    
    # Probabilidades de la Clase 1 (Sana)
    Y_pred_proba = model.predict(X_test, verbose=1).flatten() 
    
    # Aplicar Umbral (Clasificación)
    Y_pred_labels = (Y_pred_proba >= args.threshold).astype(int) 

    # 4. CÁLCULO DE MÉTRICAS
    # Precisión, Recall, F1-Score y Support por clase
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        Y_true_labels, Y_pred_labels, average=None, labels=[0, 1]
    )
    
    # 5. GENERACIÓN DE RESULTADOS
    # a) Matriz de Confusión
    cm = confusion_matrix(Y_true_labels, Y_pred_labels, labels=[0, 1])
    
    # b) Métricas globales y por clase
    results = {
        "model_name": os.path.basename(args.model),
        "input_channels": channels,
        "test_samples": len(Y_test),
        "threshold_used": args.threshold,
        "f1-score (Plaga)": float(f1_score[0]),
        "f1-score (Sana)": float(f1_score[1]),
        "f1-score (Weighted Avg)": float(np.average(f1_score, weights=np.bincount(Y_true_labels))),
        "precision (Plaga)": float(precision[0]),
        "precision (Sana)": float(precision[1]),
        "recall (Plaga)": float(recall[0]),
        "recall (Sana)": float(recall[1]),
        "confusion_matrix": cm.tolist()
    }

    # 6. GUARDADO Y PLOTEO
    
    # Directorio de resultados basado en el nombre del modelo
    output_dir_base = "evaluate_results"
    OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, output_dir_base)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardar JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_filename = f"evaluation_report_{timestamp}_c{channels}.json"
    report_path = os.path.join(OUTPUT_DIR, report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nReporte de Evaluación guardado en: {report_path}")
    print("\n--- Resultados de la Evaluación ---")
    print(f"Canales de Entrada: {channels}")
    print(f"F1-Score (Plaga): {results['f1-score (Plaga)']:.4f}")
    print(f"F1-Score (Sana): {results['f1-score (Sana)']:.4f}")
    print(f"F1-Score (Promedio Ponderado): {results['f1-score (Weighted Avg)']:.4f}")
    
    # Ploteo de la Matriz de Confusión
    title = f"Matriz de Confusión ({os.path.basename(args.model)} - {channels} Canales)"
    plot_confusion_matrix(cm, CLASSES, title=title, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()