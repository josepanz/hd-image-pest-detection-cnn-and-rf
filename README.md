# 🌿 Tesis Plagas con Imágenes HD: Detección en Cultivos de Papa

Este proyecto implementa y compara **tres escenarios** de clasificación para la detección de plagas/enfermedades en cultivos de papa a partir de imágenes de alta resolución (HD). El objetivo principal es evaluar cómo diferentes funciones de pérdida y arquitecturas manejan el alto desbalance de clases del dataset.

## 📂 Estructura del Proyecto

El código está organizado por el tipo de clasificador (CNN o RF) y la función de pérdida utilizada.

| Carpeta | Contenido | Descripción |
| :--- | :--- | :--- |
| `data/` | `Plaga/`, `Sana/` | Directorio principal de los datos de entrenamiento y validación. **Debe contener las imágenes.** |
| `data/Plaga` | Imágenes HD | Muestras de plantaciones de papas con plagas/enfermedades. |
| `data/Sana` | Imágenes HD | Muestras de plantaciones de papas sanas. |
| `src/` | | Código fuente principal. |
| `src/cnn/binary_crossentropy/` | | Modelo **Deep Learning (MobileNetV2)** con pérdida estándar. |
| `src/cnn/focal_loss/` | | Modelo **Deep Learning (MobileNetV2)** con pérdida **Focal Loss** (para desbalance). |
| `src/rf/` | | Modelo **Machine Learning Clásico (Random Forest)** usando CNN para extracción de *features*. |
| `prueba/` | Imágenes nuevas | Imágenes de prueba para los scripts de inferencia (`inference.py`). |
| `requirements.txt`| Dependencias | Lista de librerías necesarias. |

---

## ⚙️ Configuración y Requisitos

### 1. Crear y Activar el Entorno Virtual

Es fundamental usar un entorno virtual (`venv`) para evitar conflictos de librerías. Ejecuta estos comandos en la carpeta raíz del proyecto.

```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows
.\venv\Scripts\activate

# Activar en Linux/macOS
source venv/bin/activate
```

### 2. Instalar Dependencias

Asegúrate de tener un archivo `requirements.txt` que liste todas las librerías necesarias (TensorFlow, scikit-learn, etc.).

```bash
pip install -r requirements.txt
```

## 🚀 Guía de Ejecución Paso a Paso
### I. Escenario: Deep Learning con Binary Cross-Entropy (Línea Base)

Este modelo establece la referencia utilizando la función de pérdida estándar.

#### 1. Entrenamiento (`train.py`)

El script entrena el modelo CNN y guarda el mejor peso monitoreando el Recall o la Loss de validación (dependiendo de la configuración del callback).

```bash
python src/cnn/binary_crossentropy/train.py ./data
```

#### 2. Evaluación (`evaluate.py`)

Evalúa el modelo guardado. Es clave usar el argumento `-t` para probar la sensibilidad (umbral) de la clasificación binaria (por defecto es 0.5).

```bash
# Ejemplo de Evaluación Estándar (Umbral 0.5)
python src/cnn/binary_crossentropy/evaluate.py ./data -m src/cnn/binary_crossentropy/best_model.keras -t 0.5 -r report_bce_t050.json

# Ejemplo de Evaluación con Umbral Ajustado (0.75)
python src/cnn/binary_crossentropy/evaluate.py ./data -m src/cnn/binary_crossentropy/best_model.keras -t 0.75 -r report_bce_t075.json
```

#### 3. Inferencia (`inference.py`) Prueba

Prueba el modelo en imágenes de la carpeta `prueba/`.

```bash
python src/cnn/binary_crossentropy/inference.py ./prueba -m src/cnn/binary_crossentropy/best_model.keras -t 0.5
```

### II. Escenario: Deep Learning con Focal Loss (Recomendado para Desbalance)

Este modelo utiliza Focal Loss y un sampling avanzado en el dataloader para mitigar el sesgo por desbalance.

Este modelo es el enfoque principal para mejorar el rendimiento de la clase minoritaria ("Sana") mediante la pérdida focal y técnicas de sampling.

#### 1. Entrenamiento (`train.py`)

Utiliza el parámetro `-a` (`--alpha`) para configurar la pérdida focal (`-a 0.15` favorece más el enfoque en la clase Plaga).

```bash
# -e: 25 épocas, -a: Alpha de 0.50 para Focal Loss.
python src/cnn/focal_loss/train.py ./data -e 20 -a 0.50
```

#### 2. Evaluación (`evaluate.py`)
Evalúa el modelo con Focal Loss. Aquí es donde se recomienda probar diferentes umbrales si el Recall en la clase "Sana" es bajo.

Evalúa el modelo guardado (`best_model.keras`).

```bash
python src/cnn/focal_loss/evaluate.py ./data -m src/cnn/focal_loss/best_model.keras -t 0.5
```

#### 3. Inferencia (`inference.py`) Prueba

```bash
python src/cnn/focal_loss/inference.py ./prueba -m src/cnn/focal_loss/best_model.keras -t 0.5
```

### II. Escenario: Machine Learning Clásico (Random Forest)

Este enfoque usa la CNN (MobileNetV2) solo para extraer características y clasifica con Random Forest (entrena un clasificador no basado en gradiente (RF)).

#### 1. Extracción de Características y Entrenamiento (`train.py`)

El `train.py` en RF primero extrae características de todas las imágenes (proceso que puede ser lento) y luego entrena el clasificador RF, guardándolo como un archivo `.joblib.`

Este proceso es más lento porque primero extrae características de todas las imágenes. El modelo se guarda como `.joblib`.

```bash
# El modelo RF (.joblib) se guardará en src/rf/models/
python src/rf/train.py ./data
```

<b>⚠️IMPORTANTE:</b> Anota la ruta del archivo `.joblib` generado (ej: `src/rf/models/random_forest_20251103_0038.joblib`).

#### 2. Evaluación (`evaluate.py`)
Usa la ruta exacta del modelo `.joblib` para el argumento `-m`.

```bash
# REEMPLAZA <MODELO_RF.joblib> con tu ruta real.
python src/rf/evaluate.py ./data -m src/rf/models/random_forest_GUARDADO.joblib
```

#### 3. Inferencia (`inference.py`) Prueba

```bash
# REEMPLAZA <MODELO_RF.joblib> con tu ruta real.
python src/rf/inference.py ./prueba -m src/rf/models/random_forest_GUARDADO.joblib
```

## 🔬 Análisis y Comparativa de Resultados

### Enfoque de Métricas

Dada la naturaleza crítica de la detección de plagas y el alto desbalance de clases, la métrica más importante es el Recall de la clase "Sana" y el F1-Score. Un Recall bajo en "Sana" significa que el modelo está generando muchos Falsos Negativos (etiquetando muestras "Sana" como "Plaga", lo que provoca falsas alarmas), o lo que es peor, Falsos Positivos de Plaga (si se tiene Recall bajo en Plaga).

| Métrica | Importancia en este Proyecto |
| :--- | :--- |
| Recall (Plaga) | Crítico: Indica qué porcentaje de plagas reales se detectaron. Debe ser lo |más cercano a 1.0 posible. |
| Recall (Sana) | Moderado: Indica qué porcentaje de plantas sanas se clasificaron correctamente. |
| F1-Score | General: El mejor indicador para evaluar el rendimiento general, ya que balancea Precision y Recall. |

----

Tras ejecutar las evaluaciones de los tres escenarios, encontrarás los reportes de clasificación en formato JSON (y las matrices de confusión ploteadas) dentro de las carpetas de resultados de cada modelo (`src/cnn/.../results/` o `src/rf/results/`).

### Métricas Clave

La métrica más importante en este contexto, dado el desbalance y el costo de un Falso Negativo (no detectar una plaga), es el Recall de la clase "Sana" y el F1-Score de la clase minoritaria:

| Métrica | Enfoque | Interpretación para "Sana" |
| :--- | :--- | :--- |
|Recall | Deep Learning / RF | ¿Cuántas muestras de "Sana" se detectaron correctamente? |
|Precision | Deep Learning / RF | De todas las muestras predichas como "Sana", ¿cuántas eran realmente "Sana"? |
|F1-Score | Deep Learning / RF | Promedio armónico de Precision y Recall. El mejor indicador de rendimiento balanceado. |

<b>Objetivo:</b> El mejor modelo será aquel que logre un alto Recall para la clase Plaga (para no dejar ninguna plaga sin identificar) sin sacrificar demasiado el Recall de la clase Sana (para evitar la mayoría de falsas alarmas).