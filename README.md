# 🌿 Detección de Plagas en Cultivos de Papa con Imágenes HD y Multiespectrales (CNN y Random Forest)

Este proyecto implementa y compara distintos enfoques de Inteligencia
Artificial para detectar plagas en cultivos de papa utilizando imágenes
RGB y multiespectrales capturadas por drones.

El objetivo es analizar cómo diferentes funciones de pérdida, umbrales
de decisión y modelos de clasificación se comportan frente al desbalance
de clases presente en el dataset.

**Modelos evaluados:**

-   CNN basada en MobileNetV2
-   CNN utilizando Focal Loss
-   Random Forest utilizando características extraídas por CNN

**Repositorio:**

- https://github.com/josepanz/hd-image-pest-detection-cnn-and-rf

------------------------------------------------------------------------

## 📂 Estructura del Proyecto

El código está organizado por el tipo de clasificador (CNN o RF) y la función de pérdida utilizada.

| Carpeta | Contenido / Descripción |
| :--- | :--- |
| data | Contiene los datasets utilizados para entrenamiento |
| predict-test/ | Imágenes utilizadas para pruebas de inferencia |
| src/pest_detection_models/ | Código principal del proyecto
| src/pest_detection_models/train.py | Script para entrenamiento de modelos |
| src/pest_detection_models/evaluate.py | Script para evaluación de modelos |
| src/pest_detection_models/inference.py | Script para realizar predicciones |
| src/pest_detection_models/inference_random_forest.py | Inferencia específica para Random Forest |
| src/pest_detection_models/best_models/ | Modelos entrenados guardados |
| src/pest_detection_models/evaluation_results/ | Resultados de evaluación |
| requirements.txt | Dependencias del proyecto |
| README.md | Documentación del repositorio |

## Estructura simplificada:
```bash
    .
    ├── data/
    │   └── multiespectral/
    │       └── TTADDA-dataset/
    │           └── TTADDA_NARO_2023_F1/
    │               └── drone_data/
    │
    ├── predict-test/
    │   └── multiespectral/
    │
    ├── src/
    │   └── pest_detection_models/
    │       ├── train.py
    │       ├── evaluate.py
    │       ├── inference.py
    │       ├── inference_random_forest.py
    │       ├── best_models/
    │       └── evaluation_results/
    │
    ├── requirements.txt
    └── README.md
```


## ⚙️ Configuración del Entorno

### 1️⃣ Crear entorno virtual

Es fundamental usar un entorno virtual (`venv`) para evitar conflictos de librerías. Ejecuta estos comandos en la carpeta raíz del proyecto.

```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows
.\venv\Scripts\activate

# Activar en Linux/macOS
source venv/bin/activate
```

------------------------------------------------------------------------

### 2️⃣ Instalar dependencias

Asegúrate de tener un archivo `requirements.txt` que liste todas las librerías necesarias (TensorFlow, etc.).

```bash
pip install -r requirements.txt
```

### 3️⃣ Descargar archivos para pruebas 
**<u>Articulo en linea:</u>** <span><a href="https://data.4tu.nl/datasets/c5f013d0-85e0-4feb-b653-a3c59683a2bc">TTADDA_NARO_2023: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset</a> (TTADDA_NARO_2023: Un subconjunto del conjunto de datos de patatas TTADDA-UAV RGB y multiespectral de varias temporadas)</span>

**Descarga Directa:** <a href="https://data.4tu.nl/file/c5f013d0-85e0-4feb-b653-a3c59683a2bc/1baf67c0-9522-4099-b058-72ed0084c1a4">TTADDA_NARO_2023.zip</a>

**Otros archivos:** 
- <a href="https://data.4tu.nl/file/c5f013d0-85e0-4feb-b653-a3c59683a2bc/51b2b3c1-4f4c-4223-8697-aed0d93f5d4d">MIAPPE_Minimal_Spreadsheet_Template_TTADDAv4.xlsx</a>
- <a href="https://data.4tu.nl/file/c5f013d0-85e0-4feb-b653-a3c59683a2bc/24478a60-1479-43af-80d3-5cd017fdc6bc">README_TTADDA_NARO_2023.txt</a>

#### Base del estudio y datos adicionales.
- <span>**Estudio:** <a href="https://www.sciencedirect.com/science/article/pii/S2352340925007280">TTADDA-UAV: A Multi-Season RGB and Multispectral UAV Dataset of Potato Fields Collected in Japan and the Netherlands</a>. (TTADDA-UAV: Un conjunto de datos UAV RGB y multiespectrales multitemporales de campos de patatas recopilados en Japón y los Países Bajos.) <a href="https://data.4tu.nl/collections/936b5772-09fc-4856-983d-1f9cc2f38d15">DATOS</a>
</span>

  - **TTADDA_NARO_2022:** <a href="https://data.4tu.nl/datasets/ed9b9cd6-8d69-411b-9054-1ecce543ac1b">TTADDA_NARO_2022: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset</a> (TTADDA_NARO_2022: Un subconjunto del conjunto de datos de patatas TTADDA-UAV RGB y multiespectral de varias temporadas)

  - **TTADDA_NARO_2021:** <a href="https://data.4tu.nl/datasets/f2307c47-9a1a-474a-a0d9-e09ee1b7512c">TTADDA_NARO_2021: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset</a> (TTADDA_NARO_2021: Un subconjunto del conjunto de datos de patatas TTADDA-UAV RGB y multiespectral de varias temporadas)
  
  - **TTADDA_WUR_2022:** <a href="https://data.4tu.nl/datasets/1f628b56-3246-4aab-accd-1193b1566763">TTADDA_WUR_2022: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset</a> (TTADDA_WUR_2022: Un subconjunto del conjunto de datos de patatas TTADDA-UAV RGB y multiespectral de varias temporadas)

  - **TTADDA_WUR_2023:** <a href="https://data.4tu.nl/datasets/75c01fac-f00a-4980-8cd8-cd4499f1aa98">TTADDA_WUR_2023: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset
</a> (TTADDA_WUR_2023: Un subconjunto del conjunto de datos de patatas TTADDA-UAV RGB y multiespectral de varias temporadas)


#### Ubicación dentro del proyecto:
- `data/multiespectral/TTADDA-dataset/TTADDA_NARO_2023_F1/drone_data`

##### El dataset contiene:

-   imágenes RGB
-   imágenes multiespectrales
-   mediciones y metadatos agronómicos

## 🚀 Guía de Ejecución Paso a Paso
### 1️⃣ 🧠 Entrenamiento de Modelos

#### **- CNN Multiespectral — Focal Loss**
```bash
python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -t 0.45

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -t 0.50

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -t 0.70
```

#### **- CNN Multiespectral — Binary Crossentropy**
```bash
python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -t 0.45

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -t 0.50

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -t 0.70
```

#### **- CNN RGB — Focal Loss**
```bash
python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -rgb -t 0.45

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -rgb -t 0.50

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -rgb -t 0.70
```

#### **- CNN RGB — Binary Crossentropy**
```bash
python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -rgb -t 0.45

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -rgb -t 0.50

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -rgb -t 0.70
```

#### **- Random Forest**
```bash
python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf

python src\pest_detection_models\train.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf -rgb
```
------------------------------------------------------------------------

### 2️⃣ 📊 Evaluación de Modelos

#### Los resultados de evaluación se guardan en:
- `src/pest_detection_models/evaluation_results/`

#### Ejemplo de evaluación:
```bash
python src\pest_detection_models\evaluate.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m src\pest_detection_models\best_models\best_model_val_loss_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
```

#### Evaluación Random Forest:
```bash
python src\pest_detection_models\evaluate.py data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m src\pest_detection_models\best_models\best_model_random_forest_20260102_0042_MULTIESPECTRAL.joblib -mt rf
```

### 3️⃣ 🔎 Inferencia (Predicción)

Permite predecir si una imagen corresponde a plaga o planta sana.

#### Ejemplo CNN:
```bash
python src\pest_detection_models\inference.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m src\pest_detection_models\best_models\best_model_val_loss_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
```

#### Ejemplo RGB:
```bash
python src\pest_detection_models\inference.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m src\pest_detection_models\best_models\best_model_val_loss_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
```

#### Random Forest:
```bash
python src\pest_detection_models\inference.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m src\pest_detection_models\best_models\best_model_random_forest_20260102_0042_MULTIESPECTRAL.joblib -mt rf
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

## Definiciones: ¿Cuáles son los positivos y negativos? `Positivos = Sanos`, `Negativos = Plagas`
<a href= "https://www.youtube.com/watch?v=H8FSfqxRWmA">YouTube</a> | <a href="https://codificandobits.com/blog/precision-recall-f-score/">Blog</a> | <a href="https://colab.research.google.com/drive/10xngRuU0kyxGildcx7YfxQzjrk_nXXU-?usp=sharing">Colab</a>
- **Verdaderos Positivos / True Positive (VP o TP):** `"sanos"` clasificados **_realmente_** como `"sanos"`.
- **Falsos Positivos / False Positive (FP o FP):** `"plagas"` clasificados **_equivocadamente_** como `"sanos"`.
- **Verdaderos Negativos / True Negative (VN o TN):** `"plagas"` clasificados **_realmente_** como `"plagas"`.
- **Falsos Negativos / False Negative (FN o FN):** `"sanos"` clasificados **_equivocadamente_** como `"plagas"`.


<b>Objetivo:</b> El mejor modelo será aquel que logre un alto Recall para la clase Plaga (para no dejar ninguna plaga sin identificar) sin sacrificar demasiado el Recall de la clase Sana (para evitar la mayoría de falsas alarmas).