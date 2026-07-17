# 🌿 Detección de Plagas en Cultivos de Papa con Imágenes HD y Multiespectrales (CNN y Random Forest)

Este proyecto implementa y compara distintos enfoques de Inteligencia
Artificial para detectar plagas en cultivos de papa utilizando imágenes
RGB y multiespectrales capturadas por drones.

El objetivo es analizar cómo diferentes funciones de pérdida, umbrales
de decisión y modelos de clasificación se comportan frente al desbalance
de clases presente en el dataset.

**Modelos evaluados:**

-   CNN propia (arquitectura simple, no basada en un modelo pre-entrenado): al comienzo del proyecto se probó con MobileNetV2, pero se abandonó por ser costosa de reentrenar y porque su entrada esperaba 3 canales fijos — no aceptaba directamente las 5 bandas del modo multiespectral. Se reemplazó por una CNN propia (3 bloques Conv+BatchNorm+MaxPooling con 32/64/128 filtros, GlobalAveragePooling, Dense+Dropout, salida sigmoide — ver `pest_detection/models/cnn_model.py`) que acepta cualquier número de canales de entrada (3 para RGB, 5 para multiespectral) sin modificar la arquitectura.
-   CNN utilizando Focal Loss
-   Random Forest utilizando características extraídas por CNN

**Repositorio:**

- https://github.com/josepanz/hd-image-pest-detection-cnn-and-rf

------------------------------------------------------------------------

## 📂 Estructura del Proyecto

El código está organizado por el tipo de clasificador (CNN o RF) y la función de pérdida utilizada.

| Carpeta | Contenido / Descripción |
| :--- | :--- |
| data/ | Datasets de entrenamiento. No incluye los `.tif` reales del dataset TTADDA (hay que descargarlos aparte, ver más abajo); tampoco incluye ya el dataset RGB tipo PlantVillage que existió acá anteriormente (`data/rgb/Plaga`, `data/rgb/Sana`) - era un enfoque descartado y no estaba conectado al pipeline de `train.py` |
| predict-test/ | Imágenes/metadatos de ejemplo para pruebas de inferencia (tampoco incluye los `.tif` reales) |
| pest_detection/ | Código principal del proyecto, organizado como paquete Python |
| pest_detection/cli/train.py | Script para entrenamiento de modelos (CNN o Random Forest) |
| pest_detection/cli/evaluate.py | Script para evaluación de modelos ya entrenados |
| pest_detection/cli/infer.py | Script para realizar predicciones (CNN y Random Forest, RGB y multiespectral) |
| pest_detection/models/, pest_detection/datasets/, pest_detection/evaluation/ | Arquitectura de modelos, carga/preparación de datos, y métricas/motor de inferencia |
| pest_detection/tools/ | Scripts CLI standalone de preparación/inspección de datos, ver sección dedicada más abajo |
| best_models/ | Modelos entrenados guardados |
| evaluation_results/, history/, inference_results/ | Resultados de evaluación, historiales de entrenamiento e inferencias |
| tests/ | Suite de tests (unitarios en `tests/`, de integración en `tests/integration/`) |
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
    ├── pest_detection/
    │   ├── cli/
    │   │   ├── train.py
    │   │   ├── evaluate.py
    │   │   └── infer.py
    │   ├── models/
    │   ├── datasets/
    │   ├── evaluation/
    │   └── tools/
    │
    ├── best_models/
    ├── evaluation_results/
    ├── tests/
    │
    ├── requirements.txt
    └── README.md
```

`pest_detection` es un paquete Python instalable (`pip install -e .`, ver más abajo),
que agrega tres comandos (`pest-train`, `pest-evaluate`, `pest-infer`) y se puede
`import`ar directamente. Sin instalarlo, los scripts también se pueden correr con
`python -m pest_detection.cli.train` (o `.evaluate`/`.infer`) desde la raíz del repo
(Python encuentra el paquete sin hacks de `sys.path`, pero sin los comandos `pest-*`).

## 🧰 Herramientas auxiliares

Estos scripts se ejecutan manualmente para preparar o inspeccionar datos. Ninguno es
importado por `train.py`/`evaluate.py`/`infer.py` (no forman parte del pipeline
principal, son de uso puntual), y se corren igual con `python -m`:

| Script | Para qué sirve |
| :--- | :--- |
| `python -m pest_detection.tools.inspect_tif` | Inspecciona bandas, dimensiones y CRS de un GeoTIFF |
| `python -m pest_detection.tools.inspect_shapefile` | Inspecciona columnas/CRS/registros de un shapefile de parcelas |
| `python -m pest_detection.tools.inspect_datatables` | Explora un Excel de mediciones agronómicas |
| `python -m pest_detection.tools.patch_extractor` | Extrae parches de imagen por polígono desde un TIFF multiespectral |
| `python -m pest_detection.tools.save_rgb_and_bands` | Guarda cada banda + un compuesto RGB como PNG, para QA visual del dataset |
| `python -m pest_detection.tools.utils_tiff_converter` | Convierte TIFF multiespectral a PNG RGB normalizado, para inspección visual. **No hace falta correrlo para entrenar** (`train.py -rgb` lee el `*_RGB.tif` directo con rasterio, nunca un PNG) — pese a lo que dice su propio docstring, y a diferencia de lo que decía antes esta tabla. Además asume rutas (`data/multispectral_images/...`) y orden de bandas (B,G,R) que no coinciden con el dataset TTADDA real (`data/multiespectral/...`, TIFF ya en orden R,G,B) — revisarlo/corregirlo antes de usarlo si hace falta. |
| `python -m pest_detection.tools.convert_md_to_pdf` | Convierte bitácoras Markdown a PDF (requiere `pandoc`/`xelatex` instalados aparte) |
| `python -m pest_detection.evaluation.utils_matriz` | Genera figuras de evolución de matriz de confusión para el informe |
| `python -m pest_detection.evaluation.utils_roc` | Combina curvas ROC de corridas ya guardadas en un gráfico comparativo |

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

### 2️⃣.1 Instalar el paquete (opcional pero recomendado)

Instala `pest_detection` en modo editable: habilita los comandos `pest-train`/
`pest-evaluate`/`pest-infer` y `import pest_detection` desde cualquier otro proyecto
en el mismo entorno virtual.

```bash
pip install -e .
```

Uso como librería (ver `pest_detection/api.py`):
```python
from pest_detection import PestDetector

# Carga el modelo una sola vez; predict() se puede llamar muchas veces sin recargarlo.
detector = PestDetector("best_models/best_model_final_RGB_focal_loss.keras", model_type="cnn")
resultados = detector.predict("ruta/a/una/imagen_rgb.tif")
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

> Para replicar **todas** las combinaciones registradas en las bitácoras (los 6
> modelos × los 3 umbrales usados en evaluación/inferencia), con la sintaxis exacta
> de las dos ramas de trabajo, ver [`EJECUCION.md`](EJECUCION.md). Acá abajo queda
> un ejemplo mínimo de cada paso.

### 1️⃣ 🧠 Entrenamiento de Modelos

> **Nota:** `train.py` no tiene flag `-t/--threshold` (se eliminó: no afectaba en nada
> al entrenamiento, y correr el mismo comando con distintos `-t` como antes solo
> reentrenaba y sobreescribía el mismo modelo varias veces). El umbral de decisión se
> define recién al *evaluar* o *inferir* con un modelo ya entrenado, con
> `evaluate.py`/`infer.py` (ver más abajo). Los comandos de abajo usan `pest-train`/
> `pest-evaluate`/`pest-infer` (requiere `pip install -e .`, ver arriba); sin instalar
> el paquete, reemplazá cada uno por `python -m pest_detection.cli.train`/`evaluate`/
> `infer` y corré desde la raíz del repo. `-b/--base_dir` (opcional en los tres) elige
> dónde se crean `best_models/`/`evaluation_results/`/etc. y por defecto es el
> directorio actual.

#### **- CNN Multiespectral — Focal Loss**
```bash
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn
```

#### **- CNN Multiespectral — Binary Crossentropy**
```bash
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn
```

#### **- CNN RGB — Focal Loss**
```bash
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 20 -a 0.75 -g 1.5 -mt cnn -rgb
```

#### **- CNN RGB — Binary Crossentropy**
```bash
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 20 -mt cnn -rgb
```

#### **- Random Forest**
> Requiere que ya exista el `.keras` correspondiente en `best_models/` (entrenado antes
> con uno de los comandos CNN de arriba, mismo tipo RGB/MS y mismo `-lt`): Random Forest
> no entrena una CNN propia, extrae features de una ya entrenada.
```bash
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf

pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf -rgb
```
------------------------------------------------------------------------

### 2️⃣ 📊 Evaluación de Modelos

#### Los resultados de evaluación se guardan en:
- `evaluation_results/`

#### Ejemplo de evaluación:
```bash
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
```

#### Evaluación Random Forest:
> El nombre del `.joblib` incluye la fecha/hora en que se entrenó ese Random Forest en
> particular (lo imprime `train.py -mt rf` al terminar) - reemplazá el de abajo por el
> que tengas realmente en tu carpeta `best_models/`.
```bash
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf
```

### 3️⃣ 🔎 Inferencia (Predicción)

Permite predecir si una imagen (o carpeta de muestra) corresponde a plaga o planta
sana. El script vigente es `infer.py`, unificado para CNN y Random Forest, RGB y
multiespectral.

#### Ejemplo CNN Multiespectral:
```bash
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
```

#### Ejemplo CNN RGB:
```bash
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
```

#### Random Forest:
> `-l/--loss` (`fl` o `bce`) indica de qué CNN "hermana" extraer las features - debe
> existir esa CNN entrenada en `best_models/` para el mismo tipo RGB/MS. El nombre del
> `.joblib` varía por corrida (ver nota de evaluación arriba).
```bash
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -l fl
```


## ✅ Tests

```bash
pip install pytest
pytest                      # corre todo (unitarios + integración con checkpoints reales)
pytest -m "not integration" # solo los unitarios, rápido, sin cargar TensorFlow/modelos
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

Tras ejecutar las evaluaciones, encontrarás los reportes de clasificación en formato JSON/Markdown (y las matrices de confusión y curvas ROC ploteadas) dentro de `evaluation_results/{CNN|RANDOM_FOREST}/{RGB|MULTIESPECTRAL}/...`.

### Métricas Clave

La métrica más importante en este contexto, dado el desbalance y el costo de un Falso Negativo (no detectar una plaga), es el Recall de la clase "Plaga" y el F1-Score de la clase minoritaria:

| Métrica | Enfoque | Interpretación para "Plaga" |
| :--- | :--- | :--- |
|Recall | Deep Learning / RF | ¿Cuántas muestras de "Plaga" se detectaron correctamente? |
|Precision | Deep Learning / RF | De todas las muestras predichas como "Plaga", ¿cuántas eran realmente "Plaga"? |
|F1-Score | Deep Learning / RF | Promedio armónico de Precision y Recall. El mejor indicador de rendimiento balanceado. |

## Definiciones: ¿Cuáles son los positivos y negativos? `Positivos = Sanos`, `Negativos = Plagas`
<a href= "https://www.youtube.com/watch?v=H8FSfqxRWmA">YouTube</a> | <a href="https://codificandobits.com/blog/precision-recall-f-score/">Blog</a> | <a href="https://colab.research.google.com/drive/10xngRuU0kyxGildcx7YfxQzjrk_nXXU-?usp=sharing">Colab</a>
- **Verdaderos Positivos / True Positive (VP o TP):** `"plagas"` clasificados **_realmente_** como `"plagas"`.
- **Falsos Positivos / False Positive (FP o FP):** `"sanos"` clasificados **_equivocadamente_** como `"plagas"`.
- **Verdaderos Negativos / True Negative (VN o TN):** `"sanos"` clasificados **_realmente_** como `"sanos"`.
- **Falsos Negativos / False Negative (FN o FN):** `"plagas"` clasificados **_equivocadamente_** como `"sanos"`.


<b>Objetivo:</b> El mejor modelo será aquel que logre un alto Recall para la clase Plaga (para no dejar ninguna plaga sin identificar) sin sacrificar demasiado el Recall de la clase Sana (para evitar la mayoría de falsas alarmas).