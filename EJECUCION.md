# 📋 Guía completa de ejecución

Referencia con **todos** los comandos para replicar lo registrado en las bitácoras
(`BITACORA.md`, `BITACORA_INFERENCE.md`, `BITACORA_INFERENCE_MULTIFOLDERS.md`):
entrenamiento de las 6 combinaciones de modelo, evaluación, e inferencia con distintos
umbrales, para RGB y multiespectral, CNN y Random Forest.

Cada sección muestra el comando en **las dos ramas de trabajo**, que usan una sintaxis
distinta:

| Rama | Cómo se invoca |
| :--- | :--- |
| `chore/cleanup-dead-code-audit` | `python -m pest_detection.cli.train` / `.evaluate` / `.infer`, corridos desde la raíz del repo (sin instalar nada más que `requirements.txt`) |
| `feature/pest-detection-package` | `pest-train` / `pest-evaluate` / `pest-infer` (requiere `pip install -e .` una vez), o directamente `import pest_detection` desde otro script |

Qué corrige/mejora cada rama respecto del código original (`master`), por si el
desempeño difiere al comparar corridas: ver el resumen al final de este documento.

## 0️⃣ Prerequisitos (una sola vez, en cualquiera de las dos ramas)

```bash
python -m venv venv
.\venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .                   # solo en feature/pest-detection-package
pip install pytest                 # si vas a correr los tests
```

Además hace falta el dataset TTADDA descargado y descomprimido en
`data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data\` (ver sección
"Descargar archivos para pruebas" del `README.md`) — el repo no lo incluye.

### Dataset fuera del repo (por espacio en disco)

Si `drone_data/` (los `.tif` reales, ~35 GB para TTADDA_NARO_2023_F1) no entra en el
disco donde está el repo, se lo puede dejar en cualquier otra ubicación: `data_dir`
(primer argumento posicional de `train`/`evaluate`/`infer`, en ambas ramas) acepta
una ruta absoluta, en cualquier disco, no tiene que estar bajo `data\`:

```bash
# en vez de la ruta relativa data\multiespectral\TTADDA-dataset\...\drone_data
python -m pest_detection.cli.train D:\TTADDA_NARO_2023\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn
```

**Importante:** sólo hace falta mover los TIFF (`drone_data/`). El CSV de labels
(`generated_labels_unified.csv`) y el shapefile de parcelas (`plot_shapefile.shp`)
se siguen leyendo desde rutas **hardcodeadas dentro del repo**
(`LABELS_CSV`/`PARCELS_SHP` en `pest_detection/datasets/extract_data_to_img.py`,
apuntando a `C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multiespectral\...`)
— ningún flag de `train.py`/`evaluate.py`/`infer.py` las cambia. Si el repo no está
clonado exactamente en esa ruta, o si esos 2 archivos también se movieran fuera de
`data\`, la extracción falla con `FileNotFoundError` antes de abrir un solo TIFF.
`-b/--base_dir` sólo controla dónde se escriben `best_models/`/`evaluation_results/`/
`history/` — no de dónde se leen los datos de entrada.

### Validación de shapes contra el dataset real

Corrida el 2026-07-15 contra las 19 carpetas de fecha reales de
`TTADDA_NARO_2023_F1/drone_data` (665 filas etiquetadas en el CSV: Plaga=450, Sana=215):

| Archivo | Bandas | dtype | Tamaño (H×W) | CRS |
| :--- | :--- | :--- | :--- | :--- |
| `*_RGB.tif` (18 de las 19 fechas) | 3 (R, G, B) | uint8 | ~24464×31891 | EPSG:32654 |
| `20230518_RGB.tif` (única excepción) | **4** (R, G, B, **Alpha**) | uint8 | 24464×31891 | EPSG:32654 |
| `*_transparent_reflectance_{red,red edge,nir,blue,green}.tif` | 1 c/u | float32 | ~6628×11355 | EPSG:32654 |
| `*_DEM.tif` (no lo usa el pipeline) | 1 | float32 | ~24464×31891 | EPSG:32654 |
| `metadata/plot_shapefile.shp` | — | — | 35 parcelas, columna `PlotID` | EPSG:32654 (coincide con los raster) |

Se encontraron y corrigieron 2 problemas reales de datos que hoy están en `master` y
en las dos ramas de trabajo por igual (`extract_data_to_img.py` es idéntico en las 3)
— quedan documentados también en "Diferencias entre ramas" más abajo:

1. **Fecha `2023-06-05`**: los 5 TIFF multiespectrales de esa carpeta están
   nombrados con fecha `20230606` (un día después de la carpeta/RGB/DEM, que sí son
   `20230605`). El código construía `20230605_transparent_reflectance_*.tif`, no lo
   encontraba, y descartaba esas 35 filas etiquetadas **en silencio**
   (`except FileNotFoundError: continue`) — sin error ni warning visible.
2. **Fecha `2023-05-18`**: `20230518_RGB.tif` trae 4 bandas (R/G/B/**Alpha**) en vez
   de 3 (única entre las 19 fechas). El chequeo `expected_channels == 3` fallaba y
   descartaba esas 35 filas **en silencio** — sólo afectaba al modo `-rgb`.

Antes del fix, entrenar en modo `-rgb` perdía 35/665 filas (5.3%) y entrenar
multiespectral perdía otras 35/665 (5.3%, una fecha distinta), sin que se notara en
consola. Verificado post-fix llamando directamente a `extract_data_to_img_for_train`
sobre `D:\...\drone_data`: **665/665 filas se procesan en ambos modos**
(`X_train` con shape `(532, 224, 224, 3)` para RGB y `(532, 224, 224, 5)` para
multiespectral — split 532/133 con `val_split=0.2` fijo).

---

## 1️⃣ Entrenamiento

Las 6 combinaciones documentadas en `BITACORA.md`. `-mt rf` requiere que ya exista el
`.keras` de la CNN correspondiente (mismo tipo RGB/MS, mismo `-lt`) en `best_models/`.

> **Los defaults de `argparse` en `train.py` NO son los usados en las bitácoras:**
> `-e/--epochs` default es `20` (bitácora: `80`), `-a/--alpha` default es `0.50`
> (bitácora: `0.75`), `-g/--gamma` default es `3.0` (bitácora: `2.0`). Corré los
> comandos de abajo tal cual, con esos 3 flags explícitos — omitirlos entrena con
> otros hiperparámetros y no reproduce la bitácora.

<details>
<summary><b>CNN Multiespectral — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn
```
</details>

<details>
<summary><b>CNN Multiespectral — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 80 -mt cnn

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 80 -mt cnn
```
</details>

<details>
<summary><b>CNN RGB — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn -rgb

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn -rgb
```
</details>

<details>
<summary><b>CNN RGB — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 80 -mt cnn -rgb

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -lt binary_crossentropy -e 80 -mt cnn -rgb
```
</details>

<details>
<summary><b>Random Forest — Multiespectral</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf
```
</details>

<details>
<summary><b>Random Forest — RGB</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf -rgb

# feature/pest-detection-package
pest-train data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -mt rf -rgb
```
</details>

**Salida:** `best_models/best_model_final_{RGB|MULTIESPECTRAL}_{focal_loss|binary_crossentropy}.keras` para CNN; `best_models/best_model_final_random_forest_{RGB|MULTIESPECTRAL}_<timestamp>.joblib` para RF (el nombre incluye fecha/hora, no es predecible de antemano — mirá la consola al terminar, o el archivo más nuevo en `best_models/`). También se guarda un reporte de validación post-entrenamiento en `evaluation_results/CNN/.../post_train_val/` y el historial de épocas en `history/`.

Opcional en ambas ramas: `-b <carpeta>` elige dónde se crean `best_models/`/`evaluation_results/`/`history/` (por defecto, el directorio actual).

---

## 2️⃣ Evaluación

Con los 3 umbrales usados en las bitácoras (0.45 / 0.50 / 0.70), para cada uno de los 6 modelos.

<details>
<summary><b>CNN Multiespectral — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.45 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.45 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN Multiespectral — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.45 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.50 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.45 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.50 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN RGB — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.45 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.50 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.45 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.50 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_focal_loss.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN RGB — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.45 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.45 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>Random Forest — Multiespectral / RGB</b></summary>

> Reemplazá el nombre del `.joblib` por el que tengas realmente en `best_models/`
> (incluye la fecha/hora de entrenamiento).

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -t 0.45 -mt rf
python -m pest_detection.cli.evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -t 0.45 -mt rf

# feature/pest-detection-package
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -t 0.45 -mt rf
pest-evaluate data\multiespectral\TTADDA-dataset\TTADDA_NARO_2023_F1\drone_data -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -t 0.45 -mt rf
```
</details>

**Salida:** `evaluation_results/{CNN|RANDOM_FOREST}/{RGB|MULTIESPECTRAL}/{focal_loss|binary_crossentropy}/{threshold}/` (CNN) o `evaluation_results/RANDOM_FOREST/{RGB|MULTIESPECTRAL}/{threshold}/` (RF) — reporte JSON/Markdown + matriz de confusión + curva ROC.

---

## 3️⃣ Inferencia

Con los 3 umbrales de `BITACORA_INFERENCE.md`, sobre la muestra de ejemplo `2021-05-25`
de `predict-test/`. `-l fl`/`-l bce` (solo para RF) indica de qué CNN "hermana" extraer
las features.

> **Recorte por parcela (nuevo):** si junto a la ruta de entrada existe
> `metadata/plot_shapefile.shp` (mismo esquema de carpetas que usa el entrenamiento —
> caso de `predict-test/multiespectral/TTADDA_NARO_2021_F1/` y de cualquier dataset
> TTADDA real), la salida trae **una predicción por parcela** (`2021-05-25_parcela_1`,
> `..._parcela_2`, etc.) en vez de una sola predicción para la imagen completa. Si no
> se encuentra el shapefile, sigue devolviendo una predicción por muestra como antes.
> Ver "Diferencias entre ramas" más abajo para el detalle y la validación contra datos reales.

<details>
<summary><b>CNN Multiespectral — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.45 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.45 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN Multiespectral — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.45 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.50 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.45 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.50 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN RGB — Focal Loss</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.45 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.50 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.45 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.50 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_focal_loss.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>CNN RGB — Binary Crossentropy</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.45 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.70 -mt cnn

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.45 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.50 -mt cnn
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25\20210525_rgb.tif -m best_models\best_model_final_RGB_binary_crossentropy.keras -t 0.70 -mt cnn
```
</details>

<details>
<summary><b>Random Forest (CNN Extractor) — Multiespectral / RGB, Focal Loss / BCE</b></summary>

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l fl
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l bce
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l fl
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l bce

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l fl
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l bce
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l fl
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data\2021-05-25 -m best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l bce
```
</details>

<details>
<summary><b>Procesar TODAS las muestras de una carpeta de una sola corrida</b></summary>

Igual que `BITACORA_INFERENCE_MULTIFOLDERS.md`: pasando la carpeta `drone_data`
completa (en vez de una sola subcarpeta de fecha), se procesan todas las muestras
encontradas en una sola corrida.

```bash
# chore/cleanup-dead-code-audit
python -m pest_detection.cli.infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -mt cnn -t 0.70

# feature/pest-detection-package
pest-infer predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -mt cnn -t 0.70
```
</details>

**Salida:** `inference_results/{CNN|RANDOM_FOREST}/{focal_loss|binary_crossentropy}/{RGB|MULTIESPECTRAL}/{threshold}/` — JSON de resultados + gráfico de confianza.

### Uso como librería (solo `feature/pest-detection-package`)

```python
from pest_detection import PestDetector

detector = PestDetector("best_models/best_model_final_RGB_focal_loss.keras", model_type="cnn")
resultados = detector.predict("predict-test/multiespectral/TTADDA_NARO_2021_F1/drone_data/2021-05-25/20210525_rgb.tif")
```

---

## 4️⃣ Herramientas auxiliares

Scripts de preparación/inspección de datos (no forman parte del pipeline train→evaluate→infer), iguales en ambas ramas:

```bash
python -m pest_detection.tools.inspect_tif -tp <archivo.tif>
python -m pest_detection.tools.inspect_shapefile -sp <archivo.shp>
python -m pest_detection.tools.inspect_datatables
python -m pest_detection.tools.patch_extractor <tiff> <shapefile> <carpeta_salida>
python -m pest_detection.tools.save_rgb_and_bands <carpeta_tiff> -o <carpeta_salida>
python -m pest_detection.tools.utils_tiff_converter
python -m pest_detection.tools.convert_md_to_pdf
python -m pest_detection.evaluation.utils_matriz -e metrics
python -m pest_detection.evaluation.utils_roc
```

---

## 5️⃣ Tests

Igual en ambas ramas:

```bash
pytest                      # todo: unitarios + integración con checkpoints reales
pytest -m "not integration" # solo unitarios, rápido
```

---

## Diferencias entre ramas (por qué el desempeño podría no ser idéntico)

Si entrenás el mismo comando en `master` y en estas ramas y comparás resultados, tené en cuenta:

- **`focal_loss`**: bug de shape corregido (`(batch, batch)` → `(batch,)`) — afecta directamente el gradiente real usado al entrenar con `-lt focal_loss`. Los 2 checkpoints `focal_loss` que ya existen en `best_models/` se entrenaron **antes** de este fix.
- **Random Forest / inferencia**: se corrigió que el `StandardScaler` guardado junto al RF ahora sí se aplica antes de predecir (antes se le pasaban features sin escalar).
- **Inferencia RGB con archivo suelto**: antes devolvía "0 muestras" silenciosamente; ahora funciona igual que pasando la carpeta contenedora.
- **Extracción de datos — fecha de archivo distinta a la de la carpeta**: validado contra el dataset real (`TTADDA_NARO_2023_F1/drone_data`, ver sección de validación de shapes más arriba), la fecha `2023-06-05` tiene sus 5 TIFF multiespectrales nombrados `20230606_transparent_reflectance_*.tif` en vez de `20230605_...`. Antes esto descartaba en silencio las 35 filas de esa fecha (`FileNotFoundError` atrapado y absorbido); ahora, si el nombre exacto no existe, se busca por sufijo dentro de la misma carpeta sin importar el prefijo de fecha.
- **Extracción de datos — RGB con canal alpha**: `20230518_RGB.tif` trae 4 bandas (R/G/B/Alpha) en vez de 3, único caso entre las 19 fechas del dataset real. Antes esto descartaba en silencio las 35 filas de esa fecha en modo `-rgb` (fallaba el chequeo de canales esperados); ahora se recortan las 3 primeras bandas (R/G/B) y se descarta el alpha.
- **Inferencia — normalización multiespectral `/255` fijo → `/max` por imagen**: **corregido y validado contra TIFFs reales**. Las bandas de reflectancia WUR vienen en escala ~0–0.01 (float32, `nodata=-10000`), no 0–255; dividir por 255 aplastaba todo a ~1e-7 (prácticamente cero), lo que coincide con las probabilidades ~0.50 (predicción tipo azar) vistas en `BITACORA_INFERENCE*.md` para el modelo MULTIESPECTRAL. `load_and_preprocess_image` en `inference_utils.py` ahora normaliza por el máximo de la imagen apilada, igual que `extract_data_to_img_for_train`.
- **Inferencia — imagen completa vs. recorte por parcela**: **corregido**. Antes `infer.py` clasificaba la imagen del TIFF completo redimensionada a 224×224 (todo el campo, una sola predicción por fecha) — una escala visual completamente distinta a la de un parche de una sola parcela, que es con lo que se entrenó la CNN. Ahora, si junto a la muestra existe `metadata/plot_shapefile.shp` (mismo esquema que usa el entrenamiento: `<root>/drone_data/<fecha>/` + `<root>/metadata/`), se recorta cada parcela del shapefile con `rasterio.mask` — igual que en entrenamiento — y se predice una vez por parcela (`file_name: "<fecha>_parcela_<PlotID>"` en el JSON de salida). Si no se encuentra el shapefile (por ejemplo, los fixtures sintéticos de los tests de integración), se cae al comportamiento anterior de imagen completa, sin romper nada existente.
  - **Validado end-to-end** corriendo `infer.py` contra `2023-05-18` con el checkpoint MULTIESPECTRAL focal_loss real: pasó de 1 predicción (imagen completa) a **35** (una por parcela, coincide con las 35 parcelas del shapefile). Las 35 filas de ese día están etiquetadas `Plaga` en el CSV real, y post-fix el modelo predice `Plaga` con probabilidad >0.99 en las parcelas revisadas — antes del fix daba ~0.50 (azar). Confirma que las dos correcciones (escala + recorte) sí mueven la aguja.
- **⚠️ Pendiente nuevo, encontrado al validar el recorte por parcela contra TIFFs reales — contaminación por `nodata`**: las bandas de reflectancia WUR marcan fuera-de-cobertura con `nodata=-10000.0`. Al recortar por polígono de parcela, una porción importante de los píxeles cae en esa zona (ejemplo medido: parcela `PlotID=1`, fecha `2023-05-18`, banda `red` → **44.6% de los píxeles del recorte son `-10000`**, no son un caso aislado de borde). Como la normalización divide por el máximo de la imagen (que ignora los negativos), esos píxeles `-10000` terminan en valores post-normalización del orden de `-2.6 millones` en vez de recortarse/enmascararse — ruido muy fuerte de entrada al modelo. **Esto ya estaba en el pipeline de entrenamiento** (`extract_data_to_img.py` usa el mismo `rasterio.mask` + `/max`, sin manejar `nodata`), así que los checkpoints `.keras` ya entrenados en `best_models/` aprendieron con este mismo ruido — por eso **no se tocó** acá: enmascarar el `nodata` solo en inferencia, sin reentrenar, generaría un mismatch train/inferencia nuevo en vez de resolver uno. Corregirlo de raíz implica también arreglar `extract_data_to_img.py` (enmascarar/excluir `nodata` antes de normalizar) y **reentrenar los 2 checkpoints MULTIESPECTRAL** (`focal_loss` y `binary_crossentropy`) — no se hizo porque implica horas de entrenamiento y no estaba pedido; queda para decidir explícitamente antes de encararlo.

Los 4 fixes de extracción/inferencia de datos (fecha de archivo, alpha RGB, normalización `/255→/max`, recorte por parcela) están en archivos idénticos en `master` y en las dos ramas de trabajo (`pest_detection/datasets/extract_data_to_img.py`, `pest_detection/evaluation/inference_utils.py`) — el bug existía en las 3 por igual hasta esta corrección.

Ver el historial de commits de `chore/cleanup-dead-code-audit` para el detalle completo de cada fix.
