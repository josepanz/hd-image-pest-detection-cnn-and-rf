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

---

## 1️⃣ Entrenamiento

Las 6 combinaciones documentadas en `BITACORA.md`. `-mt rf` requiere que ya exista el
`.keras` de la CNN correspondiente (mismo tipo RGB/MS, mismo `-lt`) en `best_models/`.

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
- **Normalización multiespectral** (entrenamiento `/max` por imagen vs. inferencia `/255` fijo): identificada y documentada como bug conocido, **no corregida todavía** (falta validar contra TIFFs reales) — el comportamiento en este punto es el mismo en ambas ramas.

Ver el historial de commits de `chore/cleanup-dead-code-audit` para el detalle completo de cada fix.
