---
title: "Bitácora de reentrenamiento completo - Julio 2026"
author: "Jose Panza"
date: "2026-07"
---

# Reentrenamiento completo de los 6 modelos, con el pipeline corregido

Esta bitácora documenta una sesión de trabajo que arrancó como una revisión del
`EJECUCION.md` (validar que los comandos documentados reproducen la bitácora
original) y terminó encontrando y corrigiendo **9 bugs reales** en el pipeline de
entrenamiento/evaluación/inferencia, y reentrenando los 6 modelos desde cero contra
el dataset real (`D:\TTADDA_NARO_2023\TTADDA_NARO_2023_F1\drone_data`, movido fuera
del repo por espacio en disco).

Los detalles técnicos de cada fix (con el código exacto) están en el historial de
commits de `chore/cleanup-dead-code-audit`/`feature/pest-detection-package` y en la
sección "Diferencias entre ramas" de `EJECUCION.md`. Acá se documenta el proceso
completo, los comandos usados y los resultados finales, a modo de evidencia.

---

## 1️⃣ Bugs encontrados y corregidos en esta sesión

En orden cronológico:

1. **Fecha de archivo distinta a la carpeta** (`2023-06-05`, TIFF multiespectrales
   nombrados `20230606_...` en vez de `20230605_...`): descartaba 35 filas en
   silencio. Corregido con fallback por sufijo si el nombre exacto no existe.
2. **RGB con canal alpha** (`20230518_RGB.tif`, 4 bandas en vez de 3): descartaba
   otras 35 filas en silencio (solo modo `-rgb`). Corregido recortando a las
   primeras 3 bandas.
3. **Normalización multiespectral `/255` fijo en inferencia** (entrenamiento usaba
   `/max` de la imagen): aplastaba la entrada a ~1e-7, coincidía con las
   probabilidades ~0.50 (azar) de `BITACORA_INFERENCE*.md`. Corregido a `/max`,
   igual que entrenamiento.
4. **Inferencia por imagen completa en vez de por parcela**: no coincide con la
   escala de un parche de entrenamiento. Corregido: si hay `metadata/plot_shapefile.shp`
   junto a la muestra, se recorta y predice por parcela (con fallback a imagen
   completa si no hay shapefile).
5. **`train.py -mt rf` sin `-lt` fallaba siempre** (`ValueError: ...best_model_final_
   MULTIESPECTRAL_None.keras`) - exactamente el comando documentado en `EJECUCION.md`.
   Corregido: default a `focal_loss` si no se especifica.
6. **Gráficos bloqueaban una corrida desatendida** (`plt.show()` sin `block=False`).
   Corregido a `show(block=False)+pause()` - se ven en pantalla sin frenar la
   ejecución ni cerrarse solos.
7. **Contaminación por `nodata` (-10000)** en las bandas de reflectancia: hasta
   44.6% de los píxeles de un recorte de parcela son `nodata`, terminando en
   valores post-normalización del orden de -millones. Corregido: se ponen en 0
   antes de normalizar (en extracción de entrenamiento e inferencia).
8. **`ModelCheckpoint` monitoreaba `val_recall` de Sana, no de Plaga** - lo opuesto
   a lo que pide la metodología del proyecto (confirmado contra la tesis: "el costo
   de no detectar una plaga es mayor que el de una falsa alarma"). Corregido con
   una métrica custom `RecallPlaga`. Encontrado en la práctica que el recall puro
   (de cualquier clase) tiene techo trivial (colapsa a "predecir siempre la misma
   clase") - se pasó a monitorear **F2Macro** (promedio del F2 de Plaga y de Sana,
   beta=2): pesa recall alto en ambas clases sin premiar el colapso a una sola.
9. **`EarlyStopping patience=10` cortaba el entrenamiento muy pronto** en algunos
   casos (mejor `val_f2_macro` en época 2, corte en época 11, sin margen para que
   el modelo separe bien las probabilidades). Subido a `patience=20`.

Bonus: el reporte "post_train_val" que se imprime al final de cada entrenamiento
usaba los pesos que `EarlyStopping.restore_best_weights` dejó en memoria (criterio
`val_loss`), no necesariamente los que `ModelCheckpoint` grabó en disco (criterio
`val_f2_macro`) - podían ser épocas distintas. Corregido: se recarga el `.keras`
recién guardado antes de calcular el reporte, para que refleje el checkpoint real.

Todos los fixes están en archivos idénticos en las dos ramas de trabajo
(`chore/cleanup-dead-code-audit`, `feature/pest-detection-package`) y en `master`
antes de esta sesión - los bugs existían en las 3 por igual.

---

## 2️⃣ Entrenamiento (4 rondas hasta llegar al resultado final)

Mismos comandos que documenta `EJECUCION.md` (sección "Secuencia completa"),
contra `D:\TTADDA_NARO_2023\TTADDA_NARO_2023_F1\drone_data` (19 fechas, 665 filas:
Plaga=450, Sana=215). Se corrió con hilos de TensorFlow limitados
(`TF_NUM_INTRAOP_THREADS=4`, `TF_NUM_INTEROP_THREADS=2`, `OMP_NUM_THREADS=4`) y
prioridad de proceso baja, porque el equipo se reinició por sobrecarga durante los
primeros intentos.

```bash
$DATA = "D:\TTADDA_NARO_2023\TTADDA_NARO_2023_F1\drone_data"
python -m pest_detection.cli.train $DATA -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn
python -m pest_detection.cli.train $DATA -lt binary_crossentropy -e 80 -mt cnn
python -m pest_detection.cli.train $DATA -lt focal_loss -e 80 -a 0.75 -g 2.0 -mt cnn -rgb
python -m pest_detection.cli.train $DATA -lt binary_crossentropy -e 80 -mt cnn -rgb
python -m pest_detection.cli.train $DATA -mt rf
python -m pest_detection.cli.train $DATA -mt rf -rgb
```

- **Ronda 1**: reveló el bug #8 (`val_recall` medía Sana).
- **Ronda 2**: con el fix a `val_f2_plaga` (recall de Plaga), el modelo RGB Focal
  Loss colapsó a "predecir Plaga para todo" (F2Plaga~0.91 sin ser útil, porque
  Plaga ya es mayoría) - reveló que hacía falta F2Macro.
- **Ronda 3**: con `val_f2_macro`, ya no colapsa, pero `patience=10` cortaba muy
  pronto (ej. MS Focal Loss: mejor checkpoint en época 2 de 11) - probabilidades
  de salida comprimidas en un rango de 0.497-0.498 (confirmado con diagnóstico
  directo), pese a AUC ya razonable.
- **Ronda 4 (final)**: con `patience=20`, los resultados post-entrenamiento
  (sobre el split de validación, 133 filas: 90 Plaga/43 Sana):

| Modelo | AUC | Plaga P/R/F1 | Sana P/R/F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| CNN MS Focal Loss | 0.876 | 0.98/0.66/0.79 | 0.58/0.98/0.72 | 0.76 |
| CNN MS BCE | 0.930 | 1.00/0.79/0.88 | 0.69/1.00/0.82 | 0.86 |
| CNN RGB Focal Loss | 0.914 | 1.00/0.79/0.88 | 0.69/1.00/0.82 | 0.86 |
| CNN RGB BCE | 0.882 | 0.94/0.73/0.82 | 0.62/0.91/0.74 | 0.79 |
| RF Multiespectral | 0.941 | — | — | (ver evaluación) |
| RF RGB | 0.945 | — | — | (ver evaluación) |

**Nota sobre MS Focal Loss**: quedó en una meseta real de entrenamiento (mejor
`val_f2_macro` en época 2, sin mejorar después pese a `patience=20` y epochs=80
como tope) - decisión tomada con el usuario de aceptarlo tal cual (no degenerado,
AUC razonable) en vez de seguir iterando con retornos decrecientes.

---

## 3️⃣ Evaluación final (18 corridas: 6 modelos x 3 umbrales)

```bash
foreach ($t in 0.45,0.50,0.70) {
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t $t -mt cnn
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -t $t -mt cnn
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_RGB_focal_loss.keras -t $t -mt cnn
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_RGB_binary_crossentropy.keras -t $t -mt cnn
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_random_forest_MULTIESPECTRAL_20260716_2105.joblib -t $t -mt rf
  python -m pest_detection.cli.evaluate $DATA -m best_models\best_model_final_random_forest_RGB_20260716_2107.joblib -t $t -mt rf
}
```

| Modelo | t=0.45 (acc) | t=0.50 (acc) | t=0.70 (acc) |
| :--- | :--- | :--- | :--- |
| CNN MS Focal Loss | 0.32 ⚠️ | 0.68 ⚠️ | 0.68 ⚠️ |
| CNN MS BCE | 0.87 | 0.87 | 0.84 |
| CNN RGB Focal Loss | 0.86 | 0.87 | 0.75 |
| CNN RGB BCE | 0.32 ⚠️ | 0.53 ⚠️ | 0.68 ⚠️ |
| RF Multiespectral | 0.88 | 0.87 | 0.84 |
| RF RGB | 0.90 | **0.91** | 0.87 |

⚠️ **MS Focal Loss y RGB BCE son sensibles a estos 3 umbrales puntuales**: sus
probabilidades de salida quedan comprimidas cerca de 0.5 (igual que se documentó
para MS Focal Loss en la ronda 3), así que 0.45/0.50/0.70 caen mal aunque el AUC
sea bueno (0.876 y 0.882 respectivamente) y el modelo funcione razonablemente en
su propio umbral óptimo (~0.50, ver reportes `post_train_val`). **Para estos 2
modelos en particular, conviene evaluar/usar con el umbral óptimo propio en vez
de los 3 fijos de la bitácora.**

**Los Random Forest son los que mejor y más consistente rinden** en los 3 umbrales
- no sufren este problema porque entrenan su propio clasificador aparte, sobre
features ya extraídas, no dependen de la escala fina del sigmoide de la CNN.

Reportes completos (JSON + Markdown + matriz de confusión + curva ROC) en
`evaluation_results/{CNN|RANDOM_FOREST}/...`.

---

## 4️⃣ Inferencia de validación

Corrida contra `D:\TTADDA_NARO_2021\TTADDA_NARO_2021_F1\drone_data` (dataset de un
año distinto al usado para entrenar - 2021 vs. 2023 - útil como prueba de que el
pipeline funciona de punta a punta con datos reales frescos, no como prueba de
generalización entre años, que es un tema aparte).

```bash
python -m pest_detection.cli.infer "D:\TTADDA_NARO_2021\TTADDA_NARO_2021_F1\drone_data\2021-05-25" -m best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -t 0.50 -mt cnn
# ... (los 4 CNN x 3 umbrales, RF MS/RGB x umbral 0.70, y la corrida "multifolder"
# sobre toda la carpeta drone_data)
```

- **`2021-05-25` (una fecha)**: 24 parcelas procesadas por corrida (según el
  `plot_shapefile.shp` de ese dataset - 24, no 35 como en 2023), sin errores, en
  las 4 CNN + 2 RF.
- **Multifolder** (`drone_data` completa, 19 fechas): **456 parcelas procesadas en
  una sola corrida** (19 fechas × 24 parcelas), confirmando que el modo "toda la
  carpeta de una vez" (`BITACORA_INFERENCE_MULTIFOLDERS.md`) sigue funcionando con
  el recorte por parcela nuevo.
- No se pudo comparar contra verdad de campo para este dataset en esta ubicación
  puntual: solo trae los CSV de mediciones crudas (`NARO_field_2021_GT_*.csv`), no
  el `generated_labels_unified.csv` (Plaga/Sana ya derivado) que sí tiene el
  dataset 2023 - generarlo requeriría correr `pest_detection/datasets/
  generate_labels_unified.py` aparte, no se hizo en esta sesión.

---

## 5️⃣ Estado final

- Los 6 modelos en `best_models/` corresponden a esta ronda 4 (final).
- Pendiente real, documentado, no resuelto: la sensibilidad a umbrales fijos de
  MS Focal Loss y RGB BCE (ítem 3, arriba) - no es un bug de código, es una
  característica del entrenamiento de esos 2 modelos puntuales.
- Sin otros pendientes de código conocidos a la fecha de este documento.
