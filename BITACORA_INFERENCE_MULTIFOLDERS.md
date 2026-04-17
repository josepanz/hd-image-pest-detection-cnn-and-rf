# Resultados de Inferencia | Tiempos | Todos con UMBRAL 0.70
## Convolutional Neural Network (CNN) Multiespectral (MS) Focal Loss (FL)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_MULTIESPECTRAL_focal_loss.keras -mt cnn -t 0.70                        
2026-04-17 11:45:27.212173: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:45:28.621686: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.00s] ---
init. 🚀 Modo: MULTIESPECTRAL | Arq: CNN | Config: Focal Loss

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_MULTIESPECTRAL_focal_loss.keras
2026-04-17 11:45:29.765479: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.10s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.10s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.10s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=0.10s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=3.29s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=6.25s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=12.91s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=15.97s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=19.11s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=22.35s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=25.64s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=36.15s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=39.14s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=42.01s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=45.08s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=48.10s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=50.98s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=53.82s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=56.72s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=59.44s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=65.97s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=68.97s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=75.54s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1146] ---

--- [T=75.54s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/MULTIESPECTRAL/0.7\results_MULTIESPECTRAL_focal_loss_t0.7_20260417_1146.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/MULTIESPECTRAL/0.7\inference_confidence_plot_CNN_MULTIESPECTRAL_20260417_1145.png

--- [Fecha/hora inicio=20260417_1145] ---

--- [T=75.85s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/MULTIESPECTRAL/0.7 
````

## Convolutional Neural Network (CNN) Multiespectral (MS) Binary Cross Entropy (BCE)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_MULTIESPECTRAL_binary_crossentropy.keras -mt cnn -t 0.70               
2026-04-17 11:47:13.273892: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:47:15.179948: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.00s] ---
init. 🚀 Modo: MULTIESPECTRAL | Arq: CNN | Config: BCE

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_MULTIESPECTRAL_binary_crossentropy.keras
2026-04-17 11:47:16.929604: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.09s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.09s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.09s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=0.09s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=3.12s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=6.09s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=9.00s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=11.86s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=19.85s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=29.76s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=33.78s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=37.57s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=41.26s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=44.52s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=55.38s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=58.99s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=62.67s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=69.80s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=73.29s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=76.56s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=80.01s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=83.76s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=87.16s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1148] ---

--- [T=87.16s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/MULTIESPECTRAL/0.7\results_MULTIESPECTRAL_bce_t0.7_20260417_1148.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/MULTIESPECTRAL/0.7\inference_confidence_plot_CNN_MULTIESPECTRAL_20260417_1147.png

--- [Fecha/hora inicio=20260417_1147] ---

--- [T=87.49s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/MULTIESPECTRAL/0.7
````

## Convolutional Neural Network (CNN) RGB (RGB) Binary Cross Entropy (BCE)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_RGB_binary_crossentropy.keras -mt cnn -t 0.70            
2026-04-17 11:16:40.060864: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:16:41.718846: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.00s] ---
init. 🚀 Modo: RGB | Arq: CNN | Config: BCE

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_RGB_binary_crossentropy.keras
2026-04-17 11:16:43.127779: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.08s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.08s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.09s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.09s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=0.73s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=1.31s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=1.92s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=2.52s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=3.10s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=3.80s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=4.45s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=5.13s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=5.80s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=6.42s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=7.13s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=7.85s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=8.52s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=9.18s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=9.91s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=10.57s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=11.26s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=11.97s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=12.67s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=12.67s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/RGB/0.7\results_RGB_bce_t0.7_20260417_1116.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/RGB/0.7\inference_confidence_plot_CNN_RGB_20260417_1116.png

--- [Fecha/hora inicio=20260417_1116] ---

--- [T=12.87s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/BCE/RGB/0.7
````

## Convolutional Neural Network (CNN) RGB (RGB) Focal Loss (FL)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_RGB_focal_loss.keras -mt cnn -t 0.70          
2026-04-17 11:26:43.124881: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:26:44.561833: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.00s] ---
init. 🚀 Modo: RGB | Arq: CNN | Config: Focal Loss

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_RGB_focal_loss.keras
2026-04-17 11:26:45.548931: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.13s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.13s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.13s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.13s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=0.71s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=1.20s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=1.74s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=2.25s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=2.75s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=3.51s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=4.17s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=4.78s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=5.40s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=5.96s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=6.56s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=7.16s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=7.74s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=8.31s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=8.95s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=9.51s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=10.11s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=10.72s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=11.31s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=11.31s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/RGB/0.7\results_RGB_focal_loss_t0.7_20260417_1126.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/RGB/0.7\inference_confidence_plot_CNN_RGB_20260417_1126.png

--- [Fecha/hora inicio=20260417_1126] ---

--- [T=11.51s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/cnn/Focal Loss/RGB/0.7
````

## Convolutional Neural Network (CNN Extractor) + Random Forest (RF classifier) Multiespectral (MS) Focal Loss (FL)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l fl
2026-04-17 11:27:56.240506: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:27:57.407477: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.00s] ---
init. 🚀 Modo: MULTIESPECTRAL | Arq: RF | Config: Focal Loss

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib
2026-04-17 11:27:58.563127: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.34s] ---
1.1. ⏳ Cargando extractor de features desde: best_model_final_MULTIESPECTRAL_focal_loss.keras

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.38s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.38s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.38s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=0.38s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=3.41s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=6.29s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=10.11s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=15.56s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=22.16s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=25.28s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=28.30s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=31.40s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=38.04s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=40.85s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=49.80s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=54.04s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=60.54s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=67.01s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=70.04s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=72.78s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=75.65s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=78.79s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=81.72s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1129] ---

--- [T=81.72s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/MULTIESPECTRAL/0.7\results_MULTIESPECTRAL_focal_loss_t0.7_20260417_1129.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/MULTIESPECTRAL/0.7\inference_confidence_plot_RF_MULTIESPECTRAL_20260417_1127.png

--- [Fecha/hora inicio=20260417_1127] ---

--- [T=81.98s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/MULTIESPECTRAL/0.7   
````

## Convolutional Neural Network (CNN Extractor) + Random Forest (RF classifier) Multiespectral (MS) Binary Cross Entropy (BCE)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib -mt rf -t 0.70 -l bce
2026-04-17 11:30:54.303790: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:30:56.062493: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.00s] ---
init. 🚀 Modo: MULTIESPECTRAL | Arq: RF | Config: BCE

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_random_forest_MULTIESPECTRAL_20260325_1611.joblib
2026-04-17 11:30:57.783157: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.27s] ---
1.1. ⏳ Cargando extractor de features desde: best_model_final_MULTIESPECTRAL_binary_crossentropy.keras

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.31s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.31s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.32s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=0.32s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=3.39s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=6.35s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=9.40s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=12.39s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=15.38s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=18.51s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=21.56s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=24.77s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=27.73s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=30.55s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=33.55s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=36.55s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=39.41s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=42.26s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=45.19s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=48.05s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=51.05s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=54.00s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=58.75s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1131] ---

--- [T=58.82s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/MULTIESPECTRAL/0.7\results_MULTIESPECTRAL_bce_t0.7_20260417_1131.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/MULTIESPECTRAL/0.7\inference_confidence_plot_RF_MULTIESPECTRAL_20260417_1130.png

--- [Fecha/hora inicio=20260417_1130] ---

--- [T=60.73s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/MULTIESPECTRAL/0.7
````

## Convolutional Neural Network (CNN Extractor) + Random Forest (RF classifier) RGB (RGB) Focal Loss (FL)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l fl            
2026-04-17 11:41:29.848032: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:41:50.214337: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=0.00s] ---
init. 🚀 Modo: RGB | Arq: RF | Config: Focal Loss

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_random_forest_RGB_20260325_1637.joblib
2026-04-17 11:42:02.138133: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=1.57s] ---
1.1. ⏳ Cargando extractor de features desde: best_model_final_RGB_focal_loss.keras

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=1.62s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=1.62s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=1.65s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=1.65s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=7.71s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=8.36s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=9.01s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=9.67s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=10.31s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=11.12s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=11.79s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=12.51s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=13.23s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=13.95s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=14.73s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=15.51s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=16.32s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=17.07s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=17.84s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=18.62s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=19.41s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=20.20s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=20.94s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=20.94s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/RGB/0.7\results_RGB_focal_loss_t0.7_20260417_1142.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/RGB/0.7\inference_confidence_plot_RF_RGB_20260417_1142.png

--- [Fecha/hora inicio=20260417_1142] ---

--- [T=21.60s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/Focal Loss/RGB/0.7
````

## Convolutional Neural Network (CNN Extractor) + Random Forest (RF classifier) RGB (RGB) Binary Cross Entropy (BCE)
### Consola/Logs:
````bash
(venv) PS C:\workspace\hd-image-pest-detection-cnn-and-rf> python src\pest_detection_models\inference_models.py predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data -m .\src\pest_detection_models\best_models\best_model_final_random_forest_RGB_20260325_1637.joblib -mt rf -t 0.70 -l bce
2026-04-17 11:43:12.860515: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-04-17 11:43:14.214391: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.00s] ---
init. 🚀 Modo: RGB | Arq: RF | Config: BCE

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.00s] ---
1. ⏳ Cargando modelo: best_model_final_random_forest_RGB_20260325_1637.joblib
2026-04-17 11:43:15.382229: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.19s] ---
1.1. ⏳ Cargando extractor de features desde: best_model_final_RGB_binary_crossentropy.keras

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.24s] ---
2. 🔎 Procesando imágenes y realizando predicción...

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.24s] ---
riop 1. 🔎 Buscando muestras en: predict-test\multiespectral\TTADDA_NARO_2021_F1\drone_data

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.25s] ---
riop 2. ✅ Se encontraron 19 muestras para procesar.

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.25s] ---
riop 3. 🚀 Procesando muestra: 2021-05-18

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=0.83s] ---
riop 3. 🚀 Procesando muestra: 2021-05-25

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=1.35s] ---
riop 3. 🚀 Procesando muestra: 2021-06-01

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=1.91s] ---
riop 3. 🚀 Procesando muestra: 2021-06-07

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=2.45s] ---
riop 3. 🚀 Procesando muestra: 2021-06-14

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=2.98s] ---
riop 3. 🚀 Procesando muestra: 2021-06-22

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=3.60s] ---
riop 3. 🚀 Procesando muestra: 2021-06-29

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=4.17s] ---
riop 3. 🚀 Procesando muestra: 2021-07-07

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=4.80s] ---
riop 3. 🚀 Procesando muestra: 2021-07-13

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=5.43s] ---
riop 3. 🚀 Procesando muestra: 2021-07-20

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=5.99s] ---
riop 3. 🚀 Procesando muestra: 2021-07-27

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=6.60s] ---
riop 3. 🚀 Procesando muestra: 2021-08-03

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=7.23s] ---
riop 3. 🚀 Procesando muestra: 2021-08-11

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=7.82s] ---
riop 3. 🚀 Procesando muestra: 2021-08-17

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=8.42s] ---
riop 3. 🚀 Procesando muestra: 2021-08-31

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=9.03s] ---
riop 3. 🚀 Procesando muestra: 2021-09-06

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=9.61s] ---
riop 3. 🚀 Procesando muestra: 2021-09-14

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=10.20s] ---
riop 3. 🚀 Procesando muestra: 2021-09-21

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=10.86s] ---
riop 3. 🚀 Procesando muestra: 2021-09-28

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=11.50s] ---
3. ✅ Inferencia completada. Procesados: 19 items.

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=11.50s] ---
riop 5. 📂 JSON guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/RGB/0.7\results_RGB_bce_t0.7_20260417_1143.json
📈 Gráfico de confianza guardado en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/RGB/0.7\inference_confidence_plot_RF_RGB_20260417_1143.png

--- [Fecha/hora inicio=20260417_1143] ---

--- [T=11.70s] ---
END. ✨ Proceso finalizado. Resultados en: C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\inference-results/rf/BCE/RGB/0.7
````