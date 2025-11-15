import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer # Para manejar valores NaN

# 1. Cargar el CSV unificado
df = pd.read_csv('C:\workspace\hd-image-pest-detection-cnn-and-rf\data\multispectral_images\TTADDA_NARO_2023_F1\measurements\generated_labels_unified.csv') 

# 2. Limpieza y Filtrado
# Eliminar filas con la etiqueta "Incierto" o "Sin_GT" si el modelo solo debe clasificar Plaga/Sana/Indeterminado
df_filtered = df[~df['Etiqueta_FINAL'].isin([
    'Sin_GT_Rendimiento', 
    'Incierto (No hay datos)', 
    'No_Vegetacion', 
    'Indeterminado_NDVI', 
    'Indeterminado_SPAD'
])].copy()

# Opcional: Simplificar 'Indeterminado' de Yield si no quieres incluirlo
df_filtered = df_filtered[df_filtered['Etiqueta_FINAL'] != 'Indeterminado'] 

# 3. Definir Características (Features X) y Etiqueta (Target y)

# Seleccionamos las características del dron y las mediciones de punto disponibles:
FEATURES = [
    'NDVI_Dron', 
    'Mean_Red_Reflectance_Dron', 
    'Mean_NIR_Reflectance_Dron',
    'SPAD_Punto',      # SPAD de campo
    'NDVI_Punto'       # NDVI de punto de campo
]

X = df_filtered[FEATURES]
y = df_filtered['Etiqueta_FINAL']

# 4. Manejo de Valores Faltantes (NaN) en las Características
# Es crucial imputar (rellenar) los NaN en X, especialmente si 'SPAD_Punto' o 'NDVI_Punto' tienen valores perdidos.
# Usaremos la media (mean) de la columna para rellenar los valores nulos.
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=FEATURES)


# 5. Codificación de la Etiqueta (Label Encoding)
# Convertir las etiquetas de texto a números (ej: Plaga=0, Sana=1, Indeterminado=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Muestra el mapeo:
print("Mapeo de Etiquetas:", dict(zip(le.classes_, le.transform(le.classes_))))

# 6. Dividir el Conjunto de Datos
# Esto separa los datos en conjuntos de entrenamiento y prueba para evaluar el modelo.
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_encoded, 
    test_size=0.3, # 30% para prueba, 70% para entrenamiento
    random_state=42,
    stratify=y_encoded # Mantiene la proporción de etiquetas en ambos conjuntos
)

print(f"\nConjunto de Entrenamiento (X): {X_train.shape}")
print(f"Conjunto de Prueba (X): {X_test.shape}")

##########################################################################################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Inicializar el Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 2. Entrenar el Modelo
print("\nIniciando entrenamiento del modelo...")
model.fit(X_train, y_train)
print("Entrenamiento completado.")

# 3. Evaluar el Modelo
y_pred = model.predict(X_test)

# 4. Reportar Resultados
print("\n--- Reporte de Clasificación ---")
# Usamos inverse_transform para mostrar las etiquetas reales en el reporte
target_names = le.inverse_transform(model.classes_)

print(classification_report(y_test, y_pred, target_names=target_names))
print(f"Precisión General (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
