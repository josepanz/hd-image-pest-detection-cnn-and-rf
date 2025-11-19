from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

##########################################################################################################################################
def basic_train_and_evaluate_rf_model (X_train, X_test, y_train, y_test, le, _4D: bool = True, encode_label: bool = True):
  """
  Entrena y evalúa un modelo Random Forest básico.
  """
  # 0. Aplanamiento de los datos de imagen 4D a 2D para Random Forest
  # El -1 en .reshape() calcula automáticamente la nueva dimensión de las características
  if _4D:
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"\n--- Preparación de Datos para Random Forest ---")
    print(f"X_train aplanado shape: {X_train_flat.shape}")
    print(f"X_test aplanado shape: {X_test_flat.shape}")

  # 1. Inicializar el Modelo
  model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

  # 2. Entrenar el Modelo
  print("\nIniciando entrenamiento del modelo...")
  if _4D:
    model.fit(X_train_flat, y_train)
  else:
    model.fit(X_train, y_train) # si el modelo fuere 2D, como es 4D para imagenes
  print("Entrenamiento completado.")

  # 3. Evaluar el Modelo
  if _4D:
    y_pred = model.predict(X_test_flat)
  else:
    y_pred = model.predict(X_test) # si el modelo fuere 2D, como es 4D para imagenes

  # 4. Reportar Resultados
  print("\n--- Reporte de Clasificación ---")
  # Usamos inverse_transform para mostrar las etiquetas reales en el reporte
  if encode_label:
    target_names = le.inverse_transform(model.classes_) 
  else:
    target_names = le.classes_ # si no es codificado, o sea, 0 = plaga, 1 = sana, 2 = indeterminado

  print(classification_report(y_test, y_pred, target_names=target_names))
  print(f"Precisión General (Accuracy): {accuracy_score(y_test, y_pred):.4f}")


##########################################################################################################################################
