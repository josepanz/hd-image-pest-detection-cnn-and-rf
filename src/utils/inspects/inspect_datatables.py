import pandas as pd
import os

# --- Configuración de la Ruta del Archivo ---
# NOTA: Debes ajustar esta ruta para que apunte a donde descargaste el archivo.
# Suponiendo que has descargado el dataset de Zenodo y lo has descomprimido en una carpeta 'data'.
# El repositorio de GitHub indica: data/measurements/...
RUTA_BASE_DEL_PROYECTO = "./data" # Ajusta si tu ruta es diferente
NOMBRE_ARCHIVO = "Alternaria_ocenjevanje1_Ecobreed_krompir_2022.xlsx"
RUTA_COMPLETA_EXCEL = os.path.join(RUTA_BASE_DEL_PROYECTO, 'measurements', NOMBRE_ARCHIVO)

# --- Carga e Inspección ---
try:
    # 1. Cargar el archivo Excel. 
    # Usamos sheet_name=0 para cargar la primera hoja por defecto.
    df_alternaria = pd.read_excel(RUTA_COMPLETA_EXCEL)
    
    print("✅ Archivo cargado correctamente.")
    
    # 2. Imprimir los nombres de las columnas
    print("\n--- Nombres de las Columnas (Cruciales para el JOIN) ---")
    print(df_alternaria.columns.tolist())
    
    # 3. Imprimir las primeras filas para ver el formato de los datos
    print("\n--- Primeras 5 Filas (Formato de Datos) ---")
    print(df_alternaria.head())

except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo en la ruta: {RUTA_COMPLETA_EXCEL}")
    print("Asegúrate de haber descargado el dataset de Zenodo y de que la ruta 'RUTA_BASE_DEL_PROYECTO' sea correcta.")
except Exception as e:
    print(f"❌ Ocurrió un error al procesar el archivo: {e}")