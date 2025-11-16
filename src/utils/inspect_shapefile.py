import argparse
import fiona
import pandas as pd
import sys
import os

def inspect_shapefile(shapefile_path: str):
    """
    Abre un Shapefile y extrae su tabla de atributos (el contenido del .dbf) 
    para identificar la columna de etiquetas.
    """
    if not os.path.exists(shapefile_path):
        print(f"❌ Error: Shapefile no encontrado en la ruta: {shapefile_path}")
        return

    print(f"Cargando Shapefile: {shapefile_path}")
    
    try:
        # Abrir el Shapefile con fiona
        with fiona.open(shapefile_path, "r") as source:
            
            print("\n--- Metadatos del Shapefile ---")
            print(f"Sistema de Coordenadas (CRS): {source.crs}")
            print(f"Total de Registros (Polígonos): {len(source)}")
            
            # Obtener el esquema de atributos (las columnas)
            print("\n--- Esquema de Atributos (Columnas) ---")
            for prop_name, prop_type in source.schema['properties'].items():
                print(f"- **{prop_name}** (Tipo: {prop_type})")
            
            # Extraer los atributos de todos los registros
            attributes_list = []
            for i, feature in enumerate(source):
                # feature['properties'] contiene la fila de atributos (DBF)
                attributes_list.append(feature['properties'])
                # Solo mostrar una muestra de los primeros 5 registros
                if i < 4: 
                    print(f"   Muestra {i+1}: {feature['properties']}")

            # Convertir a DataFrame de Pandas para una mejor visualización
            df = pd.DataFrame(attributes_list)

            # Mostrar las 5 primeras filas del DataFrame
            print("\n--- Primeras 5 Filas de la Tabla de Atributos (.dbf) ---")
            print(df.head())
            
            # Mostrar valores únicos para las columnas más pequeñas (posibles etiquetas)
            print("\n--- Valores Únicos en Columnas (Para identificar la Etiqueta) ---")
            for col in df.columns:
                # Comprobar si la columna tiene pocos valores únicos (candidata a etiqueta)
                if df[col].nunique() <= 10 and len(df[col].dropna()) > 0:
                    print(f"- Columna **'{col}'** (Valores únicos): {df[col].unique()}")

    except Exception as e:
        print(f"❌ Error al procesar el Shapefile: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspecciona el archivo .dbf del Shapefile para encontrar la columna de etiquetas.")
    parser.add_argument("--shapefile_path", "-sp", default="data\multispectral_images\TTADDA_NARO_2023_F1\metadata\plot_shapefile.shp", help="Ruta al archivo .shp (plot_shapefile.shp)")
    args = parser.parse_args()
    
    inspect_shapefile(args.shapefile_path)

if __name__ == "__main__":
    # Ejemplo de ejecución:
    # python src/data_management/inspect_shapefile.py plot_shapefile.shp 
    main()