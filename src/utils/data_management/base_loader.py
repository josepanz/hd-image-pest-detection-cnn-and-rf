from typing import Dict

# --- CONFIGURACIÓN GLOBAL ---
CLASSES = ["Plaga", "Sana"] 

def calculate_class_weights(n_plagas: int, n_sanas: int) -> Dict[int, float]:
    """
    Calcula los pesos de clase para manejar el desbalance de datos.
    Se usa la fórmula del total de muestras / (2 * conteo de clase).
    """
    total_samples = n_plagas + n_sanas
    if total_samples == 0:
        return {0: 1.0, 1: 1.0}

    # Calcula el peso: peso_clase = total / (num_clases * conteo_clase)
    weight_plaga = total_samples / (len(CLASSES) * n_plagas) if n_plagas > 0 else 1.0
    weight_sana = total_samples / (len(CLASSES) * n_sanas) if n_sanas > 0 else 1.0
    factor_sensibilidad = 1.5 # 2.5 
    weight_plaga = weight_plaga * factor_sensibilidad

    print(f"Pesos de Clase Calculados: Plaga (0): {weight_plaga:.2f}, Sana (1): {weight_sana:.2f}")
    print("-" * 30)
    print(f"DEBUG - Conteo: Plaga: {n_plagas}, Sana: {n_sanas}")
    print(f"RESULTADO - Pesos Finales: Plaga (0): {weight_plaga:.2f}, Sana (1): {weight_sana:.2f}")
    print("-" * 30)

    return {0: weight_plaga, 1: weight_sana}