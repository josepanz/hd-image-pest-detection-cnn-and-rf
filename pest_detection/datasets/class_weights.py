from typing import Dict

# --- CONFIGURACIÓN GLOBAL ---
CLASSES = ["Plaga", "Sana"] 

def calculate_class_weights(n_plagas: int, n_sanas: int) -> Dict[int, float]:
    """
    Calcula los pesos de clase para manejar el desbalance de datos, usando la fórmula
    estándar de peso inverso a la frecuencia: peso_clase = total / (num_clases * conteo_clase).

    Solo se usa para binary_crossentropy (ver train.py::train): focal_loss ya compensa
    el desbalance internamente vía su parámetro alpha, así que no debe combinarse con
    class_weight (sería aplicar la corrección dos veces).
    """
    total_samples = n_plagas + n_sanas
    if total_samples == 0:
        return {0: 1.0, 1: 1.0}

    # Calcula el peso: peso_clase = total / (num_clases * conteo_clase)
    weight_plaga = total_samples / (len(CLASSES) * n_plagas) if n_plagas > 0 else 1.0
    weight_sana = total_samples / (len(CLASSES) * n_sanas) if n_sanas > 0 else 1.0
    # factor_sensibilidad: boost manual extra sobre el peso de "Plaga" (más allá del
    # balanceo por frecuencia), porque en este dominio un falso negativo de plaga es
    # más costoso que uno de sana. Valor elegido empíricamente, no derivado de los datos.
    factor_sensibilidad = 1.5
    weight_plaga = weight_plaga * factor_sensibilidad

    print(f"Pesos de Clase Calculados: Plaga (0): {weight_plaga:.2f}, Sana (1): {weight_sana:.2f}")
    print("-" * 30)
    print(f"DEBUG - Conteo: Plaga: {n_plagas}, Sana: {n_sanas}")
    print(f"RESULTADO - Pesos Finales: Plaga (0): {weight_plaga:.2f}, Sana (1): {weight_sana:.2f}")
    print("-" * 30)

    return {0: weight_plaga, 1: weight_sana}