import time
from datetime import datetime

def print_time_and_step(step_number, message, timestamp = datetime.now().strftime("%Y%m%d_%H%M"), start_time = time.time()):
    """Calcula y imprime el tiempo transcurrido desde el inicio."""
    elapsed = time.time() - start_time
    print(f"\n--- [Fecha/hora inicio={timestamp}] ---")
    # Usar f-string para formatear el tiempo a segundos con 2 decimales
    print(f"\n--- [T={elapsed:.2f}s] ---")
    print(f"{step_number}. {message}")