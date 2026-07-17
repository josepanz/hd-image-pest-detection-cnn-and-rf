"""Fuerza un backend de matplotlib no interactivo para toda la suite.

Varios módulos del proyecto llaman plt.show()/crean figuras (utils_metrics.py,
print_utils.py) pensados para correr a mano en una consola interactiva. Corriendo
muchos tests seguidos en este proceso, el backend interactivo por defecto en Windows
(TkAgg) puede fallar por falta de una sesión de escritorio real / Tcl-Tk mal
inicializado. Esto no cambia el comportamiento real de los scripts para un usuario
interactivo (solo afecta cómo corren los tests en este proceso).
"""

import matplotlib

matplotlib.use("Agg")
