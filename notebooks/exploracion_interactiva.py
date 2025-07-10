"""
Notebook de Exploración Interactiva
===================================

Notebook Jupyter para exploración interactiva de datos y experimentación
con diferentes técnicas de Machine Learning.

Autor: Equipo Grupo 4
Fecha: 2025
"""

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configurar el entorno
sys.path.insert(0, str(Path.cwd().parent / "src"))

# Configurar matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Importar módulos del proyecto
from src.config import ARCHIVO_DATOS_ORIGINAL
from src.eda import cargar_datos_originales, explorar_datos_basicos
from src.preprocesamiento import ejecutar_preprocesamiento_completo
from src.entrenar_modelo import ejecutar_entrenamiento_completo

print("🔬 Notebook de Exploración Interactiva - Proyecto ML")
print("="*60)
print("📊 Predicción de Rendimiento Académico")
print("👨‍🎓 Equipo Grupo 4")
print("="*60)
