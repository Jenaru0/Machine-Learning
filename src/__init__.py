"""
Proyecto Integrador de Machine Learning
======================================

Paquete principal para el proyecto de predicción de rendimiento académico.
Incluye todos los módulos necesarios para el pipeline completo.

Autor: Equipo Grupo 4
Fecha: 2025
"""

# Importaciones usando importlib para archivos con números
import importlib

config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')
eda = importlib.import_module('src.01_eda')
preprocesamiento = importlib.import_module('src.02_preprocesamiento')
entrenar_modelo = importlib.import_module('src.03_entrenar_modelo')
predecir = importlib.import_module('src.04_predecir')

__version__ = "1.0.0"
__author__ = "Equipo Grupo 4"
__email__ = "grupo4@ml.com"
__description__ = "Proyecto Integrador de Machine Learning - Predicción de Rendimiento Académico"
