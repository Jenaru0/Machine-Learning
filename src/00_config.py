"""
Configuración del Proyecto Integrador de Machine Learning
========================================================

Este archivo contiene todas las rutas, parámetros y configuraciones
utilizadas en el proyecto de predicción de rendimiento académico.

Autor: Equipo Grupo 4
Fecha: 2025
"""

import os
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Ruta base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Rutas de datos
DATOS_DIR = BASE_DIR / "datos"
RAW_DATA_DIR = DATOS_DIR / "raw"
PROFILING_DIR = DATOS_DIR / "profiling"
PROCESADOS_DIR = DATOS_DIR / "procesados"

# Archivos específicos
ARCHIVO_DATOS_ORIGINAL = RAW_DATA_DIR / "StudentPerformanceFactors.csv"
ARCHIVO_REPORTE_EDA = PROFILING_DIR / "reporte_estudiantes.html"
ARCHIVO_DATOS_TRANSFORMADOS = PROCESADOS_DIR / "student_performance_transformado_numerico.csv"
ARCHIVO_TRAIN = PROCESADOS_DIR / "train_student_performance.csv"
ARCHIVO_TEST = PROCESADOS_DIR / "test_student_performance.csv"
ARCHIVO_PREDICCIONES = PROCESADOS_DIR / "predicciones_exam_score.csv"

# Rutas de modelos
MODELOS_DIR = BASE_DIR / "modelos"
MODELO_RIDGE_PATH = MODELOS_DIR / "ridge_alpha_10.pkl"

# Rutas de notebooks
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# =============================================================================
# PARÁMETROS DEL PREPROCESAMIENTO
# =============================================================================

# Columnas a eliminar por baja correlación (de la Guía 2)
COLUMNAS_A_ELIMINAR = [
    'Gender',
    'School_Type', 
    'Sleep_Hours',
    'Physical_Activity',
    'Internet_Access'
]

# Variables categóricas (de la Guía 2)
VARIABLES_CATEGORICAS = [
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Motivation_Level',
    'Family_Income',
    'Teacher_Quality',
    'Peer_Influence',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home'
]

# Variables numéricas (de la Guía 2)
VARIABLES_NUMERICAS = [
    'Hours_Studied',
    'Attendance',
    'Previous_Scores',
    'Tutoring_Sessions'
]

# Variable objetivo
VARIABLE_OBJETIVO = 'Exam_Score'

# =============================================================================
# PARÁMETROS DEL MODELADO
# =============================================================================

# Parámetros de división train/test
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Hiperparámetros para optimización (de la Guía 4)
ALPHAS_RIDGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ALPHAS_LASSO = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MAX_ITER_LASSO = [1000, 2000, 3000]

# Validación cruzada
CV_FOLDS = 5

# Mejor modelo encontrado (de la Guía 4)
MEJOR_ALPHA_RIDGE = 10.0

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

# Configuración básica de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# =============================================================================
# MENSAJES DEL SISTEMA
# =============================================================================

MENSAJES = {
    'inicio_eda': '🔍 Iniciando Análisis Exploratorio de Datos (EDA)...',
    'fin_eda': '✅ EDA completado exitosamente',
    'inicio_preproceso': '🔧 Iniciando preprocesamiento de datos...',
    'fin_preproceso': '✅ Preprocesamiento completado exitosamente',
    'inicio_entrenamiento': '🚀 Iniciando entrenamiento de modelos...',
    'fin_entrenamiento': '✅ Entrenamiento completado exitosamente',
    'inicio_prediccion': '📊 Generando predicciones finales...',
    'fin_prediccion': '✅ Predicciones generadas exitosamente',
    'pipeline_completo': '🎉 Pipeline completo ejecutado exitosamente!'
}

# =============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# =============================================================================

# Tamaños de figura para matplotlib
FIGSIZE_SMALL = (10, 6)
FIGSIZE_MEDIUM = (15, 10)
FIGSIZE_LARGE = (20, 12)

# Colores para gráficos
COLORES_PRINCIPALES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# =============================================================================
# VALIDACIÓN DE RUTAS
# =============================================================================

def crear_directorios():
    """Crea los directorios necesarios si no existen."""
    directorios = [
        DATOS_DIR,
        RAW_DATA_DIR,
        PROFILING_DIR,
        PROCESADOS_DIR,
        MODELOS_DIR,
        NOTEBOOKS_DIR
    ]
    
    for directorio in directorios:
        directorio.mkdir(parents=True, exist_ok=True)

def verificar_archivo_datos():
    """Verifica que el archivo de datos original existe."""
    if not ARCHIVO_DATOS_ORIGINAL.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ARCHIVO_DATOS_ORIGINAL}")
    return True

# Crear directorios al importar el módulo
crear_directorios()
