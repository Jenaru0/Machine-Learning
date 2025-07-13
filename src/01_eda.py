"""
Análisis Exploratorio de Datos (EDA)
====================================

Módulo que implementa exactamente el código de la GUÍA 1 para el análisis
exploratorio de datos del proyecto de rendimiento académico.

Incluye funciones para:
- Cargar y explorar datos
- Generar reportes con ydata-profiling
- Análisis estadístico básico
- Detección de patrones y tendencias

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Importar configuración y utilidades
import importlib
config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')

# Extraer funciones y constantes necesarias
ARCHIVO_DATOS_ORIGINAL = config.ARCHIVO_DATOS_ORIGINAL
ARCHIVO_REPORTE_EDA = config.ARCHIVO_REPORTE_EDA
MENSAJES = config.MENSAJES
cargar_dataframe = utils.cargar_dataframe
mostrar_info_dataframe = utils.mostrar_info_dataframe
imprimir_separador = utils.imprimir_separador
limpiar_memoria = utils.limpiar_memoria

def cargar_datos_originales() -> pd.DataFrame:
    """
    Carga los datos originales del archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame con los datos originales
    """
    print(MENSAJES['inicio_eda'])
    
    # Cargar datos usando la función de utilidades
    df = cargar_dataframe(ARCHIVO_DATOS_ORIGINAL)
    
    return df

def explorar_datos_basicos(df: pd.DataFrame) -> None:
    """
    Realiza exploración básica de los datos (equivalente al código de la Guía 1).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos originales
    """
    imprimir_separador("EXPLORACIÓN BÁSICA DE DATOS")
    
    # Información general del dataset (de la Guía 1)
    print("\n🔍 INFORMACIÓN GENERAL:")
    print(df.info())
    print("\n📊 PRIMERAS 5 FILAS:")
    print(df.head())
    
    # Valores nulos por columna (de la Guía 1)
    print("\n⚠️  VALORES NULOS POR COLUMNA:")
    print(df.isnull().sum())
    print("-----------------------------------")
    
    # Estadísticas descriptivas (de la Guía 1)
    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())
    print("-----------------------------------")
    
    # Columnas únicas (de la Guía 1)
    print("\n🔢 VALORES ÚNICOS POR COLUMNA:")
    print(df.nunique())
    print("-----------------------------------")
    
    # Duplicados (de la Guía 1)
    print(f"\n📋 FILAS DUPLICADAS: {df.duplicated().sum()}")

def analizar_patrones_tendencias(df: pd.DataFrame) -> None:
    """
    Analiza patrones y tendencias específicas (código exacto de la Guía 1).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos originales
    """
    imprimir_separador("ANÁLISIS DE PATRONES Y TENDENCIAS")
    
    # Código exacto de la Guía 1 - Paso 5
    print("\n📊 ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS:")
    
    print("\n🕐 Hours_Studied (Horas de estudio):")
    print("Estadísticas descriptivas (media, mediana, desviación estándar, etc.) de horas de estudio.")
    print(df["Hours_Studied"].describe())
    
    print("\n🎯 Attendance (Asistencia):")
    print("Distribución de asistencias.")
    print(df["Attendance"].value_counts())
    
    print("\n👨‍👩‍👧‍👦 Parental_Involvement (Participación parental):")
    print("Frecuencias del nivel de participación parental.")
    print(df["Parental_Involvement"].value_counts())
    
    print("\n📚 Access_to_Resources (Acceso a recursos):")
    print("Frecuencias del nivel de acceso a recursos.")
    print(df["Access_to_Resources"].value_counts())
    
    print("\n🏃‍♀️ Extracurricular_Activities (Actividades extracurriculares):")
    print("Conteo de estudiantes que participan vs. no participan.")
    print(df["Extracurricular_Activities"].value_counts())
    
    print("\n😴 Sleep_Hours (Horas de sueño):")
    print("Estadísticas descriptivas de horas de sueño por noche.")
    print(df["Sleep_Hours"].describe())
    
    print("\n📝 Previous_Scores (Puntajes anteriores):")
    print("Resumen estadístico de puntajes anteriores.")
    print(df["Previous_Scores"].describe())
    
    print("\n💪 Motivation_Level (Nivel de motivación):")
    print("Distribución de niveles de motivación.")
    print(df["Motivation_Level"].value_counts())
    
    print("\n🌐 Internet_Access (Acceso a internet):")
    print("Frecuencias de disponibilidad de internet.")
    print(df["Internet_Access"].value_counts())
    
    print("\n👨‍🏫 Tutoring_Sessions (Sesiones de tutoría):")
    print("Resumen estadístico del número de tutorías adicionales.")
    print(df["Tutoring_Sessions"].describe())
    
    print("\n💰 Family_Income (Ingreso familiar):")
    print("Estadísticas básicas de ingreso familiar.")
    print(df["Family_Income"].describe())
    
    print("\n🎓 Teacher_Quality (Calidad docente):")
    print("Distribución de percepción de calidad docente.")
    print(df["Teacher_Quality"].value_counts())
    
    print("\n🏫 School_Type (Tipo de escuela):")
    print("Frecuencias según tipo de escuela.")
    print(df["School_Type"].value_counts())
    
    print("\n👥 Peer_Influence (Influencia de pares):")
    print("Conteo de influencia de compañeros.")
    print(df["Peer_Influence"].value_counts())
    
    print("\n🏃‍♂️ Physical_Activity (Actividad física):")
    print("Resumen de frecuencia de actividad física.")
    print(df["Physical_Activity"].describe())
    
    print("\n🧠 Learning_Disabilities (Discapacidades de aprendizaje):")
    print("Número de estudiantes con o sin discapacidades de aprendizaje.")
    print(df["Learning_Disabilities"].value_counts())
    
    print("\n🎓 Parental_Education_Level (Nivel educativo de padres):")
    print("Frecuencias por nivel educativo de los padres.")
    print(df["Parental_Education_Level"].value_counts())
    
    print("\n📏 Distance_from_Home (Distancia desde casa):")
    print("Estadísticas de la distancia desde el hogar hasta la escuela.")
    print(df["Distance_from_Home"].describe())
    
    print("\n👤 Gender (Género):")
    print("Distribución por género.")
    print(df["Gender"].value_counts())
    
    print("\n🎯 Exam_Score (Puntuación del examen):")
    print("Estadísticas descriptivas de la variable objetivo.")
    print(df["Exam_Score"].describe())
    
    print("\n📊 ANÁLISIS DE CORRELACIONES:")
    print("Promedio de Exam_Score según género:")
    print(df.groupby("Gender")["Exam_Score"].mean().sort_values(ascending=False))
    
    print("\nCorrelación entre variables numéricas y Exam_Score:")
    # Seleccionar solo las variables numéricas para el análisis de correlación
    variables_numericas = df.select_dtypes(include=[np.number])
    correlaciones = variables_numericas.corr()["Exam_Score"].sort_values(ascending=False)
    print(correlaciones)
    
    print("\n⚠️  VALORES NULOS FINALES:")
    print("Cantidad de valores nulos en cada columna:")
    print(df.isnull().sum())

def generar_reporte_profiling(df: pd.DataFrame) -> None:
    """
    Genera reporte HTML con ydata-profiling (código exacto de la Guía 1).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos originales
    """
    imprimir_separador("GENERANDO REPORTE DE PROFILING")
    
    try:
        # Código exacto de la Guía 1 - Instalar ydata-profiling
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            print("⚠️  ydata-profiling no está instalado. Instalando...")
            import subprocess
            subprocess.run(['pip', 'install', 'ydata-profiling'], check=True)
            from ydata_profiling import ProfileReport
        
        # Código exacto de la Guía 1 - Generar informe
        print("📊 Generando reporte de profiling HTML...")
        profile = ProfileReport(df, title="Reporte rendimiento", explorative=True)
        profile.to_file(ARCHIVO_REPORTE_EDA)
        
        print(f"✅ Reporte HTML generado: {ARCHIVO_REPORTE_EDA}")
        
    except Exception as e:
        print(f"❌ Error generando reporte de profiling: {str(e)}")
        print("⚠️  Continuando sin reporte de profiling...")

def generar_reporte_basico(df: pd.DataFrame) -> None:
    """
    Genera reporte básico en HTML cuando ydata-profiling no está disponible.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos originales
    """
    try:
        # Crear HTML básico
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte Rendimiento Estudiantil</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; }}
                .stat {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 Reporte de Rendimiento Estudiantil</h1>
            <p>Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>📋 Información General</h2>
                <div class="stat">
                    <strong>Filas:</strong> {len(df):,}<br>
                    <strong>Columnas:</strong> {len(df.columns)}<br>
                    <strong>Valores nulos:</strong> {df.isnull().sum().sum():,}<br>
                    <strong>Duplicados:</strong> {df.duplicated().sum():,}
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Estadísticas Descriptivas</h2>
                {df.describe().to_html(classes='table')}
            </div>
            
            <div class="section">
                <h2>⚠️ Valores Nulos por Columna</h2>
                {df.isnull().sum().to_frame('Valores_Nulos').to_html(classes='table')}
            </div>
            
            <div class="section">
                <h2>🔢 Valores Únicos por Columna</h2>
                {df.nunique().to_frame('Valores_Unicos').to_html(classes='table')}
            </div>
            
            <div class="section">
                <h2>🎯 Correlaciones con Exam_Score</h2>
                {df.select_dtypes(include=[np.number]).corr()['Exam_Score'].sort_values(ascending=False).to_frame('Correlacion').to_html(classes='table')}
            </div>
            
        </body>
        </html>
        """
        
        with open(ARCHIVO_REPORTE_EDA, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Reporte básico HTML generado: {ARCHIVO_REPORTE_EDA}")
        
    except Exception as e:
        print(f"❌ Error generando reporte básico: {str(e)}")
        print("⚠️  Continuando sin reporte de profiling...")

def ejecutar_eda_completo() -> pd.DataFrame:
    """
    Ejecuta el análisis exploratorio de datos completo.
    
    Returns:
        pd.DataFrame: DataFrame con los datos originales
    """
    try:
        # Cargar datos originales
        df = cargar_datos_originales()
        
        # Exploración básica
        explorar_datos_basicos(df)
        
        # Análisis de patrones y tendencias
        analizar_patrones_tendencias(df)
        
        # Generar reporte de profiling
        generar_reporte_profiling(df)
        
        # Limpiar memoria
        limpiar_memoria()
        
        print(f"\n{MENSAJES['fin_eda']}")
        return df
        
    except Exception as e:
        logging.error(f"❌ Error en EDA: {str(e)}")
        raise

def main():
    """Función principal para ejecutar el EDA."""
    df = ejecutar_eda_completo()
    return df

if __name__ == "__main__":
    main()
