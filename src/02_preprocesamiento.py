"""
Preprocesamiento de Datos
========================

Módulo que implementa exactamente el código de la GUÍA 2 para el preprocesamiento
de datos del proyecto de rendimiento académico.

Incluye funciones para:
- Limpieza de datos
- Transformación de variables
- Ingeniería de características
- División train/test

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

# Importar configuración y utilidades
import importlib
config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')

# Extraer funciones y constantes necesarias
PROCESADOS_DIR = config.PROCESADOS_DIR
COLUMNAS_A_ELIMINAR = config.COLUMNAS_A_ELIMINAR
VARIABLES_CATEGORICAS = config.VARIABLES_CATEGORICAS
VARIABLES_NUMERICAS = config.VARIABLES_NUMERICAS
VARIABLE_OBJETIVO = config.VARIABLE_OBJETIVO
ARCHIVO_DATOS_TRANSFORMADOS = config.ARCHIVO_DATOS_TRANSFORMADOS
ARCHIVO_TRAIN = config.ARCHIVO_TRAIN
ARCHIVO_TEST = config.ARCHIVO_TEST
TEST_SIZE = config.TEST_SIZE
RANDOM_STATE = config.RANDOM_STATE
MENSAJES = config.MENSAJES

cargar_dataframe = utils.cargar_dataframe
guardar_dataframe = utils.guardar_dataframe
imprimir_separador = utils.imprimir_separador
limpiar_memoria = utils.limpiar_memoria

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos siguiendo exactamente el código de la Guía 2.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    imprimir_separador("LIMPIEZA DE DATOS")
    
    # Código exacto de la Guía 2 - Paso 2.2 y 2.3
    print("🗑️  Eliminando columnas con baja correlación...")
    
    # 2.2 Columnas a eliminar por baja correlación
    cols_a_eliminar = [
        'Gender',
        'School_Type',
        'Sleep_Hours',
        'Physical_Activity',
        'Internet_Access'
    ]
    
    # 2.3 Eliminar esas columnas
    df_limpio = df.drop(columns=cols_a_eliminar)
    
    # 2.4 Verificar
    print("Columnas tras eliminación:", df_limpio.columns.tolist())
    
    # Código exacto de la Guía 2 - Paso 3.1 y 3.2
    # 3.1 Convertir cualquier '' (cadena vacía) a np.nan
    df_limpio.replace({'': np.nan}, inplace=True)
    
    # 3.2 Mostrar conteo de nulos
    print("\nNulos por columna tras reemplazo de '':")
    print(df_limpio.isnull().sum())
    
    # Código exacto de la Guía 2 - Paso 4.1 y 4.2
    # 4.1 Listar columnas categóricas:
    cols_categoricas = [
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
    
    # 4.2 Convertirlas a tipo category
    for col in cols_categoricas:
        df_limpio[col] = df_limpio[col].astype('category')
    
    # 4.3 Verificar tipos de datos
    print("\nTipos tras categorizar:")
    print(df_limpio.dtypes)
    
    # Código exacto de la Guía 2 - Paso 5.1, 5.2, 5.3 y 5.4
    # 5.1 Imputar Teacher_Quality con su moda
    modo_tq = df_limpio['Teacher_Quality'].mode()[0]
    df_limpio['Teacher_Quality'] = df_limpio['Teacher_Quality'].fillna(modo_tq)
    
    # 5.2 Imputar Parental_Education_Level con su moda
    modo_pel = df_limpio['Parental_Education_Level'].mode()[0]
    df_limpio['Parental_Education_Level'] = df_limpio['Parental_Education_Level'].fillna(modo_pel)
    
    # 5.3 Imputar Distance_from_Home con su moda
    modo_dist = df_limpio['Distance_from_Home'].mode()[0]
    df_limpio['Distance_from_Home'] = df_limpio['Distance_from_Home'].fillna(modo_dist)
    
    # 5.4 Verificar que no queden nulos
    print("\nNulos tras imputación:")
    print(df_limpio.isnull().sum())
    
    print("✅ Limpieza de datos completada")
    return df_limpio

def transformar_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma características siguiendo exactamente el código de la Guía 2.
    
    Args:
        df (pd.DataFrame): DataFrame limpio
        
    Returns:
        pd.DataFrame: DataFrame transformado
    """
    imprimir_separador("TRANSFORMACIÓN DE CARACTERÍSTICAS")
    
    # Código exacto de la Guía 2 - Paso 6 (importaciones ya realizadas)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    # Código exacto de la Guía 2 - Paso 7.1 y 7.2
    # 7.1 X = todas las columnas excepto 'Exam_Score'
    X = df.drop('Exam_Score', axis=1)
    
    # 7.2 y = Serie 'Exam_Score'
    y = df['Exam_Score']
    
    print("\nDimensiones de X:", X.shape)
    print("Dimensiones de y:", y.shape)
    
    # Código exacto de la Guía 2 - Paso 8.1 y 8.2
    # 8.1: Variables numéricas a escalar (4 columnas)
    numerical_features = [
        'Hours_Studied',
        'Attendance',
        'Previous_Scores',
        'Tutoring_Sessions'
    ]
    
    # 8.2: Variables categóricas a codificar (10 columnas)
    categorical_features = [
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
    
    print("Variables numéricas (a escalar):", numerical_features)
    print("Variables categóricas (a codificar):", categorical_features)
    
    # Código exacto de la Guía 2 - Paso 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
    # 9.1: Crear transformadores
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # 9.2: Definir ColumnTransformer combinado
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 9.3: Aplicar fit_transform sobre X
    X_transformed = preprocessor.fit_transform(X)
    
    # 9.4: Obtener nombres de columnas generadas por OneHotEncoder
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # 9.5: Concatenar nombres numéricos + nombres one-hot
    all_feature_names = numerical_features + list(onehot_feature_names)
    
    # 9.6: Construir DataFrame resultante con encabezados
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    # 9.7: Verificar que todas las columnas son numéricas
    print("Tipos de dato de X_transformed_df:")
    print(X_transformed_df.dtypes.unique())
    
    # 9.8: Mostrar dimensiones y primeras filas
    print(f"\n=== Paso 9: X_transformed_df ===\nDimensiones: {X_transformed_df.shape}")
    print(X_transformed_df.head())
    
    print("✅ Transformación de características completada")
    return X_transformed_df, y

def crear_variables_ingenieria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de ingeniería siguiendo exactamente el código de la Guía 2.
    
    Args:
        df (pd.DataFrame): DataFrame limpio
        
    Returns:
        pd.DataFrame: DataFrame con variables de ingeniería
    """
    imprimir_separador("INGENIERÍA DE CARACTERÍSTICAS")
    
    # Código exacto de la Guía 2 - Paso 11.1, 11.2, 11.3
    # 11.1 Study_Efficiency = Hours_Studied / (Previous_Scores + 1)
    df['Study_Efficiency'] = df['Hours_Studied'] / (df['Previous_Scores'] + 1)
    
    # 11.2 High_Support = 1 si Tutoring_Sessions > mediana y Access_to_Resources == "High"
    mediana_tut = df['Tutoring_Sessions'].median()
    df['High_Support'] = (
        (df['Tutoring_Sessions'] > mediana_tut) &
        (df['Access_to_Resources'] == 'High')
    ).astype(int)
    
    # 11.3 Family_Education_Support = 1 si Parental_Education_Level == 'Postgraduate' y Parental_Involvement == 'High'
    df['Family_Education_Support'] = (
        (df['Parental_Education_Level'] == 'Postgraduate') &
        (df['Parental_Involvement'] == 'High')
    ).astype(int)
    
    print("\n=== PASO 11: Nuevas variables creadas ===")
    print("* Study_Efficiency (float)")
    print("* High_Support (0/1)")
    print("* Family_Education_Support (0/1)\n")
    
    # Verificar brevemente sus estadísticas
    print("Study_Efficiency descr.:\n", df['Study_Efficiency'].describe())
    print("\nHigh_Support conteo:\n", df['High_Support'].value_counts())
    print("\nFamily_Education_Support conteo:\n", df['Family_Education_Support'].value_counts())
    
    print("✅ Variables de ingeniería creadas")
    return df

def analizar_distribucion_objetivo(df: pd.DataFrame) -> None:
    """
    Analiza la distribución de la variable objetivo siguiendo exactamente el código de la Guía 2.
    
    Args:
        df (pd.DataFrame): DataFrame con variable objetivo
    """
    imprimir_separador("ANÁLISIS DE DISTRIBUCIÓN DE EXAM_SCORE")
    
    # Código exacto de la Guía 2 - Paso 12
    print("\n" + "="*60)
    print("PASO 12: Distribución de Exam_Score")
    print("="*60)
    print(f"Rango: {df['Exam_Score'].min():.1f} - {df['Exam_Score'].max():.1f}")
    print(f"Media: {df['Exam_Score'].mean():.2f}")
    print(f"Mediana: {df['Exam_Score'].median():.2f}")
    print(f"Desv. estándar: {df['Exam_Score'].std():.2f}")
    
    # Deciles (aprox. 10 % de filas por decil)
    try:
        deciles = pd.qcut(df['Exam_Score'], q=10, labels=[f'D{i}' for i in range(1,11)])
        print("\nConteo por deciles:")
        print(deciles.value_counts().sort_index())
    except Exception as e:
        print("Error al calcular deciles:", e)
    
    print("\nNo se aplica SMOTE (regresión continua); mantenemos distribución original.")

def crear_dataset_numerico_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea el dataset numérico final siguiendo exactamente el código de la Guía 2.
    
    Args:
        df (pd.DataFrame): DataFrame con todas las transformaciones
        
    Returns:
        pd.DataFrame: DataFrame numérico final
    """
    imprimir_separador("CREACIÓN DE DATASET NUMÉRICO FINAL")
    
    # Código exacto de la Guía 2 - Paso 13.1, 13.2, 13.3, 13.4, 13.5
    # 13.1: X_full = todas las 17 columnas de features
    X_full = df.drop('Exam_Score', axis=1)
    
    # 13.2: Convertir cada columna categórica a códigos numéricos
    # Lista de categóricas (las 10 originales)
    cat_cols = [
        'Access_to_Resources',
        'Family_Income',
        'Extracurricular_Activities',
        'Distance_from_Home',
        'Learning_Disabilities',
        'Motivation_Level',
        'Parental_Education_Level',
        'Parental_Involvement',
        'Peer_Influence',
        'Teacher_Quality'
    ]
    
    df_num = X_full.copy()
    for col in cat_cols:
        df_num[col] = df_num[col].cat.codes  # convierte cada nivel a un entero
    
    # 13.3: Añadir las 3 variables de ingeniería (ya son numéricas) y las 4 numéricas escaladas
    #       En df_num ya están las 4 escaladas (porque df_num hereda df),
    #       y las 3 de ingeniería se preservan también.
    
    # 13.4: Concatenar con la etiqueta 'Exam_Score'
    df_numeric_final = pd.concat([df_num.reset_index(drop=True),
                                  df['Exam_Score'].reset_index(drop=True)], axis=1)
    
    print("\n=== PASO 13: DataFrame final (solo numérico) ===")
    print(f"Dimensiones: {df_numeric_final.shape}")
    print("Columnas (todas numéricas):")
    print(df_numeric_final.columns.tolist())
    print("\nPrimeras 5 filas:")
    print(df_numeric_final.head())
    
    # 13.5: Guardar a CSV
    guardar_dataframe(df_numeric_final, ARCHIVO_DATOS_TRANSFORMADOS)
    
    print("✅ Dataset numérico final creado")
    return df_numeric_final

def dividir_train_test(df_numeric_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train y test siguiendo exactamente el código de la Guía 2.
    
    Args:
        df_numeric_final (pd.DataFrame): DataFrame numérico final
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de train y test
    """
    imprimir_separador("DIVISIÓN TRAIN/TEST")
    
    # Código exacto de la Guía 2 - Paso 14.1, 14.2, 14.3, 14.4, 14.5
    from sklearn.model_selection import train_test_split
    
    # 14.1: Separar X_final e y_final
    X_final = df_numeric_final.drop('Exam_Score', axis=1)
    y_final = df_numeric_final['Exam_Score']
    
    # 14.2: train 70% / test 30%
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final,
        test_size=0.30,
        random_state=42
    )
    
    # 14.3: DataFrame de entrenamiento (features + etiqueta)
    df_train = pd.concat([X_train.reset_index(drop=True),
                          y_train.reset_index(drop=True)], axis=1)
    
    # 14.4: DataFrame de prueba (features + etiqueta para poder evaluar)
    df_test = pd.concat([X_test.reset_index(drop=True),
                         y_test.reset_index(drop=True)], axis=1)
    
    # 14.5: Guardar a CSV
    guardar_dataframe(df_train, ARCHIVO_TRAIN)
    guardar_dataframe(df_test, ARCHIVO_TEST)
    
    print("\n=== PASO 14: Conjuntos finales guardados ===")
    print(f"• train_student_performance.csv = {df_train.shape}")
    print(f"• test_student_performance.csv = {df_test.shape}")
    
    print("✅ División train/test completada")
    return df_train, df_test

def ejecutar_preprocesamiento_completo(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el preprocesamiento completo siguiendo exactamente el código de la Guía 2.
    
    Args:
        df_original (pd.DataFrame): DataFrame original
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de train y test
    """
    try:
        print(MENSAJES['inicio_preproceso'])
        
        # Limpieza de datos
        df_limpio = limpiar_datos(df_original)
        
        # Crear variables de ingeniería
        df_con_ingenieria = crear_variables_ingenieria(df_limpio)
        
        # Analizar distribución de la variable objetivo
        analizar_distribucion_objetivo(df_con_ingenieria)
        
        # Crear dataset numérico final
        df_numeric_final = crear_dataset_numerico_final(df_con_ingenieria)
        
        # Dividir en train y test
        df_train, df_test = dividir_train_test(df_numeric_final)
        
        # Limpiar memoria
        limpiar_memoria()
        
        print(f"\n{MENSAJES['fin_preproceso']}")
        return df_train, df_test
        
    except Exception as e:
        logging.error(f"❌ Error en preprocesamiento: {str(e)}")
        raise

def cargar_datos_entrenamiento() -> tuple:
    """
    Carga los datos de entrenamiento y prueba procesados.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    from sklearn.preprocessing import StandardScaler
    
    # Cargar datasets procesados
    train_df = cargar_dataframe(PROCESADOS_DIR / ARCHIVO_TRAIN)
    test_df = cargar_dataframe(PROCESADOS_DIR / ARCHIVO_TEST)
    
    # Separar características y variable objetivo
    X_train = train_df.drop(columns=[VARIABLE_OBJETIVO])
    y_train = train_df[VARIABLE_OBJETIVO]
    X_test = test_df.drop(columns=[VARIABLE_OBJETIVO])
    y_test = test_df[VARIABLE_OBJETIVO]
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    """Función principal para ejecutar el preprocesamiento."""
    eda = importlib.import_module('src.01_eda')
    cargar_datos_originales = eda.cargar_datos_originales
    
    # Cargar datos originales
    df_original = cargar_datos_originales()
    
    # Ejecutar preprocesamiento
    df_train, df_test = ejecutar_preprocesamiento_completo(df_original)
    
    return df_train, df_test

if __name__ == "__main__":
    main()
