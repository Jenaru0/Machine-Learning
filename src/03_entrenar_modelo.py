"""
Entrenamiento y Optimización de Modelos
=======================================

Módulo que implementa exactamente el código de las GUÍAS 3 y 4 para el entrenamiento
y optimización de modelos de Machine Learning.

Incluye:
- Regresión lineal simple (Guía 3)
- Optimización de hiperparámetros con Ridge y Lasso (Guía 4)
- Validación cruzada y selección del mejor modelo
- Evaluación de métricas y serialización

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importaciones de sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importar configuración y utilidades
import importlib
config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')

# Extraer funciones y constantes necesarias
ARCHIVO_TRAIN = config.ARCHIVO_TRAIN
ARCHIVO_TEST = config.ARCHIVO_TEST
ARCHIVO_PREDICCIONES = config.ARCHIVO_PREDICCIONES
MODELO_RIDGE_PATH = config.MODELO_RIDGE_PATH
ALPHAS_RIDGE = config.ALPHAS_RIDGE
ALPHAS_LASSO = config.ALPHAS_LASSO
MAX_ITER_LASSO = config.MAX_ITER_LASSO
CV_FOLDS = config.CV_FOLDS
MEJOR_ALPHA_RIDGE = config.MEJOR_ALPHA_RIDGE
MENSAJES = config.MENSAJES
FIGSIZE_LARGE = config.FIGSIZE_LARGE

cargar_dataframe = utils.cargar_dataframe
guardar_dataframe = utils.guardar_dataframe
guardar_pickle = utils.guardar_pickle
imprimir_separador = utils.imprimir_separador
limpiar_memoria = utils.limpiar_memoria

def entrenar_regresion_lineal_simple() -> Dict[str, Any]:
    """
    Entrena regresión lineal simple siguiendo exactamente el código de la Guía 3.
    
    Returns:
        Dict[str, Any]: Diccionario con modelo y predicciones
    """
    imprimir_separador("REGRESIÓN LINEAL SIMPLE - GUÍA 3")
    
    # Código exacto de la Guía 3 - Paso 1
    # 1. Importar librerías (importadas al inicio del archivo)
    
    # Código exacto de la Guía 3 - Paso 3 (cargar datos)
    # 3. Leer los CSV en DataFrames
    df_train = cargar_dataframe(ARCHIVO_TRAIN)
    df_test = cargar_dataframe(ARCHIVO_TEST)
    
    # Código exacto de la Guía 3 - Paso 4
    # 4. Confirmar nombre de la columna objetivo
    print("Columnas en train:", df_train.columns.tolist())
    print("Columnas en test: ", df_test.columns.tolist())
    
    # Código exacto de la Guía 3 - Paso 5
    # 5. Separar features y target
    target_col = 'Exam_Score'   # ← debe estar en df_train y df_test
    X_train = df_train.drop(target_col, axis=1)
    y_train = df_train[target_col]
    
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col]
    
    # Código exacto de la Guía 3 - Paso 6
    # 6. Entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Código exacto de la Guía 3 - Paso 7
    # 7. Generar predicciones sobre el test
    y_pred = modelo.predict(X_test)
    
    # Código exacto de la Guía 3 - Paso 7.1
    # 7.1 Mostrar las primeras 5 predicciones
    print("Primeras predicciones:")
    print(pd.DataFrame(y_pred, columns=['Predicted_Exam_Score']).head())
    
    # Código exacto de la Guía 3 - Paso 7.2
    # 7.2 Estadísticos descriptivos de las predicciones
    print("\nEstadísticos descriptivos de Predicted_Exam_Score:")
    print(pd.Series(y_pred, name='Predicted_Exam_Score').describe().round(3))
    
    print("✅ Regresión lineal simple completada")
    
    return {
        'modelo': modelo,
        'predicciones': y_pred,
        'X_test': X_test,
        'y_test': y_test,
        'nombre': 'Linear Regression'
    }

def entrenar_modelos_optimizados() -> Dict[str, Any]:
    """
    Entrena modelos con optimización de hiperparámetros siguiendo exactamente el código de la Guía 4.
    
    Returns:
        Dict[str, Any]: Diccionario con resultados de todos los modelos
    """
    imprimir_separador("OPTIMIZACIÓN DE HIPERPARÁMETROS - GUÍA 4")
    
    # Código exacto de la Guía 4 - Paso 1
    # 1. Importar librerías (importadas al inicio del archivo)
    
    # Código exacto de la Guía 4 - Paso 3
    # 3. Leer los CSV en DataFrames
    df_train = cargar_dataframe(ARCHIVO_TRAIN)
    df_test = cargar_dataframe(ARCHIVO_TEST)
    
    # Código exacto de la Guía 4 - Paso 4
    # 4. Confirmar nombre de la columna objetivo
    print("Columnas en train:", df_train.columns.tolist())
    print("Columnas en test: ", df_test.columns.tolist())
    
    # Código exacto de la Guía 4 - Paso 5
    # 5. Separar features y target
    target_col = 'Exam_Score'
    X_train = df_train.drop(target_col, axis=1)
    y_train = df_train[target_col]
    
    # Ahora el test SÍ tiene Exam_Score
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col]
    
    print(f"📊 Tamaño del conjunto de entrenamiento: {X_train.shape}")
    print(f"📊 Tamaño del conjunto de prueba: {X_test.shape}")
    
    # Código exacto de la Guía 4 - Paso 6
    # 6. Preparar datos para modelos regularizados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Código exacto de la Guía 4 - Paso 7
    # 7. Definir modelos y sus hiperparámetros
    print("\n" + "="*50)
    print("🔧 ENTRENANDO Y OPTIMIZANDO MODELOS")
    print("="*50)
    
    modelos = {
        'Linear Regression': {
            'modelo': LinearRegression(),
            'params': {},
            'scaled': False
        },
        'Ridge': {
            'modelo': Ridge(random_state=42),
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            'scaled': True
        },
        'Lasso': {
            'modelo': Lasso(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                'max_iter': [1000, 2000, 3000]
            },
            'scaled': True
        }
    }
    
    # Código exacto de la Guía 4 - Paso 8
    # 8. Entrenar y optimizar cada modelo
    resultados = {}
    
    for nombre, config in modelos.items():
        print(f"\n🔹 Entrenando {nombre}...")
        
        # Seleccionar datos (escalados o no)
        X_train_use = X_train_scaled if config['scaled'] else X_train
        X_test_use = X_test_scaled if config['scaled'] else X_test
        
        if config['params']:  # Si tiene hiperparámetros que optimizar
            # Grid Search con validación cruzada
            grid_search = GridSearchCV(
                config['modelo'],
                config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train_use, y_train)
            mejor_modelo = grid_search.best_estimator_
            print(f"   📊 Mejores hiperparámetros: {grid_search.best_params_}")
            print(f"   📈 Mejor score CV: {-grid_search.best_score_:.3f}")
        else:  # Regresión lineal simple
            mejor_modelo = config['modelo']
            mejor_modelo.fit(X_train_use, y_train)
        
        # Predicciones
        y_pred = mejor_modelo.predict(X_test_use)
        
        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardar resultados
        resultados[nombre] = {
            'modelo': mejor_modelo,
            'predicciones': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'X_test_use': X_test_use,
            'scaler': scaler if config['scaled'] else None
        }
        
        print(f"   ✅ R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    print("✅ Optimización de modelos completada")
    return resultados, y_test

def comparar_modelos(resultados: Dict[str, Any]) -> pd.DataFrame:
    """
    Compara modelos siguiendo exactamente el código de la Guía 4.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        
    Returns:
        pd.DataFrame: DataFrame con comparación de modelos
    """
    imprimir_separador("COMPARACIÓN DE MODELOS")
    
    # Código exacto de la Guía 4 - Paso 9
    # 9. COMPARACIÓN DE MODELOS
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*60)
    
    # Crear tabla comparativa
    df_comparacion = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'R²': [res['r2'] for res in resultados.values()],
        'RMSE': [res['rmse'] for res in resultados.values()],
        'MAE': [res['mae'] for res in resultados.values()],
        'MSE': [res['mse'] for res in resultados.values()]
    })
    
    # Ordenar por R² (descendente)
    df_comparacion = df_comparacion.sort_values('R²', ascending=False)
    print(df_comparacion.round(6))
    
    # Identificar el mejor modelo
    mejor_modelo_nombre = df_comparacion.iloc[0]['Modelo']
    print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
    print(f"   📈 R² = {df_comparacion.iloc[0]['R²']:.4f}")
    print(f"   📉 RMSE = {df_comparacion.iloc[0]['RMSE']:.4f}")
    
    return df_comparacion

def generar_visualizaciones(resultados: Dict[str, Any], y_test: pd.Series, 
                          df_comparacion: pd.DataFrame) -> None:
    """
    Genera visualizaciones siguiendo exactamente el código de la Guía 4.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        y_test (pd.Series): Valores reales de test
        df_comparacion (pd.DataFrame): DataFrame con comparación de modelos
    """
    imprimir_separador("GENERANDO VISUALIZACIONES")
    
    # Código exacto de la Guía 4 - Paso 10
    # 10. Visualización comparativa
    plt.figure(figsize=(18, 12))
    
    # Gráfico 1: Comparación de métricas
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(df_comparacion))
    plt.bar(x_pos, df_comparacion['R²'], alpha=0.7, color=['gold', 'silver', 'brown'])
    plt.xlabel('Modelos')
    plt.ylabel('R²')
    plt.title('Comparación R²')
    plt.xticks(x_pos, df_comparacion['Modelo'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.bar(x_pos, df_comparacion['RMSE'], alpha=0.7, color=['gold', 'silver', 'brown'])
    plt.xlabel('Modelos')
    plt.ylabel('RMSE')
    plt.title('Comparación RMSE')
    plt.xticks(x_pos, df_comparacion['Modelo'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Gráficos 3-5: Predicciones vs Reales para cada modelo
    for i, (nombre, res) in enumerate(resultados.items()):
        plt.subplot(2, 3, i+3)
        plt.scatter(y_test, res['predicciones'], alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(f'{nombre}\nR² = {res["r2"]:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones generadas")

def analizar_coeficientes(resultados: Dict[str, Any], X_train: pd.DataFrame) -> None:
    """
    Analiza coeficientes siguiendo exactamente el código de la Guía 4.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        X_train (pd.DataFrame): DataFrame de entrenamiento
    """
    imprimir_separador("ANÁLISIS DE COEFICIENTES")
    
    # Código exacto de la Guía 4 - Paso 11
    # 11. Análisis de coeficientes (para Ridge y Lasso)
    print("\n" + "="*50)
    print("🔍 ANÁLISIS DE COEFICIENTES")
    print("="*50)
    
    feature_names = X_train.columns.tolist()
    
    for nombre in ['Ridge', 'Lasso']:
        if nombre in resultados:
            modelo = resultados[nombre]['modelo']
            coeficientes = modelo.coef_
            
            # Crear DataFrame con coeficientes
            df_coef = pd.DataFrame({
                'Feature': feature_names,
                'Coeficiente': coeficientes,
                'Abs_Coeficiente': np.abs(coeficientes)
            }).sort_values('Abs_Coeficiente', ascending=False)
            
            print(f"\n{nombre} - Top 5 características más importantes:")
            print(df_coef.head().round(6))
            
            # Contar coeficientes cero en Lasso
            if nombre == 'Lasso':
                coef_cero = np.sum(np.abs(coeficientes) < 1e-10)
                print(f"   📊 Características eliminadas por Lasso: {coef_cero}/{len(coeficientes)}")
    
    print("✅ Análisis de coeficientes completado")

def evaluar_mejor_modelo(resultados: Dict[str, Any], y_test: pd.Series, 
                        mejor_modelo_nombre: str) -> None:
    """
    Evalúa el mejor modelo siguiendo exactamente el código de la Guía 4.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        y_test (pd.Series): Valores reales de test
        mejor_modelo_nombre (str): Nombre del mejor modelo
    """
    imprimir_separador(f"EVALUACIÓN DETALLADA: {mejor_modelo_nombre}")
    
    # Código exacto de la Guía 4 - Paso 13
    # 13. EVALUACIÓN DETALLADA DEL MEJOR MODELO
    mejor_resultado = resultados[mejor_modelo_nombre]
    y_pred_final = mejor_resultado['predicciones']
    
    print("\n" + "="*50)
    print(f"📊 EVALUACIÓN DETALLADA: {mejor_modelo_nombre}")
    print("="*50)
    
    # Usar las predicciones del mejor modelo
    mse = mean_squared_error(y_test, y_pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_final)
    r2 = r2_score(y_test, y_pred_final)
    
    print(f"MSE (Error Cuadrático Medio): {mse:.3f}")
    print(f"RMSE (Raíz del ECM): {rmse:.3f}")
    print(f"MAE (Error Absoluto Medio): {mae:.3f}")
    print(f"R² (Coeficiente de Determinación): {r2:.3f}")
    
    # Interpretación del R²
    if r2 >= 0.8:
        interpretacion = "Excelente"
    elif r2 >= 0.6:
        interpretacion = "Bueno"
    elif r2 >= 0.4:
        interpretacion = "Moderado"
    else:
        interpretacion = "Pobre"
    
    print(f"📋 Interpretación del modelo: {interpretacion} (R² = {r2:.3f})")
    
    # Código exacto de la Guía 4 - Paso 14
    # 14. Comparación de predicciones vs valores reales (mejor modelo)
    print("\n" + "="*50)
    print("🔍 COMPARACIÓN: PREDICCIONES vs VALORES REALES")
    print("="*50)
    
    # Crear DataFrame con comparación
    comparacion = pd.DataFrame({
        'Valor_Real': y_test,
        'Prediccion': np.round(y_pred_final, 6),
        'Error_Absoluto': np.abs(y_test - y_pred_final)
    })
    
    print("Primeras 10 comparaciones:")
    print(comparacion.head(10))
    
    print(f"\nError promedio: {comparacion['Error_Absoluto'].mean():.3f}")
    print(f"Error máximo: {comparacion['Error_Absoluto'].max():.3f}")
    print(f"Error mínimo: {comparacion['Error_Absoluto'].min():.3f}")
    
    # Código exacto de la Guía 4 - Paso 16
    # 16. Estadísticos descriptivos
    print("\n" + "="*50)
    print("📈 ESTADÍSTICOS DESCRIPTIVOS")
    print("="*50)
    
    print("Valores Reales:")
    print(y_test.describe().round(6))
    
    print(f"\nPredicciones ({mejor_modelo_nombre}):")
    print(pd.Series(y_pred_final, name='Exam_Score').describe().round(6))
    
    print("✅ Evaluación detallada completada")

def guardar_mejor_modelo(resultados: Dict[str, Any], mejor_modelo_nombre: str) -> None:
    """
    Guarda el mejor modelo siguiendo las especificaciones del proyecto.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        mejor_modelo_nombre (str): Nombre del mejor modelo
    """
    imprimir_separador("GUARDANDO MEJOR MODELO")
    
    mejor_resultado = resultados[mejor_modelo_nombre]
    mejor_modelo = mejor_resultado['modelo']
    
    # Guardar modelo
    guardar_pickle(mejor_modelo, MODELO_RIDGE_PATH)
    
    # Si hay scaler, guardarlo también
    if mejor_resultado.get('scaler'):
        scaler_path = MODELO_RIDGE_PATH.parent / "scaler.pkl"
        guardar_pickle(mejor_resultado['scaler'], scaler_path)
    
    print(f"✅ Mejor modelo guardado: {mejor_modelo_nombre}")

def generar_predicciones_finales(resultados: Dict[str, Any], mejor_modelo_nombre: str) -> None:
    """
    Genera predicciones finales siguiendo exactamente el código de la Guía 4.
    
    Args:
        resultados (Dict[str, Any]): Resultados de todos los modelos
        mejor_modelo_nombre (str): Nombre del mejor modelo
    """
    imprimir_separador("GENERANDO PREDICCIONES FINALES")
    
    # Código exacto de la Guía 4 - Paso 17
    # 17. Guardar predicciones del mejor modelo
    mejor_resultado = resultados[mejor_modelo_nombre]
    y_pred_final = mejor_resultado['predicciones']
    
    # Cargar datos de test para obtener las features
    df_test = cargar_dataframe(ARCHIVO_TEST)
    X_test = df_test.drop('Exam_Score', axis=1)
    
    df_pred = X_test.copy()
    df_pred['Exam_Score'] = np.round(y_pred_final, 6)
    
    guardar_dataframe(df_pred, ARCHIVO_PREDICCIONES)
    
    print(f"✅ Predicciones finales guardadas usando {mejor_modelo_nombre}")

def ejecutar_entrenamiento_completo() -> Dict[str, Any]:
    """
    Ejecuta el entrenamiento completo de modelos.
    
    Returns:
        Dict[str, Any]: Resultados del entrenamiento
    """
    try:
        print(MENSAJES['inicio_entrenamiento'])
        
        # Entrenar regresión lineal simple (Guía 3)
        resultado_lineal = entrenar_regresion_lineal_simple()
        
        # Entrenar modelos optimizados (Guía 4)
        resultados, y_test = entrenar_modelos_optimizados()
        
        # Comparar modelos
        df_comparacion = comparar_modelos(resultados)
        
        # Generar visualizaciones
        generar_visualizaciones(resultados, y_test, df_comparacion)
        
        # Analizar coeficientes
        df_train = cargar_dataframe(ARCHIVO_TRAIN)
        X_train = df_train.drop('Exam_Score', axis=1)
        analizar_coeficientes(resultados, X_train)
        
        # Evaluar mejor modelo
        mejor_modelo_nombre = df_comparacion.iloc[0]['Modelo']
        evaluar_mejor_modelo(resultados, y_test, mejor_modelo_nombre)
        
        # Guardar mejor modelo
        guardar_mejor_modelo(resultados, mejor_modelo_nombre)
        
        # Generar predicciones finales
        generar_predicciones_finales(resultados, mejor_modelo_nombre)
        
        # Limpiar memoria
        limpiar_memoria()
        
        print(f"\n{MENSAJES['fin_entrenamiento']}")
        
        return {
            'resultados': resultados,
            'comparacion': df_comparacion,
            'mejor_modelo': mejor_modelo_nombre,
            'y_test': y_test
        }
        
    except Exception as e:
        logging.error(f"❌ Error en entrenamiento: {str(e)}")
        raise

def main():
    """Función principal para ejecutar el entrenamiento."""
    resultados = ejecutar_entrenamiento_completo()
    return resultados

if __name__ == "__main__":
    # Importar métricas necesarias (ya importadas al inicio)
    main()
