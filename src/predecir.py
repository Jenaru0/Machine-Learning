"""
Generación de Predicciones Finales
==================================

Módulo para generar predicciones finales utilizando el mejor modelo entrenado.
Carga el modelo serializado y genera predicciones sobre datos nuevos.

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Importar configuración y utilidades
from .config import (
    ARCHIVO_TEST,
    ARCHIVO_PREDICCIONES,
    MODELO_RIDGE_PATH,
    VARIABLE_OBJETIVO,
    MENSAJES
)
from .utils import (
    cargar_dataframe,
    guardar_dataframe,
    cargar_pickle,
    imprimir_separador,
    limpiar_memoria
)

def cargar_modelo_entrenado():
    """
    Carga el mejor modelo entrenado desde el archivo pickle.
    
    Returns:
        modelo: Modelo entrenado
    """
    try:
        modelo = cargar_pickle(MODELO_RIDGE_PATH)
        print(f"✅ Modelo cargado exitosamente: {MODELO_RIDGE_PATH.name}")
        return modelo
    except FileNotFoundError:
        print(f"❌ No se encontró el modelo en: {MODELO_RIDGE_PATH}")
        print("⚠️  Ejecute primero el entrenamiento de modelos")
        raise
    except Exception as e:
        print(f"❌ Error cargando el modelo: {str(e)}")
        raise

def cargar_scaler_si_existe():
    """
    Carga el scaler si existe (para modelos regularizados).
    
    Returns:
        scaler: Scaler entrenado o None si no existe
    """
    scaler_path = MODELO_RIDGE_PATH.parent / "scaler.pkl"
    
    try:
        if scaler_path.exists():
            scaler = cargar_pickle(scaler_path)
            print(f"✅ Scaler cargado exitosamente: {scaler_path.name}")
            return scaler
        else:
            print("ℹ️  No se encontró scaler (modelo no regularizado)")
            return None
    except Exception as e:
        print(f"❌ Error cargando scaler: {str(e)}")
        return None

def preparar_datos_prediccion(archivo_datos: Optional[str] = None) -> pd.DataFrame:
    """
    Prepara los datos para predicción.
    
    Args:
        archivo_datos (str, optional): Ruta al archivo de datos. 
                                     Si None, usa el archivo de test por defecto.
    
    Returns:
        pd.DataFrame: DataFrame con los datos preparados
    """
    if archivo_datos is None:
        archivo_datos = ARCHIVO_TEST
    
    # Cargar datos
    df = cargar_dataframe(archivo_datos)
    
    # Separar features (eliminar variable objetivo si existe)
    if VARIABLE_OBJETIVO in df.columns:
        X = df.drop(VARIABLE_OBJETIVO, axis=1)
        y_real = df[VARIABLE_OBJETIVO]
        print(f"📊 Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
        print(f"ℹ️  Variable objetivo encontrada para comparación")
        return X, y_real
    else:
        print(f"📊 Datos cargados: {df.shape[0]} filas, {df.shape[1]} características")
        print(f"ℹ️  No se encontró variable objetivo")
        return df, None

def generar_predicciones(modelo, X: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    Genera predicciones usando el modelo entrenado.
    
    Args:
        modelo: Modelo entrenado
        X (pd.DataFrame): Datos de entrada
        scaler: Scaler entrenado (opcional)
    
    Returns:
        np.ndarray: Predicciones generadas
    """
    try:
        # Aplicar escalado si es necesario
        if scaler is not None:
            X_scaled = scaler.transform(X)
            print("✅ Datos escalados aplicados")
        else:
            X_scaled = X
            print("ℹ️  Datos sin escalar utilizados")
        
        # Generar predicciones
        predicciones = modelo.predict(X_scaled)
        
        print(f"✅ Predicciones generadas: {len(predicciones)} valores")
        
        return predicciones
        
    except Exception as e:
        print(f"❌ Error generando predicciones: {str(e)}")
        raise

def mostrar_estadisticas_predicciones(predicciones: np.ndarray) -> None:
    """
    Muestra estadísticas de las predicciones generadas.
    
    Args:
        predicciones (np.ndarray): Predicciones generadas
    """
    print("\n📈 ESTADÍSTICAS DE PREDICCIONES:")
    print("-" * 40)
    
    stats = pd.Series(predicciones, name='Predicciones')
    print(stats.describe().round(3))
    
    print(f"\n📊 Rango de predicciones: {predicciones.min():.3f} - {predicciones.max():.3f}")
    print(f"📊 Media: {predicciones.mean():.3f}")
    print(f"📊 Desviación estándar: {predicciones.std():.3f}")

def comparar_con_valores_reales(predicciones: np.ndarray, y_real: pd.Series) -> None:
    """
    Compara las predicciones con los valores reales si están disponibles.
    
    Args:
        predicciones (np.ndarray): Predicciones generadas
        y_real (pd.Series): Valores reales
    """
    if y_real is None:
        print("ℹ️  No hay valores reales para comparar")
        return
    
    print("\n🔍 COMPARACIÓN CON VALORES REALES:")
    print("-" * 45)
    
    # Calcular métricas
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_real, predicciones)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, predicciones)
    r2 = r2_score(y_real, predicciones)
    
    print(f"📊 R² (Coeficiente de Determinación): {r2:.4f}")
    print(f"📊 RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
    print(f"📊 MAE (Error Absoluto Medio): {mae:.4f}")
    print(f"📊 MSE (Error Cuadrático Medio): {mse:.4f}")
    
    # Mostrar primeras comparaciones
    comparacion = pd.DataFrame({
        'Valor_Real': y_real.values,
        'Predicción': predicciones,
        'Error_Absoluto': np.abs(y_real.values - predicciones)
    })
    
    print(f"\n📋 PRIMERAS 10 COMPARACIONES:")
    print(comparacion.head(10).round(4))
    
    print(f"\n📊 Error promedio: {comparacion['Error_Absoluto'].mean():.4f}")
    print(f"📊 Error máximo: {comparacion['Error_Absoluto'].max():.4f}")
    print(f"📊 Error mínimo: {comparacion['Error_Absoluto'].min():.4f}")

def guardar_predicciones(X: pd.DataFrame, predicciones: np.ndarray, 
                        archivo_salida: Optional[str] = None) -> None:
    """
    Guarda las predicciones en un archivo CSV.
    
    Args:
        X (pd.DataFrame): Datos de entrada
        predicciones (np.ndarray): Predicciones generadas
        archivo_salida (str, optional): Archivo de salida. Si None, usa el archivo por defecto.
    """
    if archivo_salida is None:
        archivo_salida = ARCHIVO_PREDICCIONES
    
    try:
        # Crear DataFrame con predicciones
        df_predicciones = X.copy()
        df_predicciones[VARIABLE_OBJETIVO] = np.round(predicciones, 6)
        
        # Guardar archivo
        guardar_dataframe(df_predicciones, archivo_salida)
        
        print(f"✅ Predicciones guardadas en: {archivo_salida}")
        print(f"📊 Archivo generado con {len(df_predicciones)} predicciones")
        
    except Exception as e:
        print(f"❌ Error guardando predicciones: {str(e)}")
        raise

def ejecutar_prediccion_completa(archivo_datos: Optional[str] = None,
                                archivo_salida: Optional[str] = None) -> np.ndarray:
    """
    Ejecuta el proceso completo de predicción.
    
    Args:
        archivo_datos (str, optional): Archivo de datos para predicción
        archivo_salida (str, optional): Archivo para guardar predicciones
    
    Returns:
        np.ndarray: Predicciones generadas
    """
    try:
        print(MENSAJES['inicio_prediccion'])
        
        # Cargar modelo entrenado
        modelo = cargar_modelo_entrenado()
        
        # Cargar scaler si existe
        scaler = cargar_scaler_si_existe()
        
        # Preparar datos
        datos_resultado = preparar_datos_prediccion(archivo_datos)
        if isinstance(datos_resultado, tuple):
            X, y_real = datos_resultado
        else:
            X, y_real = datos_resultado, None
        
        # Generar predicciones
        predicciones = generar_predicciones(modelo, X, scaler)
        
        # Mostrar estadísticas
        mostrar_estadisticas_predicciones(predicciones)
        
        # Comparar con valores reales si están disponibles
        if y_real is not None:
            comparar_con_valores_reales(predicciones, y_real)
        
        # Guardar predicciones
        guardar_predicciones(X, predicciones, archivo_salida)
        
        # Limpiar memoria
        limpiar_memoria()
        
        print(f"\n{MENSAJES['fin_prediccion']}")
        
        return predicciones
        
    except Exception as e:
        logging.error(f"❌ Error en predicción: {str(e)}")
        raise

def predecir_nuevos_datos(datos_nuevos: pd.DataFrame) -> np.ndarray:
    """
    Predice sobre datos nuevos proporcionados directamente.
    
    Args:
        datos_nuevos (pd.DataFrame): DataFrame con nuevos datos
        
    Returns:
        np.ndarray: Predicciones generadas
    """
    imprimir_separador("PREDICCIÓN SOBRE DATOS NUEVOS")
    
    try:
        # Cargar modelo y scaler
        modelo = cargar_modelo_entrenado()
        scaler = cargar_scaler_si_existe()
        
        # Generar predicciones
        predicciones = generar_predicciones(modelo, datos_nuevos, scaler)
        
        # Mostrar estadísticas
        mostrar_estadisticas_predicciones(predicciones)
        
        print("✅ Predicción sobre datos nuevos completada")
        return predicciones
        
    except Exception as e:
        logging.error(f"❌ Error en predicción de datos nuevos: {str(e)}")
        raise

def main():
    """Función principal para ejecutar las predicciones."""
    predicciones = ejecutar_prediccion_completa()
    return predicciones

if __name__ == "__main__":
    main()
