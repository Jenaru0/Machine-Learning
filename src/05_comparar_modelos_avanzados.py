"""
Comparación Avanzada de Modelos de Machine Learning
==================================================

Módulo que amplía la comparación de modelos para cumplir completamente 
la rúbrica del proyecto. Incluye modelos estado del arte y redes neuronales.

Modelos implementados:
- Linear, Ridge, Lasso (ya implementados en 03_)
- Random Forest Regressor
- Gradient Boosting Regressor  
- Support Vector Regression (SVR)
- XGBoost Regressor (estado del arte)
- Multi-Layer Perceptron (Red Neuronal)

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, Tuple, Any

# Modelos de scikit-learn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Importaciones con nueva estructura numerada
import importlib
config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')

# Extraer funciones y constantes necesarias
MODELOS_DIR = config.MODELOS_DIR
MENSAJES = config.MENSAJES
CV_FOLDS = config.CV_FOLDS
RANDOM_STATE = config.RANDOM_STATE

imprimir_separador = utils.imprimir_separador
guardar_pickle = utils.guardar_pickle
limpiar_memoria = utils.limpiar_memoria

# Suprimir warnings de convergencia
warnings.filterwarnings('ignore', category=UserWarning)

def obtener_modelos_avanzados() -> Dict[str, Dict[str, Any]]:
    """
    Define los modelos avanzados a comparar con sus hiperparámetros.
    
    Returns:
        Dict: Diccionario con modelos y configuraciones
    """
    modelos = {
        'Random_Forest': {
            'modelo': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'escalado': False,
            'descripcion': 'Ensemble robusto para capturar interacciones complejas'
        },
        'Gradient_Boosting': {
            'modelo': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'escalado': False,
            'descripcion': 'Boosting secuencial para alta precisión'
        },
        'SVR': {
            'modelo': SVR(),
            'params': {
                'kernel': ['rbf', 'poly'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'escalado': True,
            'descripcion': 'Máquina de vectores de soporte para relaciones no lineales'
        },
        'XGBoost': {
            'modelo': None,  # Se cargará dinámicamente
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'escalado': False,
            'descripcion': 'Gradient boosting optimizado, estado del arte'
        },
        'MLP_Neural_Network': {
            'modelo': MLPRegressor(
                random_state=RANDOM_STATE,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'escalado': True,
            'descripcion': 'Red neuronal multicapa para patrones complejos'
        }
    }
    
    return modelos

def cargar_xgboost():
    """
    Carga XGBoost dinámicamente. Si no está instalado, lo omite.
    
    Returns:
        XGBRegressor or None: Modelo XGBoost o None si no está disponible
    """
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    except ImportError:
        print("⚠️  XGBoost no está instalado. Omitiendo este modelo.")
        print("   💡 Instalar con: pip install xgboost")
        return None

def entrenar_modelo_avanzado(nombre: str, config_modelo: Dict, 
                           X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
                           y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Entrena un modelo individual con optimización de hiperparámetros.
    
    Args:
        nombre (str): Nombre del modelo
        config_modelo (Dict): Configuración del modelo
        X_train_scaled, X_test_scaled: Features escaladas
        y_train, y_test: Variables objetivo
    
    Returns:
        Dict: Resultados del entrenamiento
    """
    print(f"\n🔄 Entrenando {nombre}...")
    start_time = time.time()
    
    try:
        # Obtener modelo
        if nombre == 'XGBoost':
            modelo = cargar_xgboost()
            if modelo is None:
                return None
        else:
            modelo = config_modelo['modelo']
        
        # Grid Search para optimización
        print(f"   🔍 Optimizando hiperparámetros...")
        grid_search = GridSearchCV(
            modelo,
            config_modelo['params'],
            cv=CV_FOLDS,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred_train = mejor_modelo.predict(X_train_scaled)
        y_pred_test = mejor_modelo.predict(X_test_scaled)
        
        # Validación cruzada adicional
        cv_scores = cross_val_score(
            mejor_modelo, X_train_scaled, y_train, 
            cv=CV_FOLDS, scoring='r2'
        )
        
        # Calcular métricas
        resultados = {
            'modelo': mejor_modelo,
            'mejores_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'tiempo_entrenamiento': time.time() - start_time,
            'descripcion': config_modelo['descripcion'],
            
            # Métricas en entrenamiento
            'r2_train': r2_score(y_train, y_pred_train),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            
            # Métricas en test
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            
            # Validación cruzada
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            
            # Predicciones para análisis posterior
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train
        }
        
        # Detectar overfitting
        overfitting = resultados['r2_train'] - resultados['r2_test']
        resultados['overfitting'] = overfitting
        
        print(f"   ✅ {nombre} completado:")
        print(f"      R² Test: {resultados['r2_test']:.4f}")
        print(f"      RMSE Test: {resultados['rmse_test']:.4f}")
        print(f"      CV R² Mean: {resultados['cv_r2_mean']:.4f} ± {resultados['cv_r2_std']:.4f}")
        print(f"      Overfitting: {overfitting:.4f}")
        print(f"      Tiempo: {resultados['tiempo_entrenamiento']:.2f}s")
        
        return resultados
        
    except Exception as e:
        print(f"   ❌ Error con {nombre}: {str(e)}")
        return None

def ejecutar_comparacion_avanzada(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Ejecuta la comparación completa de modelos avanzados.
    
    Args:
        X_train, X_test: Features de entrenamiento y test
        y_train, y_test: Variables objetivo
    
    Returns:
        Dict: Resultados de todos los modelos
    """
    imprimir_separador("COMPARACIÓN AVANZADA DE MODELOS ML")
    
    print("🚀 Iniciando comparación de modelos avanzados...")
    print(f"📊 Modelos a evaluar: Random Forest, Gradient Boosting, SVR, XGBoost, Red Neuronal")
    print(f"🔧 Configuración: {CV_FOLDS}-fold CV, Grid Search, métricas completas")
    
    # Preparar escaladores
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener configuraciones de modelos
    modelos_config = obtener_modelos_avanzados()
    resultados = {}
    
    # Entrenar cada modelo
    for nombre, config in modelos_config.items():
        # Seleccionar datos (escalados o no)
        if config['escalado']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        resultado = entrenar_modelo_avanzado(
            nombre, config, X_train_use, X_test_use, y_train, y_test
        )
        
        if resultado is not None:
            resultados[nombre] = resultado
            # Guardar escalador si es necesario
            if config['escalado']:
                resultados[nombre]['scaler'] = scaler
    
    if not resultados:
        print("❌ No se pudieron entrenar modelos avanzados")
        return {}
    
    print(f"\n✅ Comparación completada. {len(resultados)} modelos entrenados exitosamente.")
    return resultados

def mostrar_ranking_completo(resultados: Dict[str, Any]) -> Tuple[str, Dict]:
    """
    Muestra ranking completo y detallado de todos los modelos.
    
    Args:
        resultados (Dict): Resultados de todos los modelos
    
    Returns:
        Tuple: Nombre y datos del mejor modelo
    """
    imprimir_separador("RANKING COMPLETO DE MODELOS")
    
    # Crear DataFrame para comparación
    datos_comparacion = []
    for nombre, resultado in resultados.items():
        datos_comparacion.append({
            'Modelo': nombre,
            'R² Test': resultado['r2_test'],
            'RMSE Test': resultado['rmse_test'],
            'MAE Test': resultado['mae_test'],
            'CV R² Mean': resultado['cv_r2_mean'],
            'CV R² Std': resultado['cv_r2_std'],
            'Overfitting': resultado['overfitting'],
            'Tiempo (s)': resultado['tiempo_entrenamiento'],
            'Descripción': resultado['descripcion']
        })
    
    df_ranking = pd.DataFrame(datos_comparacion)
    df_ranking = df_ranking.sort_values('R² Test', ascending=False)
    
    print("\n🏆 RANKING DE MODELOS (ordenado por R² Test):")
    print("=" * 120)
    
    # Mostrar ranking formateado
    for i, (_, fila) in enumerate(df_ranking.iterrows(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        overfitting_status = "⚠️ Alto" if fila['Overfitting'] > 0.1 else "✅ Bajo"
        
        print(f"{emoji} {fila['Modelo']:20} | R²: {fila['R² Test']:6.4f} | "
              f"RMSE: {fila['RMSE Test']:6.3f} | CV: {fila['CV R² Mean']:6.4f}±{fila['CV R² Std']:5.3f} | "
              f"Overfitting: {overfitting_status}")
    
    print("=" * 120)
    
    # Análisis del mejor modelo
    mejor_modelo_nombre = df_ranking.iloc[0]['Modelo']
    mejor_resultado = resultados[mejor_modelo_nombre]
    
    print(f"\n🎯 ANÁLISIS DEL MEJOR MODELO: {mejor_modelo_nombre}")
    print("-" * 60)
    print(f"📊 R² Score: {mejor_resultado['r2_test']:.4f}")
    print(f"📊 RMSE: {mejor_resultado['rmse_test']:.3f}")
    print(f"📊 MAE: {mejor_resultado['mae_test']:.3f}")
    print(f"🔧 Mejores parámetros: {mejor_resultado['mejores_params']}")
    print(f"💡 Descripción: {mejor_resultado['descripcion']}")
    
    # Alertas de calidad
    if mejor_resultado['overfitting'] > 0.1:
        print("⚠️  ALERTA: Posible overfitting detectado")
    if mejor_resultado['cv_r2_std'] > 0.05:
        print("⚠️  ALERTA: Alta variabilidad en validación cruzada")
    
    return mejor_modelo_nombre, mejor_resultado

def guardar_mejor_modelo_avanzado(nombre: str, resultado: Dict, 
                                X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Guarda el mejor modelo avanzado y sus metadatos.
    
    Args:
        nombre (str): Nombre del mejor modelo
        resultado (Dict): Resultado del mejor modelo
        X_train, y_train: Datos de entrenamiento completos
    """
    imprimir_separador("GUARDANDO MEJOR MODELO AVANZADO")
    
    print(f"💾 Guardando modelo: {nombre}")
    
    # Re-entrenar en todos los datos disponibles
    modelo_final = resultado['modelo']
    
    # Usar escalador si es necesario
    if 'scaler' in resultado:
        X_train_final = resultado['scaler'].transform(X_train)
        modelo_final.fit(X_train_final, y_train)
        
        # Guardar escalador
        ruta_scaler = MODELOS_DIR / "scaler_avanzado.pkl"
        guardar_pickle(resultado['scaler'], ruta_scaler)
        print(f"✅ Scaler guardado: {ruta_scaler.name}")
    else:
        modelo_final.fit(X_train, y_train)
    
    # Guardar modelo
    ruta_modelo = MODELOS_DIR / f"mejor_modelo_avanzado_{nombre.lower()}.pkl"
    guardar_pickle(modelo_final, ruta_modelo)
    
    # Guardar metadatos completos
    metadatos = {
        'nombre_modelo': nombre,
        'descripcion': resultado['descripcion'],
        'parametros': resultado['mejores_params'],
        'metricas': {
            'r2_test': resultado['r2_test'],
            'rmse_test': resultado['rmse_test'],
            'mae_test': resultado['mae_test'],
            'cv_r2_mean': resultado['cv_r2_mean'],
            'cv_r2_std': resultado['cv_r2_std'],
            'overfitting': resultado['overfitting']
        },
        'tiempo_entrenamiento': resultado['tiempo_entrenamiento'],
        'requiere_escalado': 'scaler' in resultado,
        'fecha_entrenamiento': pd.Timestamp.now().isoformat()
    }
    
    ruta_metadatos = MODELOS_DIR / "metadatos_mejor_modelo_avanzado.pkl"
    guardar_pickle(metadatos, ruta_metadatos)
    
    print(f"✅ Modelo guardado: {ruta_modelo.name}")
    print(f"✅ Metadatos guardados: {ruta_metadatos.name}")
    print(f"📊 R² Score final: {resultado['r2_test']:.4f}")

def ejecutar_comparacion_completa() -> Dict[str, Any]:
    """
    Ejecuta la comparación completa de modelos avanzados.
    
    Returns:
        Dict: Resultados completos de la comparación
    """
    try:
        print("🚀 Iniciando comparación avanzada de modelos...")
        
        # Cargar datos procesados
        preprocesamiento = importlib.import_module('src.02_preprocesamiento')
        X_train, X_test, y_train, y_test = preprocesamiento.cargar_datos_entrenamiento()
        
        print(f"📊 Datos cargados: Train {X_train.shape}, Test {X_test.shape}")
        
        # Ejecutar comparación
        resultados = ejecutar_comparacion_avanzada(X_train, X_test, y_train, y_test)
        
        if not resultados:
            print("❌ No se pudieron obtener resultados")
            return {}
        
        # Mostrar ranking y seleccionar mejor
        mejor_nombre, mejor_resultado = mostrar_ranking_completo(resultados)
        
        # Guardar mejor modelo
        guardar_mejor_modelo_avanzado(mejor_nombre, mejor_resultado, X_train, y_train)
        
        # Limpiar memoria
        limpiar_memoria()
        
        print(f"\n{MENSAJES['fin_entrenamiento']}")
        print("✅ Comparación avanzada completada exitosamente")
        
        return resultados
        
    except Exception as e:
        print(f"❌ Error en comparación avanzada: {str(e)}")
        return {}

def main():
    """Función principal para ejecutar la comparación avanzada."""
    print("🔄 Iniciando módulo de comparación avanzada de modelos...")
    resultados = ejecutar_comparacion_completa()
    return resultados

if __name__ == "__main__":
    main()
