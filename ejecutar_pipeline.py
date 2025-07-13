"""
Pipeline Principal del Proyecto
==============================

Script principal que ejecuta todo el pipeline de Machine Learning
de forma secuencial y automatizada.

Ejecuta:
1. Análisis Exploratorio de Datos (EDA)
2. Preprocesamiento de datos
3. Entrenamiento y optimización de modelos
4. Generación de predicciones finales

Autor: Equipo Grupo 4
Fecha: 2025
"""

import sys
import logging
from pathlib import Path
import time

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importar módulos del proyecto (con nueva numeración)
import importlib
config = importlib.import_module('src.00_config')
utils = importlib.import_module('src.00_utils')
eda = importlib.import_module('src.01_eda')
preprocesamiento = importlib.import_module('src.02_preprocesamiento')
entrenar_modelo = importlib.import_module('src.03_entrenar_modelo')
predecir = importlib.import_module('src.04_predecir')
comparar_avanzado = importlib.import_module('src.05_comparar_modelos_avanzados')

# Extraer funciones necesarias
MENSAJES = config.MENSAJES
verificar_archivo_datos = config.verificar_archivo_datos
mostrar_info_sistema = utils.mostrar_info_sistema
imprimir_separador = utils.imprimir_separador
ejecutar_eda_completo = eda.ejecutar_eda_completo
ejecutar_preprocesamiento_completo = preprocesamiento.ejecutar_preprocesamiento_completo
ejecutar_entrenamiento_completo = entrenar_modelo.ejecutar_entrenamiento_completo
ejecutar_prediccion_completa = predecir.ejecutar_prediccion_completa
ejecutar_comparacion_avanzada = comparar_avanzado.ejecutar_comparacion_completa

def mostrar_banner():
    """Muestra el banner del proyecto."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║            🎓 PROYECTO INTEGRADOR DE MACHINE LEARNING 🎓                     ║
    ║                                                                               ║
    ║                   📊 Predicción de Rendimiento Académico 📊                  ║
    ║                                                                               ║
    ║                            👨‍🎓 Equipo Grupo 4 👩‍🎓                            ║
    ║                                                                               ║
    ║    • Candela Vargas Aitor Baruc                                              ║
    ║    • Godoy Bautista Denilson Miguel                                          ║
    ║    • Molina Lazaro Eduardo Jeampier                                          ║
    ║    • Napanga Ruiz Jhonatan Jesus                                             ║
    ║    • Quispe Romani Angela Isabel                                             ║
    ║                                                                               ║
    ║                        📚 Machine Learning - 2025 📚                         ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def verificar_requisitos():
    """Verifica que todos los requisitos estén cumplidos."""
    print("\n🔍 VERIFICANDO REQUISITOS DEL SISTEMA...")
    
    try:
        # Verificar archivo de datos
        verificar_archivo_datos()
        print("✅ Archivo de datos encontrado")
        
        # Verificar librerías necesarias
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Librerías principales instaladas")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: Falta instalar librerías: {str(e)}")
        print("💡 Ejecute: pip install -r requirements.txt")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        return False

def ejecutar_pipeline_completo():
    """
    Ejecuta el pipeline completo de Machine Learning.
    
    Returns:
        bool: True si se ejecutó exitosamente, False en caso contrario
    """
    inicio_total = time.time()
    
    try:
        # Mostrar información del sistema
        mostrar_info_sistema()
        
        # Verificar requisitos
        if not verificar_requisitos():
            return False
        
        imprimir_separador("INICIANDO PIPELINE COMPLETO", "=", 80)
        
        # Paso 1: Análisis Exploratorio de Datos (EDA)
        print("\n🔍 PASO 1: ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 50)
        inicio_eda = time.time()
        
        df_original = ejecutar_eda_completo()
        
        fin_eda = time.time()
        print(f"⏱️  Tiempo EDA: {fin_eda - inicio_eda:.2f} segundos")
        
        # Paso 2: Preprocesamiento de datos
        print("\n🔧 PASO 2: PREPROCESAMIENTO DE DATOS")
        print("=" * 50)
        inicio_preproceso = time.time()
        
        df_train, df_test = ejecutar_preprocesamiento_completo(df_original)
        
        fin_preproceso = time.time()
        print(f"⏱️  Tiempo preprocesamiento: {fin_preproceso - inicio_preproceso:.2f} segundos")
        
        # Paso 3: Entrenamiento y optimización de modelos básicos
        print("\n🚀 PASO 3: ENTRENAMIENTO Y OPTIMIZACIÓN DE MODELOS BÁSICOS")
        print("=" * 60)
        inicio_entrenamiento = time.time()
        
        resultados_entrenamiento = ejecutar_entrenamiento_completo()
        
        fin_entrenamiento = time.time()
        print(f"⏱️  Tiempo entrenamiento básico: {fin_entrenamiento - inicio_entrenamiento:.2f} segundos")
        
        # Paso 4: Comparación avanzada de modelos (NUEVO)
        print("\n🏆 PASO 4: COMPARACIÓN AVANZADA DE MODELOS")
        print("=" * 60)
        inicio_comparacion = time.time()
        
        resultados_avanzados = ejecutar_comparacion_avanzada()
        
        fin_comparacion = time.time()
        print(f"⏱️  Tiempo comparación avanzada: {fin_comparacion - inicio_comparacion:.2f} segundos")
        
        # Paso 5: Generación de predicciones finales
        print("\n📊 PASO 5: GENERACIÓN DE PREDICCIONES FINALES")
        print("=" * 60)
        inicio_prediccion = time.time()
        
        predicciones = ejecutar_prediccion_completa()
        
        fin_prediccion = time.time()
        print(f"⏱️  Tiempo predicción: {fin_prediccion - inicio_prediccion:.2f} segundos")
        
        # Resumen final
        fin_total = time.time()
        tiempo_total = fin_total - inicio_total
        
        imprimir_separador("PIPELINE COMPLETADO EXITOSAMENTE", "=", 80)
        
        print(f"\n🎉 {MENSAJES['pipeline_completo']}")
        print(f"\n📊 RESUMEN DE EJECUCIÓN:")
        print(f"   ⏱️  Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")
        print(f"   🔍 EDA: {fin_eda - inicio_eda:.2f}s")
        print(f"   🔧 Preprocesamiento: {fin_preproceso - inicio_preproceso:.2f}s")
        print(f"   🚀 Entrenamiento básico: {fin_entrenamiento - inicio_entrenamiento:.2f}s")
        print(f"   🏆 Comparación avanzada: {fin_comparacion - inicio_comparacion:.2f}s")
        print(f"   📊 Predicción: {fin_prediccion - inicio_prediccion:.2f}s")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📊 Reporte EDA: datos/profiling/reporte_estudiantes.html")
        print(f"   📋 Datos procesados: datos/procesados/")
        print(f"   🤖 Modelos básicos: modelos/ridge_alpha_10.pkl")
        print(f"   🏆 Mejor modelo avanzado: modelos/mejor_modelo_avanzado_*.pkl")
        print(f"   📈 Predicciones: datos/procesados/predicciones_exam_score.csv")
        
        # Mostrar resumen de modelos
        if resultados_entrenamiento and 'mejor_modelo' in resultados_entrenamiento:
            mejor_modelo_basico = resultados_entrenamiento['mejor_modelo']
            mejor_r2_basico = resultados_entrenamiento['comparacion'].iloc[0]['R²']
            print(f"\n🥈 MEJOR MODELO BÁSICO: {mejor_modelo_basico}")
            print(f"   📈 R² Score: {mejor_r2_basico:.4f}")
        
        if resultados_avanzados:
            print(f"🏆 COMPARACIÓN AVANZADA: {len(resultados_avanzados)} modelos evaluados")
            print(f"   📊 Incluye: Random Forest, Gradient Boosting, SVR, XGBoost, Red Neuronal")
        
        print(f"   📊 Predicciones generadas: {len(predicciones)}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error en pipeline: {str(e)}")
        print(f"\n❌ ERROR EN PIPELINE: {str(e)}")
        return False

def main():
    """Función principal."""
    # Mostrar banner
    mostrar_banner()
    
    # Ejecutar pipeline
    exito = ejecutar_pipeline_completo()
    
    if exito:
        print(f"\n✅ Pipeline ejecutado exitosamente!")
        print(f"📚 Revise los archivos generados en las carpetas correspondientes.")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline falló. Revise los errores anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
