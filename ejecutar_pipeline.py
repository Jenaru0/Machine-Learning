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

# Importar módulos del proyecto
from src.config import MENSAJES, verificar_archivo_datos
from src.utils import mostrar_info_sistema, imprimir_separador
from src.eda import ejecutar_eda_completo
from src.preprocesamiento import ejecutar_preprocesamiento_completo
from src.entrenar_modelo import ejecutar_entrenamiento_completo
from src.predecir import ejecutar_prediccion_completa

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
        
        # Paso 3: Entrenamiento y optimización de modelos
        print("\n🚀 PASO 3: ENTRENAMIENTO Y OPTIMIZACIÓN DE MODELOS")
        print("=" * 50)
        inicio_entrenamiento = time.time()
        
        resultados_entrenamiento = ejecutar_entrenamiento_completo()
        
        fin_entrenamiento = time.time()
        print(f"⏱️  Tiempo entrenamiento: {fin_entrenamiento - inicio_entrenamiento:.2f} segundos")
        
        # Paso 4: Generación de predicciones finales
        print("\n📊 PASO 4: GENERACIÓN DE PREDICCIONES FINALES")
        print("=" * 50)
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
        print(f"   🚀 Entrenamiento: {fin_entrenamiento - inicio_entrenamiento:.2f}s")
        print(f"   📊 Predicción: {fin_prediccion - inicio_prediccion:.2f}s")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📊 Reporte EDA: datos/profiling/reporte_estudiantes.html")
        print(f"   📋 Datos procesados: datos/procesados/")
        print(f"   🤖 Mejor modelo: modelos/ridge_alpha_10.pkl")
        print(f"   📈 Predicciones: datos/procesados/predicciones_exam_score.csv")
        
        mejor_modelo = resultados_entrenamiento['mejor_modelo']
        mejor_r2 = resultados_entrenamiento['comparacion'].iloc[0]['R²']
        
        print(f"\n🏆 MEJOR MODELO: {mejor_modelo}")
        print(f"   📈 R² Score: {mejor_r2:.4f}")
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
