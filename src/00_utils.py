"""
Utilidades Comunes del Proyecto
===============================

Funciones auxiliares que son utilizadas en múltiples módulos del proyecto.
Incluye funciones para carga, guardado, logging y otras utilidades.

Autor: Equipo Grupo 4
Fecha: 2025
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Union, Any, Dict, List
import warnings

# Configurar warnings
warnings.filterwarnings('ignore')

def configurar_logging(nivel=logging.INFO):
    """
    Configura el sistema de logging del proyecto.
    
    Args:
        nivel (int): Nivel de logging (por defecto INFO)
    """
    logging.basicConfig(
        level=nivel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silenciar warnings de sklearn
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def cargar_dataframe(ruta_archivo: Union[str, Path]) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo CSV.
    
    Args:
        ruta_archivo (str|Path): Ruta del archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        Exception: Si hay error al leer el archivo
    """
    try:
        ruta = Path(ruta_archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        
        df = pd.read_csv(ruta)
        logging.info(f"✅ Archivo cargado exitosamente: {ruta.name}")
        logging.info(f"📊 Dimensiones: {df.shape}")
        
        return df
        
    except Exception as e:
        logging.error(f"❌ Error al cargar el archivo {ruta_archivo}: {str(e)}")
        raise

def guardar_dataframe(df: pd.DataFrame, ruta_archivo: Union[str, Path], 
                     index: bool = False) -> None:
    """
    Guarda un DataFrame en un archivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        ruta_archivo (str|Path): Ruta donde guardar el archivo
        index (bool): Si incluir el índice en el CSV
    """
    try:
        ruta = Path(ruta_archivo)
        
        # Crear directorio si no existe
        ruta.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(ruta, index=index)
        logging.info(f"✅ DataFrame guardado exitosamente: {ruta.name}")
        logging.info(f"📊 Dimensiones guardadas: {df.shape}")
        
    except Exception as e:
        logging.error(f"❌ Error al guardar el DataFrame: {str(e)}")
        raise

def guardar_pickle(objeto: Any, ruta_archivo: Union[str, Path]) -> None:
    """
    Guarda un objeto usando pickle.
    
    Args:
        objeto: Objeto a serializar
        ruta_archivo (str|Path): Ruta donde guardar el archivo
    """
    try:
        ruta = Path(ruta_archivo)
        
        # Crear directorio si no existe
        ruta.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ruta, 'wb') as f:
            pickle.dump(objeto, f)
        
        logging.info(f"✅ Objeto guardado exitosamente: {ruta.name}")
        
    except Exception as e:
        logging.error(f"❌ Error al guardar el objeto: {str(e)}")
        raise

def cargar_pickle(ruta_archivo: Union[str, Path]) -> Any:
    """
    Carga un objeto desde un archivo pickle.
    
    Args:
        ruta_archivo (str|Path): Ruta del archivo pickle
        
    Returns:
        Any: Objeto deserializado
    """
    try:
        ruta = Path(ruta_archivo)
        
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        
        with open(ruta, 'rb') as f:
            objeto = pickle.load(f)
        
        logging.info(f"✅ Objeto cargado exitosamente: {ruta.name}")
        
        return objeto
        
    except Exception as e:
        logging.error(f"❌ Error al cargar el objeto: {str(e)}")
        raise

def mostrar_info_dataframe(df: pd.DataFrame, nombre: str = "DataFrame") -> None:
    """
    Muestra información básica de un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        nombre (str): Nombre descriptivo del DataFrame
    """
    print(f"\n{'='*50}")
    print(f"📊 INFORMACIÓN DE {nombre.upper()}")
    print(f"{'='*50}")
    
    print(f"📏 Dimensiones: {df.shape}")
    print(f"📋 Columnas: {df.columns.tolist()}")
    print(f"🔢 Tipos de datos:")
    print(df.dtypes.value_counts())
    
    # Valores nulos
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(f"\n⚠️  Valores nulos encontrados:")
        print(nulos[nulos > 0])
    else:
        print(f"\n✅ No hay valores nulos")
    
    # Duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        print(f"⚠️  Filas duplicadas: {duplicados}")
    else:
        print(f"✅ No hay filas duplicadas")

def mostrar_estadisticas_variable(df: pd.DataFrame, columna: str) -> None:
    """
    Muestra estadísticas detalladas de una variable.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la variable
        columna (str): Nombre de la columna a analizar
    """
    if columna not in df.columns:
        print(f"❌ La columna '{columna}' no existe en el DataFrame")
        return
    
    print(f"\n📈 ESTADÍSTICAS DE '{columna}':")
    print("-" * 40)
    
    # Si es numérica
    if df[columna].dtype in ['int64', 'float64']:
        print(df[columna].describe())
        
        # Valores extremos
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        print(f"\n📊 Posibles outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        
    # Si es categórica
    else:
        print(df[columna].value_counts())
        print(f"\n📊 Categorías únicas: {df[columna].nunique()}")

def validar_columnas_requeridas(df: pd.DataFrame, columnas_requeridas: List[str]) -> bool:
    """
    Valida que un DataFrame contenga las columnas requeridas.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        columnas_requeridas (List[str]): Lista de columnas requeridas
        
    Returns:
        bool: True si todas las columnas están presentes
        
    Raises:
        ValueError: Si faltan columnas requeridas
    """
    columnas_faltantes = set(columnas_requeridas) - set(df.columns)
    
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {list(columnas_faltantes)}")
    
    logging.info("✅ Todas las columnas requeridas están presentes")
    return True

def limpiar_memoria():
    """Limpia variables innecesarias de memoria."""
    import gc
    gc.collect()
    logging.info("🧹 Memoria limpiada")

def imprimir_separador(texto: str = "", caracter: str = "=", longitud: int = 50) -> None:
    """
    Imprime un separador visual con texto opcional.
    
    Args:
        texto (str): Texto a mostrar en el separador
        caracter (str): Caracter del separador
        longitud (int): Longitud del separador
    """
    if texto:
        print(f"\n{caracter * longitud}")
        print(f"{texto.center(longitud)}")
        print(f"{caracter * longitud}")
    else:
        print(f"\n{caracter * longitud}")

def obtener_info_sistema():
    """
    Obtiene información del sistema y las librerías.
    
    Returns:
        dict: Información del sistema
    """
    import platform
    import sys
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "No instalado"
    
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except ImportError:
        matplotlib_version = "No instalado"
    
    try:
        import seaborn
        seaborn_version = seaborn.__version__
    except ImportError:
        seaborn_version = "No instalado"
    
    info = {
        'Sistema': platform.system(),
        'Versión Python': sys.version,
        'Pandas': pd.__version__,
        'NumPy': np.__version__,
        'Scikit-learn': sklearn_version,
        'Matplotlib': matplotlib_version,
        'Seaborn': seaborn_version
    }
    
    return info

def mostrar_info_sistema():
    """Muestra información del sistema y versiones de librerías."""
    info = obtener_info_sistema()
    
    print("\n" + "="*50)
    print("🖥️  INFORMACIÓN DEL SISTEMA")
    print("="*50)
    
    for clave, valor in info.items():
        print(f"{clave}: {valor}")
    
    print("="*50)

# Configurar logging al importar el módulo
configurar_logging()
