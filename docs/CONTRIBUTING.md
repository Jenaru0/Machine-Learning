# 👥 Guía de Contribución al Proyecto

## 🎯 Bienvenido al Proyecto Integrador de Machine Learning

Esta guía te ayudará a configurar y contribuir al proyecto de predicción de rendimiento académico.

## 📋 Información del Equipo

### 👨‍🎓 Grupo 4 - Machine Learning 2025

- **Candela Vargas Aitor Baruc**
- **Godoy Bautista Denilson Miguel**
- **Molina Lazaro Eduardo Jeampier**
- **Napanga Ruiz Jhonatan Jesus**
- **Quispe Romani Angela Isabel**

## 🚀 Configuración Inicial

### 1. Clonar el Repositorio

```bash
git clone https://github.com/[usuario]/proyecto-integrador-ml.git
cd proyecto-integrador-ml
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python ejecutar_pipeline.py
```

## 📊 Estructura del Proyecto

```
proyecto-integrador-ml/
├── datos/                 # Datasets y archivos de datos
├── modelos/              # Modelos entrenados
├── notebooks/            # Jupyter notebooks para exploración
├── src/                  # Código fuente principal
├── tests/                # Pruebas unitarias
├── ejecutar_pipeline.py  # Script principal de ejecución
└── README.md            # Documentación principal
```

## 🔧 Flujo de Trabajo

### 1. Antes de Empezar

```bash
# Actualizar desde el repositorio principal
git pull origin main

# Crear una nueva rama para tu feature
git checkout -b feature/nombre-de-tu-feature
```

### 2. Desarrollar

- Haz tus cambios en la rama correspondiente
- Asegúrate de que el código siga las convenciones del proyecto
- Ejecuta las pruebas antes de hacer commit

```bash
# Ejecutar pruebas
python -m pytest tests/ -v

# Ejecutar pipeline completo
python ejecutar_pipeline.py
```

### 3. Commit y Push

```bash
# Agregar cambios
git add .

# Commit con mensaje descriptivo
git commit -m "feat: descripción clara de los cambios"

# Push a tu rama
git push origin feature/nombre-de-tu-feature
```

### 4. Pull Request

- Crea un Pull Request desde tu rama hacia `main`
- Incluye una descripción clara de los cambios
- Espera la revisión de al menos un compañero

## 📝 Convenciones de Código

### Nombres de Variables y Funciones

```python
# ✅ Buenas prácticas
def cargar_datos_estudiantes():
    """Carga el dataset de estudiantes."""
    pass

variable_importante = "valor"

# ❌ Evitar
def cargaDatos():
    pass

var1 = "valor"
```

### Comentarios y Documentación

```python
def procesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa y limpia el dataset de estudiantes.
    
    Args:
        df (pd.DataFrame): DataFrame con datos originales
        
    Returns:
        pd.DataFrame: DataFrame procesado
    """
    # Código aquí
    return df
```

### Estructura de Commits

```bash
# Tipos de commits
feat: nueva funcionalidad
fix: corrección de errores
docs: documentación
test: pruebas
refactor: refactorización
style: formato de código
```

## 🧪 Pruebas

### Ejecutar Todas las Pruebas

```bash
python -m pytest tests/ -v
```

### Ejecutar Pruebas Específicas

```bash
python -m pytest tests/test_preprocesamiento.py -v
```

### Agregar Nuevas Pruebas

```python
# En tests/test_nuevo_modulo.py
import pytest
from src.nuevo_modulo import nueva_funcion

def test_nueva_funcion():
    """Prueba para la nueva función."""
    resultado = nueva_funcion()
    assert resultado is not None
```

## 📂 Organización de Archivos

### Agregar Nuevos Módulos

```python
# src/nuevo_modulo.py
"""
Descripción del módulo
"""

def nueva_funcion():
    """Descripción de la función."""
    pass
```

### Actualizar Configuración

```python
# src/config.py
# Agregar nuevas configuraciones al final
NUEVA_CONFIGURACION = "valor"
```

## 🚨 Problemas Comunes

### Error de Dependencias

```bash
# Actualizar requirements.txt
pip freeze > requirements.txt

# Instalar dependencias faltantes
pip install -r requirements.txt
```

### Conflictos de Merge

```bash
# Resolver conflictos manualmente
git status
git add .
git commit -m "resolve: conflicts resolved"
```

### Problemas con Datos

```bash
# Verificar que los datos estén en la carpeta correcta
ls datos/raw/
ls datos/procesados/
```

## 📊 Próximos Pasos del Proyecto

### 🔄 Fase Actual (Completada)

- ✅ Pipeline de ML completo
- ✅ Modelos entrenados y evaluados
- ✅ Documentación básica
- ✅ Pruebas unitarias

### 🚀 Próximas Fases

1. **Interfaz Web**
   - Desarrollo de dashboard interactivo
   - Carga de datos personalizada
   - Visualización de predicciones

2. **Integración de APIs**
   - APIs de IA externa
   - Endpoints para predicciones
   - Documentación de API

3. **Mejoras del Modelo**
   - Ensemble methods
   - Hyperparameter tuning avanzado
   - Validación cruzada extendida

## 📞 Contacto y Soporte

### Canales de Comunicación

- **WhatsApp del Grupo**: [Agregar enlace]
- **Email**: [Agregar emails del equipo]
- **GitHub Issues**: Para reportar problemas técnicos

### Reuniones

- **Reuniones semanales**: [Definir día y hora]
- **Stand-ups**: [Definir frecuencia]
- **Revisiones de código**: Por Pull Request

## 📚 Recursos Adicionales

### Documentación Técnica

- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Estándares del Proyecto

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) - Guía de estilo de Python
- [Conventional Commits](https://www.conventionalcommits.org/) - Formato de commits

---

## 🎉 ¡Gracias por Contribuir!

Tu aporte es valioso para el éxito del proyecto. Si tienes dudas o sugerencias, no dudes en contactar al equipo.

**¡Vamos por la mejor calificación! 🏆**
