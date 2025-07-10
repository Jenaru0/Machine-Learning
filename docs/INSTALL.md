# 🛠️ Guía de Instalación

## 📋 Requisitos del Sistema

### Requisitos Mínimos

- **Python**: 3.8 o superior (recomendado: 3.9+)
- **RAM**: 4GB mínimo (recomendado: 8GB)
- **Espacio en disco**: 2GB libres
- **Sistema operativo**: Windows 10/11, macOS 10.14+, o Linux Ubuntu 18.04+

### Verificar Python

```bash
python --version
# o
python3 --version
```

## 🚀 Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/[usuario]/proyecto-integrador-ml.git
cd proyecto-integrador-ml
```

### 2. Crear Entorno Virtual

#### Windows

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate

# Verificar que está activo (debe mostrar (venv) al inicio)
```

#### Linux/macOS

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Verificar que está activo (debe mostrar (venv) al inicio)
```

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 5. Verificar Instalación

```bash
# Ejecutar pipeline completo
python ejecutar_pipeline.py
```

Si todo funciona correctamente, deberías ver:
- Análisis exploratorio de datos
- Preprocesamiento
- Entrenamiento de modelos
- Generación de predicciones
- Mensaje de éxito final

## 🔧 Instalación Alternativa

### Usando conda (Anaconda/Miniconda)

```bash
# Crear entorno con conda
conda create -n ml-proyecto python=3.9
conda activate ml-proyecto

# Instalar dependencias principales
conda install pandas numpy scikit-learn matplotlib seaborn jupyter

# Instalar dependencias adicionales con pip
pip install -r requirements.txt
```

### Usando pip directamente (no recomendado)

```bash
# Solo si no puedes usar entornos virtuales
pip install -r requirements.txt
```

## 🧪 Verificación de la Instalación

### 1. Ejecutar Pruebas Unitarias

```bash
python -m pytest tests/ -v
```

**Resultado esperado:**
```
tests/test_preprocesamiento.py::TestPreprocesamiento::test_crear_dataset_numerico_final PASSED
tests/test_preprocesamiento.py::TestPreprocesamiento::test_crear_variables_ingenieria PASSED
tests/test_preprocesamiento.py::TestPreprocesamiento::test_integridad_datos PASSED
tests/test_preprocesamiento.py::TestPreprocesamiento::test_limpiar_datos PASSED
============= 4 passed in 3.74s =============
```

### 2. Verificar Estructura de Archivos

```bash
# Verificar que los archivos principales existen
ls -la datos/raw/
ls -la datos/procesados/
ls -la modelos/
ls -la src/
```

### 3. Ejecutar Pipeline Individual

```bash
# Solo EDA
python -c "from src.eda import main; main()"

# Solo preprocesamiento
python -c "from src.preprocesamiento import main; main()"

# Solo entrenamiento
python -c "from src.entrenar_modelo import main; main()"
```

## 🔍 Solución de Problemas

### Error: "Python no encontrado"

```bash
# Windows: Instalar desde python.org
# Linux: sudo apt-get install python3
# macOS: brew install python3
```

### Error: "No module named 'xyz'"

```bash
# Asegúrate de que el entorno virtual está activo
pip install xyz
# o
pip install -r requirements.txt
```

### Error: "Permission denied"

```bash
# Linux/macOS: usar sudo para instalaciones globales
sudo pip install xyz

# O crear entorno virtual en tu directorio home
python3 -m venv ~/venv-ml
source ~/venv-ml/bin/activate
```

### Error: "Dataset no encontrado"

```bash
# Verificar que el archivo existe
ls datos/raw/StudentPerformanceFactors.csv

# Si no existe, descargar desde el repositorio
```

### Error: "Memoria insuficiente"

```bash
# Reducir el tamaño del dataset para pruebas
# O usar una máquina con más RAM
```

## 📊 Verificación de Recursos

### Verificar Dependencias Instaladas

```bash
pip list | grep -E "(pandas|numpy|scikit-learn|matplotlib|seaborn)"
```

### Verificar Espacio en Disco

```bash
# Linux/macOS
df -h

# Windows
dir
```

### Verificar RAM Disponible

```bash
# Linux
free -h

# macOS
vm_stat

# Windows
systeminfo | find "Available Physical Memory"
```

## 🔄 Actualización del Proyecto

### Obtener Últimos Cambios

```bash
git pull origin main
```

### Actualizar Dependencias

```bash
pip install -r requirements.txt --upgrade
```

### Actualizar Solo Librerías Específicas

```bash
pip install pandas --upgrade
pip install scikit-learn --upgrade
```

## 📝 Configuración del IDE

### Visual Studio Code

1. Instalar extensiones recomendadas:
   - Python
   - Jupyter
   - GitLens

2. Configurar intérprete de Python:
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - Seleccionar el intérprete del entorno virtual

### PyCharm

1. Abrir proyecto
2. Configurar intérprete: File → Settings → Project → Python Interpreter
3. Seleccionar el entorno virtual creado

### Jupyter Notebook

```bash
# Instalar Jupyter si no está instalado
pip install jupyter

# Iniciar notebook
jupyter notebook

# Abrir notebooks/exploracion_interactiva.ipynb
```

## 🆘 Soporte Técnico

### Recursos de Ayuda

- **GitHub Issues**: Reportar problemas técnicos
- **Documentación**: Revisar README.md y CONTRIBUTING.md
- **Logs**: Revisar archivos de log para errores detallados

### Información del Sistema

```bash
# Obtener información completa del sistema
python -c "import platform; print(platform.platform())"
python -c "import sys; print(sys.version)"
pip --version
```

### Crear Reporte de Error

Si encuentras un problema, incluye:

1. **Sistema operativo** y versión
2. **Versión de Python**
3. **Mensaje de error completo**
4. **Pasos para reproducir el error**
5. **Archivos de log** si están disponibles

---

## ✅ Instalación Completada

¡Felicidades! Si llegaste hasta aquí sin errores, el proyecto está listo para usar.

**Próximos pasos:**
1. Leer `README.md` para entender la estructura
2. Leer `CONTRIBUTING.md` para contribuir al código
3. Ejecutar `python ejecutar_pipeline.py` para ver el proyecto en acción

**¡Buen trabajo! 🎉**
