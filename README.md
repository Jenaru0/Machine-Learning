# 🎓 Predicción de Rendimiento Académico - Machine Learning

Un pipeline completo de Machine Learning para predecir el rendimiento académico de estudiantes basado en múltiples factores educativos y sociales.

## ✨ Características

- **Pipeline automatizado** desde datos hasta predicción
- **Análisis exploratorio** con reportes HTML interactivos
- **Preprocesamiento robusto** con manejo inteligente de datos
- **Modelo predictivo** con validación cruzada
- **Código modular** y bien documentado

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.11+ (recomendado para compatibilidad completa)
- pip

### Instalación

```bash
# Clonar repositorio
git clone "https://github.com/Jenaru0/Machine-Learning.git"
cd "machine learning"

# Crear entorno virtualpy --version

python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecucióna

```bash
# Ejecutar pipeline completo
python ejecutar_pipeline.py

# Ejecutar módulos individuales
python -m src.eda                    # Análisis exploratorio
python -m src.preprocesamiento       # Limpieza de datos
python -m src.entrenar_modelo        # Entrenamiento
python -m src.predecir               # Predicciones
```

## 📁 Estructura del Proyecto

```
proyecto-integrador-ml/
├── 📊 datos/
│   ├── raw/                     # Datos originales
│   ├── procesados/              # Datos limpios
│   └── profiling/               # Reportes HTML
├── 🔧 src/
│   ├── config.py                # Configuración central
│   ├── eda.py                   # Análisis exploratorio
│   ├── preprocesamiento.py      # Limpieza y transformación
│   ├── entrenar_modelo.py       # Entrenamiento del modelo
│   ├── predecir.py              # Predicciones
│   └── utils.py                 # Utilidades generales
├── 🤖 modelos/                  # Modelos entrenados (.pkl)
├── 📓 notebooks/                # Jupyter notebooks
├── 🎯 ejecutar_pipeline.py      # Script principal
├── 📋 requirements.txt          # Dependencias
└── 📖 README.md                 # Este archivo
```

## 🛠️ Tecnologías

- **Python 3.11+** - Lenguaje principal
- **pandas** - Manipulación de datos
- **scikit-learn** - Machine Learning
- **ydata-profiling** - Análisis exploratorio automático
- **numpy** - Computación numérica

## 📊 Funcionalidades

### Análisis Exploratorio (EDA)

- Estadísticas descriptivas completas
- Análisis de correlaciones
- Detección de valores atípicos
- Reporte HTML interactivo con ydata-profiling

### Preprocesamiento

- Limpieza de datos automatizada
- Codificación de variables categóricas
- Escalado de características
- División train/test

### Modelado

- Regresión Ridge con optimización de hiperparámetros
- Validación cruzada
- Métricas de evaluación completas
- Guardado automático del modelo

### Predicción

- Predicciones sobre nuevos datos
- Métricas de rendimiento
- Visualización de resultados

## 🔧 Configuración

Todas las configuraciones se encuentran en `src/config.py`:

- Rutas de archivos
- Parámetros del modelo
- Configuración de logging
- Mensajes del sistema

## 🏆 Rendimiento del Modelo

- **R² Score**: 0.6926 (69.26% varianza explicada)
- **RMSE**: 2.0552
- **Algoritmo**: Ridge Regression

## 📈 Resultados

El pipeline genera:

- **Reporte EDA**: `datos/profiling/reporte_eda.html`
- **Modelo entrenado**: `modelos/ridge_alpha_10.pkl`
- **Scaler**: `modelos/scaler.pkl`
- **Métricas**: Mostradas en consola

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Añade nueva funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para más detalles.

## 👥 Autores

**Equipo Grupo 4** - Proyecto Integrador Machine Learning

- Candela Vargas Aitor Baruc
- Godoy Bautista Denilson Miguel
- Molina Lazaro Eduardo Jeampier
- Napanga Ruiz Jhonatan Jesus
- Quispe Romani Angela Isabel

---

⭐ Si este proyecto te fue útil, ¡dale una estrella!

- **Archivo**: `StudentPerformanceFactors.csv`
- **Registros**: 6,607 estudiantes
- **Variables**: 20 características
- **Objetivo**: Predecir `Exam_Score`

### 🏗️ Estructura

```
proyecto-integrador-ml/
├── datos/raw/                  # Datos originales
├── datos/procesados/           # Datos procesados
├── modelos/                    # Modelos entrenados
├── notebooks/                  # Exploración interactiva
├── src/                        # Código fuente
├── docs/                       # Documentación adicional
├── ejecutar_pipeline.py        # Script principal
└── requirements.txt            # Dependencias
```

### 🔧 Funcionalidades

1. **EDA**: Análisis exploratorio automático
2. **Preprocesamiento**: Limpieza e ingeniería de características
3. **Modelado**: Ridge, Lasso, Linear Regression
4. **Predicción**: Generación de resultados finales

### 📈 Variables Más Importantes

1. **Attendance** (2.29) - Asistencia
2. **Hours_Studied** (1.57) - Horas de estudio
3. **Previous_Scores** (0.81) - Puntajes previos

### 📁 Archivos Generados

- `student_performance_transformado_numerico.csv`
- `train_student_performance.csv`
- `test_student_performance.csv`
- `predicciones_exam_score.csv`
- `ridge_alpha_10.pkl` (modelo final)

### 🎯 Próximos Pasos

- [ ] Interfaz web
- [ ] Integración con APIs
- [ ] Dashboard interactivo

---

**Machine Learning 2025** - Facultad de Ingeniería
