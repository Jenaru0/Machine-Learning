# 📋 Historial de Cambios

Todos los cambios importantes de este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-09

### ✨ Agregado

#### Pipeline Completo de Machine Learning

- **Análisis Exploratorio de Datos (EDA)**

  - Estadísticas descriptivas completas
  - Análisis de correlaciones
  - Detección de valores nulos y duplicados
  - Análisis de distribuciones

- **Preprocesamiento de Datos**

  - Limpieza automática de datos
  - Ingeniería de características
  - Codificación de variables categóricas
  - División train/test (70/30)

- **Modelado y Optimización**

  - Regresión Lineal Simple
  - Ridge Regression con optimización de hiperparámetros
  - Lasso Regression con validación cruzada
  - Comparación automática de modelos

- **Predicción y Evaluación**
  - Generación de predicciones finales
  - Métricas de evaluación (R², RMSE, MAE, MSE)
  - Análisis de importancia de características
  - Guardado de modelos entrenados

#### Estructura del Proyecto

- **Código modular** en el directorio `src/`
- **Configuración centralizada** en `src/config.py`
- **Utilidades reutilizables** en `src/utils.py`
- **Pruebas unitarias** en `tests/`
- **Notebooks interactivos** en `notebooks/`

#### Automatización

- **Pipeline ejecutable** con un solo comando
- **Scripts de ejecución** para Windows (.bat) y Linux/macOS (.sh)
- **Logging comprehensivo** para seguimiento de ejecución
- **Manejo robusto de errores**

#### Documentación

- **README.md** completo con instrucciones claras
- **CONTRIBUTING.md** para guías de contribución
- **INSTALL.md** con instrucciones de instalación
- **Docstrings** en todas las funciones
- **Comentarios explicativos** en español

#### Calidad del Código

- **Pruebas unitarias** con pytest
- **Convenciones de código** siguiendo PEP 8
- **Separación de responsabilidades**
- **Configuración de Git** con .gitignore

### 📊 Resultados Técnicos

#### Métricas del Modelo

- **R² Score**: 0.6926 (69.26% de varianza explicada)
- **RMSE**: 2.0552
- **MAE**: 1.0352
- **MSE**: 4.224

#### Variables Más Importantes

1. **Attendance** (2.29) - Asistencia a clases
2. **Hours_Studied** (1.57) - Horas de estudio
3. **Previous_Scores** (0.81) - Puntajes anteriores
4. **Tutoring_Sessions** (0.55) - Sesiones de tutoría
5. **Peer_Influence** (0.40) - Influencia de pares

#### Rendimiento del Pipeline

- **Tiempo total de ejecución**: 36.90 segundos
- **EDA**: 1.70s
- **Preprocesamiento**: 0.23s
- **Entrenamiento**: 34.80s
- **Predicción**: 0.16s

### 🗂️ Archivos Generados

#### Datos Procesados

- `student_performance_transformado_numerico.csv` - Dataset final numérico
- `train_student_performance.csv` - Conjunto de entrenamiento
- `test_student_performance.csv` - Conjunto de prueba
- `predicciones_exam_score.csv` - Predicciones finales

#### Modelos Entrenados

- `ridge_alpha_10.pkl` - Mejor modelo Ridge (alpha=10.0)
- `scaler.pkl` - Escalador StandardScaler

#### Reportes y Visualizaciones

- Gráficos de comparación de modelos
- Análisis de residuos
- Matrices de correlación

### 🧪 Pruebas y Validación

#### Pruebas Unitarias

- `test_preprocesamiento.py` - 4 pruebas pasadas
- Cobertura de funciones críticas
- Validación de integridad de datos

#### Validación del Pipeline

- ✅ Carga de datos: 6,607 registros
- ✅ Preprocesamiento: 18 características finales
- ✅ Entrenamiento: 3 modelos optimizados
- ✅ Predicciones: 1,983 valores generados

### 🔧 Configuración Técnica

#### Dependencias

- Python 3.8+
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- jupyter>=1.0.0
- pytest>=7.0.0

#### Compatibilidad

- Windows 10/11
- macOS 10.14+
- Linux Ubuntu 18.04+

### 📋 Estructura de Archivos

```
proyecto-integrador-ml/
├── datos/
│   ├── raw/StudentPerformanceFactors.csv
│   ├── procesados/[archivos generados]
│   └── profiling/
├── modelos/
│   ├── ridge_alpha_10.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── exploracion_interactiva.ipynb
│   └── exploracion_interactiva.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── eda.py
│   ├── preprocesamiento.py
│   ├── entrenar_modelo.py
│   └── predecir.py
├── tests/
│   └── test_preprocesamiento.py
├── ejecutar_pipeline.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── INSTALL.md
├── LICENSE
├── .gitignore
└── CHANGELOG.md
```

## 🚀 Próximos Pasos Planificados

### [2.0.0] - Planificado

#### Interfaz Web

- [ ] Dashboard interactivo con Streamlit/Flask
- [ ] Carga de datos personalizada
- [ ] Visualización de predicciones en tiempo real
- [ ] Reportes descargables

#### Integración de APIs

- [ ] APIs de IA externa (OpenAI, Hugging Face)
- [ ] Endpoints RESTful para predicciones
- [ ] Documentación de API con Swagger
- [ ] Autenticación y rate limiting

#### Mejoras del Modelo

- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Hyperparameter tuning avanzado con Optuna
- [ ] Validación cruzada extendida
- [ ] Análisis de feature importance avanzado

#### Escalabilidad

- [ ] Containerización con Docker
- [ ] CI/CD con GitHub Actions
- [ ] Despliegue en la nube (AWS/GCP/Azure)
- [ ] Monitoreo de modelos en producción

## 📞 Equipo de Desarrollo

### Integrantes

- **Candela Vargas Aitor Baruc** - Desarrollo y documentación
- **Godoy Bautista Denilson Miguel** - Análisis de datos y modelado
- **Molina Lazaro Eduardo Jeampier** - Preprocesamiento y pruebas
- **Napanga Ruiz Jhonatan Jesus** - Optimización y validación
- **Quispe Romani Angela Isabel** - Documentación y presentación

### Información Académica

- **Asignatura:** Machine Learning
- **Docente:** M.SC. Magaly Roxana Aranguena Yllanes
- **Institución:** Facultad de Ingeniería - Escuela Profesional de Ingeniería de Sistemas
- **Año:** 2025

---

## 🏆 Reconocimientos

- **Mejor Pipeline de ML** - Implementación completa y funcional
- **Código Limpio** - Estructura modular y bien documentada
- **Reproducibilidad** - Resultados consistentes y automatizados
- **Documentación Excepcional** - Guías claras para principiantes

## [1.1.0] - 2025-07-13

### 🎯 Decisión Estratégica: Modelos Híbridos

#### Análisis de Rendimiento

- **Ridge Regression (Modelo Base):**

  - R² Score: 0.6926 (69.26% varianza explicada)
  - RMSE: 2.0552
  - MAE: 1.0352
  - ✅ Cumple exactamente las guías académicas
  - ✅ Interpretable y rápido

- **SVR (Modelo Avanzado):**
  - R² Score: 0.7561 (75.61% varianza explicada)
  - RMSE: 1.8284
  - MAE: 0.9127
  - 🏆 **Mejora del 9.2% en R²**
  - 🏆 **Mejora del 11.0% en RMSE**

#### Estrategia Implementada

- **Modelo Principal:** Ridge (para cumplimiento académico)
- **Modelo Avanzado:** SVR (para puntos extra y demo técnica)
- **Pipeline Flexible:** Ambos modelos disponibles según necesidad

#### Justificación Técnica

1. **Seguridad Académica:** Ridge garantiza cumplimiento de guías
2. **Excelencia Técnica:** SVR demuestra dominio avanzado de ML
3. **Flexibilidad:** Opciones para diferentes escenarios de presentación
4. **Aprendizaje Completo:** Experiencia con modelos lineales y no lineales

### ✨ Nuevos Componentes Agregados

#### Comparación Avanzada de Modelos

- **Random Forest:** R² = 0.6716
- **Gradient Boosting:** R² = 0.7460
- **SVR:** R² = 0.7561 ⭐ (Ganador)
- **XGBoost:** R² = 0.7480
- **Red Neuronal MLP:** R² = 0.6608

#### Pipeline Mejorado

- `src/05_comparar_modelos_avanzados.py` - Comparación exhaustiva
- Validación cruzada con 5 folds
- Detección automática de overfitting
- Guardado de metadatos detallados
- Ranking automático de modelos

#### Documentación Estratégica

- `ESTRATEGIA_MODELOS.md` - Guía de decisión técnica
- Análisis de trade-offs Ridge vs SVR
- Recomendaciones para diferentes escenarios
- Criterios de selección de modelos
