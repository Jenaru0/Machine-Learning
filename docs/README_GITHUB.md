# 🎓 Proyecto Integrador - Machine Learning

## Predicción de Rendimiento Académico

### 👥 Equipo Grupo 4
- Candela Vargas Aitor Baruc
- Godoy Bautista Denilson Miguel  
- Molina Lazaro Eduardo Jeampier
- Napanga Ruiz Jhonatan Jesus
- Quispe Romani Angela Isabel

### 📋 Descripción
Pipeline completo de Machine Learning para predecir el rendimiento académico usando el dataset `StudentPerformanceFactors.csv`.

### 🚀 Ejecutar el Proyecto

#### 1. Clonar el repositorio
```bash
git clone [URL_DEL_REPOSITORIO]
cd proyecto-integrador-ml
```

#### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 3. Ejecutar pipeline completo
```bash
python ejecutar_pipeline.py
```

**¡Eso es todo!** El pipeline se ejecutará automáticamente y generará todos los resultados.

### 📊 Resultados Esperados

- **R² Score**: 0.6926 (69.26% de varianza explicada)
- **RMSE**: 2.0552
- **Tiempo de ejecución**: ~37 segundos
- **Predicciones**: 1,983 valores generados

### 📁 Archivos Generados

El pipeline genera automáticamente:
- `datos/procesados/` - Datasets procesados
- `modelos/` - Modelos entrenados (Ridge + Scaler)
- `predicciones_exam_score.csv` - Predicciones finales

### 🔧 Estructura del Proyecto

```
proyecto-integrador-ml/
├── datos/
│   ├── raw/StudentPerformanceFactors.csv
│   └── procesados/
├── modelos/
├── notebooks/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── eda.py
│   ├── preprocesamiento.py
│   ├── entrenar_modelo.py
│   └── predecir.py
├── ejecutar_pipeline.py
├── requirements.txt
└── README.md
```

### 📈 Variables Más Importantes

1. **Attendance** - Asistencia a clases
2. **Hours_Studied** - Horas de estudio  
3. **Previous_Scores** - Puntajes anteriores
4. **Tutoring_Sessions** - Sesiones de tutoría
5. **Peer_Influence** - Influencia de pares

### 🎯 Objetivo Académico

Este proyecto implementa las 4 guías de aprendizaje:
- **Guía 1**: Análisis Exploratorio de Datos (EDA)
- **Guía 2**: Preprocesamiento de Datos
- **Guía 3**: Regresión Lineal Simple
- **Guía 4**: Optimización de Hiperparámetros

### 🧪 Validación

Para ejecutar las pruebas básicas:
```bash
python -m pytest tests/ -v
```

### 📚 Uso del Notebook

Para exploración interactiva:
```bash
jupyter notebook notebooks/exploracion_interactiva.ipynb
```

### 🔄 Próximos Pasos

- [ ] Interfaz web interactiva
- [ ] Integración con APIs de IA
- [ ] Mejoras del modelo
- [ ] Despliegue en la nube

---

### 📞 Contacto

**Asignatura:** Machine Learning  
**Docente:** M.SC. Magaly Roxana Aranguena Yllanes  
**Institución:** Facultad de Ingeniería - Ingeniería de Sistemas  
**Año:** 2025

---

**Estado:** ✅ Completado y funcional | **Última actualización:** 9 de julio de 2025
