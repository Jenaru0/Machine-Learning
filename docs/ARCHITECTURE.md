# 🏗️ ARQUITECTURA DEL PROYECTO

## 📊 Flujo de Datos

```
StudentPerformanceFactors.csv
           ↓
    [01_eda.py] → profiling/reporte_eda.html
           ↓
    [02_preprocesamiento.py] → train/test split + encoding
           ↓
    [03_entrenar_modelo.py] → Ridge (modelo principal)
           ↓
    [05_comparar_modelos_avanzados.py] → SVR + otros modelos
           ↓
    [04_predecir.py] → predicciones_exam_score.csv
```

## 🔧 Módulos del Sistema

| Módulo | Responsabilidad | Input | Output |
|--------|----------------|-------|--------|
| `00_config.py` | Configuración global | - | Parámetros del sistema |
| `00_utils.py` | Funciones compartidas | - | Utilidades comunes |
| `01_eda.py` | Análisis exploratorio | CSV crudo | Reporte HTML + insights |
| `02_preprocesamiento.py` | Limpieza de datos | CSV crudo | Train/test procesados |
| `03_entrenar_modelo.py` | Entrenamiento base | Datos procesados | Modelo Ridge + métricas |
| `04_predecir.py` | Predicciones | Modelo + datos | CSV predicciones |
| `05_comparar_modelos_avanzados.py` | Modelos avanzados | Datos procesados | Comparación + SVR |

## 🎯 Decisiones de Diseño

### ¿Por qué Pipeline Modular?

- **Mantenibilidad**: Cada módulo tiene una responsabilidad específica
- **Testabilidad**: Fácil testing de componentes individuales  
- **Flexibilidad**: Posibilidad de ejecutar partes específicas
- **Académico**: Sigue estructura de guías universitarias

### ¿Por qué Ridge como Principal?

- **Cumplimiento**: Especificado en guías académicas
- **Interpretabilidad**: Coeficientes fáciles de explicar
- **Estabilidad**: Menos propenso a overfitting
- **Simplicidad**: Ideal para presentación académica

### ¿Por qué SVR como Extensión?

- **Performance**: +9% mejora en R²
- **Innovación**: Demuestra dominio técnico avanzado
- **Comparación**: Permite análisis de trade-offs
- **Valor Agregado**: Diferencia el proyecto de otros

## 📁 Convenciones de Archivos

### Nomenclatura

```
XX_nombre_modulo.py  # Donde XX = orden de ejecución
```

### Datos
```
datos/
├── raw/              # Datos originales (no modificar)
├── procesados/       # Datos limpios y transformados
└── profiling/        # Reportes de análisis
```

### Modelos
```
modelos/
├── ridge_alpha_10.pkl           # Modelo principal académico
├── mejor_modelo_avanzado_svr.pkl # Mejor modelo técnico
└── scaler.pkl                   # Transformadores
```

## 🔄 Workflow de Desarrollo

1. **Setup**: Instalar dependencias con `requirements.txt`
2. **EDA**: Ejecutar análisis exploratorio
3. **Preprocessing**: Limpiar y transformar datos
4. **Base Models**: Entrenar Ridge, Lasso, Linear
5. **Advanced Models**: Experimentar con SVR, RF, XGB
6. **Evaluation**: Comparar todos los modelos
7. **Production**: Generar predicciones finales
8. **Documentation**: Actualizar análisis y README

## 🎓 Alineación Académica

### Guías Universitarias Cumplidas

- ✅ **Guía 1**: EDA completo con ydata-profiling
- ✅ **Guía 2**: Preprocesamiento robusto 
- ✅ **Guía 3**: Ridge, Lasso, Linear implementados
- ✅ **Guía 4**: Validación cruzada e hiperparámetros

### Extensiones Técnicas

- 🚀 **SVR**: Modelo avanzado para máxima precisión
- 🚀 **Ensemble Methods**: RF, Gradient Boosting, XGBoost
- 🚀 **Neural Networks**: MLP para deep learning
- 🚀 **Análisis Comparativo**: Trade-offs documentados
