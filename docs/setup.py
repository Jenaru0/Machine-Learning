"""
Configuración de instalación del paquete
=======================================

Configuración para instalar el proyecto como un paquete Python
usando pip install -e .

Autor: Equipo Grupo 4
Fecha: 2025
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer el README para la descripción larga
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Proyecto Integrador de Machine Learning - Predicción de Rendimiento Académico"

# Leer requirements.txt para las dependencias
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "ydata-profiling>=4.0.0",
        "jupyter>=1.0.0"
    ]

setup(
    name="proyecto-integrador-ml",
    version="1.0.0",
    author="Equipo Grupo 4",
    author_email="grupo4@ml.com",
    description="Proyecto Integrador de Machine Learning - Predicción de Rendimiento Académico",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grupo4/proyecto-integrador-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "sphinx>=4.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-pipeline=src.main:main",
            "ml-eda=src.eda:main",
            "ml-preprocess=src.preprocesamiento:main",
            "ml-train=src.entrenar_modelo:main",
            "ml-predict=src.predecir:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "datos": ["raw/*.csv", "procesados/*.csv"],
        "modelos": ["*.pkl"],
    },
    project_urls={
        "Bug Reports": "https://github.com/grupo4/proyecto-integrador-ml/issues",
        "Source": "https://github.com/grupo4/proyecto-integrador-ml",
        "Documentation": "https://github.com/grupo4/proyecto-integrador-ml/wiki",
    },
    keywords=[
        "machine learning",
        "education",
        "student performance",
        "regression",
        "data science",
        "predictive modeling",
        "academic analytics",
    ],
    license="MIT",
    zip_safe=False,
)
