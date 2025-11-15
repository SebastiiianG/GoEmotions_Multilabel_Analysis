# Tesis: Clasificación de Emociones con Selección de Features Chi-cuadrado

## 🎯 Resumen Ejecutivo

Esta tesis investiga la **clasificación automática de emociones en texto** utilizando técnicas de **selección de features Chi-cuadrado (χ²)** para mejorar el rendimiento de algoritmos de machine learning. El proyecto compara múltiples enfoques de clasificación (multiclase vs multilabel) y evalúa cuatro algoritmos diferentes con el dataset **GoEmotions** de Google.

### Problema de Investigación
- **¿Cómo afecta la selección de features Chi-cuadrado al rendimiento de clasificación de emociones?**
- **¿Qué algoritmo de ML funciona mejor para clasificación de emociones con diferentes números de features?**
- **¿Es más efectivo el enfoque multiclase o multilabel para este problema?**

### Contribuciones Principales
1. **Mapeo sistemático** de 28 emociones GoEmotions a 6 categorías básicas de Ekman
2. **Comparación exhaustiva** de 4 algoritmos ML con diferentes valores de Chi-cuadrado
3. **Metodología reproducible** para clasificación de emociones con selección de features
4. **Análisis detallado** del impacto de la reducción dimensional en el rendimiento

---

## 📊 Dataset y Metodología

### Dataset: GoEmotions (Google Research)
- **28 emociones originales**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **Mapeo a 6 categorías Ekman**: anger, disgust, fear, joy, sadness, surprise, neutral
- **Distribución**: ~43K entrenamiento, ~5.4K validación, ~5.4K prueba

### Preprocesamiento
1. **Lematización** con WordNetLemmatizer (NLTK)
2. **Vectorización TF-IDF** (max_features=10,000)
3. **Selección Chi-cuadrado** con k ∈ {500, 1000, 2000, 3000, 5000}
4. **Filtrado de conflictos** (eliminación de textos con múltiples categorías Ekman)

---

## 🧪 Experimentos Realizados

### Fase 1: Preparación de Datos
```
CSV_Scripts/CSV_Ekman.ipynb
```
- Mapeo GoEmotions → Ekman
- Eliminación de registros conflictivos
- Generación de visualizaciones de distribución

### Fase 2: Clasificación Multiclase (Ekman)
```
Experiments/Multiclass_Experiments/
├── Multiple_Emotions_Models.ipynb    # Comparación inicial
├── Multiple_emotions.ipynb           # Con aumentación de datos  
├── EkmanExperiment.ipynb            # Experimentos Ekman
└── Multiple_emotions_refactored.ipynb # Versión final optimizada
```

### Fase 3: Clasificación Multilabel (GoEmotions Original)
```
Experiments/MultilabelExperiments/
├── Chi2_Complete_Comparison.ipynb    # Comparación completa 4 modelos
├── Chi2_Multilabel.ipynb            # Experimentos base
└── Neural_Chi2.ipynb                # Específico para MLP
```

### Fase 4: Hiperparametrización
```
Experiments/MultilabelExperiments/
├── RF_Hyperparameter_Chi2_500.ipynb     # Random Forest (k=500)
├── SVM_Hyperparameter_Chi2_1000.ipynb   # SVM (k=1000)
├── MLP_Hyperparameter_Chi2_1000.ipynb   # MLP (k=1000)
└── MLkNN_Hyperparameter_Chi2_1000.ipynb # MLkNN (k=1000)
```

---

## 🤖 Algoritmos Evaluados

| Algoritmo | Tipo | Hiperparámetros Optimizados | Mejor Chi2 |
|-----------|------|----------------------------|------------|
| **SVM** | Multilabel | C, gamma, kernel, class_weight | k=1000 |
| **Random Forest** | Multilabel | n_estimators, max_depth, min_samples_split | k=500 |
| **MLP** | Multilabel | hidden_layers, activation, alpha, learning_rate | k=1000 |
| **MLkNN** | Multilabel | k_neighbors, smoothing_parameter | k=1000 |

---

## 📈 Resultados Principales

### Clasificación Multilabel (Mejor Configuración)

| Modelo | Chi2 | Accuracy | F1-Macro | Recall-Macro | Observaciones |
|--------|------|----------|----------|--------------|---------------|
| **MLP** | 1000 | 0.3245 | **0.3669** | 0.6234 | Mejor balance general |
| **Random Forest** | 500 | **0.3262** | 0.3495 | 0.5891 | Mejor accuracy |
| **SVM** | 1000 | 0.3156 | 0.3401 | **0.7017** | Mejor recall |
| **MLkNN** | 1000 | 0.3300 | 0.3287 | 0.6145 | Buen balance |

### Hallazgos Clave
1. **MLP** logra el mejor F1-macro (0.3669) con 1000 features
2. **Random Forest** obtiene la mejor accuracy (0.3262) con solo 500 features
3. **SVM** alcanza el mejor recall (0.7017) pero menor precision
4. **Reducción de features** mejora eficiencia sin pérdida significativa de rendimiento

---

## 🗂️ Estructura del Proyecto

```
Tesis_Chi2/
├── 📁 Data/                          # Datasets y archivos procesados
│   ├── GoEmotions/                   # Dataset original + mapeos
│   │   ├── emotions.txt              # Lista de 28 emociones
│   │   ├── ekman_mapping.json        # Mapeo a categorías Ekman
│   │   └── test.tsv                  # Datos de prueba originales
│   ├── BasedOnEkman/                 # Datos procesados para multiclase
│   ├── train_indexado.csv            # Entrenamiento indexado
│   ├── valid_indexado.csv            # Validación indexada
│   └── test_indexado.csv             # Prueba indexada
├── 📁 Experiments/                   # Experimentos principales
│   ├── Multiclass_Experiments/       # Clasificación multiclase (Ekman)
│   ├── MultilabelExperiments/        # Clasificación multilabel (GoEmotions)
│   └── Preparation/                  # Preparación y análisis preliminar
├── 📁 CSV_Scripts/                   # Scripts de procesamiento
│   ├── CSV_Ekman.ipynb              # Mapeo y limpieza Ekman
│   ├── CSV_Sentiments.ipynb         # Análisis de sentimientos
│   └── CSV_MultipleEmotionsToOne.ipynb
├── 📁 Data_Preparation/              # Indexación y preparación
└── 📁 Plots/                        # Visualizaciones generadas
    ├── Experiment1/                  # Gráficos multiclase
    └── Experiment2/                  # Gráficos multilabel
```

---

## 🚀 Guía de Ejecución Paso a Paso

### Prerequisitos
```bash
pip install pandas numpy scikit-learn scikit-multilearn nltk matplotlib seaborn scipy
```

### Orden de Ejecución Recomendado

#### 1️⃣ Preparación de Datos
```bash
# Ejecutar en orden:
CSV_Scripts/CSV_Ekman.ipynb                    # Mapeo Ekman + limpieza
CSV_Scripts/CSV_Sentiments.ipynb               # Análisis sentimientos
Data_Preparation/Indexation.ipynb              # Indexación de datos
```

#### 2️⃣ Experimentos Base
```bash
# Multiclase (Ekman):
Experiments/Multiclass_Experiments/Multiple_Emotions_Models.ipynb
Experiments/Multiclass_Experiments/EkmanExperiment.ipynb

# Multilabel (GoEmotions):
Experiments/MultilabelExperiments/Chi2_Multilabel.ipynb
Experiments/MultilabelExperiments/Neural_Chi2.ipynb
```

#### 3️⃣ Comparación Completa
```bash
Experiments/MultilabelExperiments/Chi2_Complete_Comparison.ipynb
```

#### 4️⃣ Hiperparametrización (Basada en resultados del paso 3)
```bash
Experiments/MultilabelExperiments/RF_Hyperparameter_Chi2_500.ipynb
Experiments/MultilabelExperiments/SVM_Hyperparameter_Chi2_1000.ipynb
Experiments/MultilabelExperiments/MLP_Hyperparameter_Chi2_1000.ipynb
Experiments/MultilabelExperiments/MLkNN_Hyperparameter_Chi2_1000.ipynb
```

#### 5️⃣ Análisis Final
```bash
Experiments/Multiclass_Experiments/Multiple_emotions_refactored.ipynb
```

---

## 📊 Métricas de Evaluación

### Métricas Principales
- **Accuracy**: Precisión general del modelo
- **F1-Score (macro avg)**: Promedio macro del F1-score por clase
- **Recall (macro avg)**: Promedio macro del recall por clase
- **Precision (macro avg)**: Promedio macro de la precisión por clase

### Validación
- **Validación cruzada**: 3-fold CV para hiperparametrización
- **Reproducibilidad**: `random_state=42` en todos los experimentos
- **Paralelización**: `n_jobs=-1` para optimizar tiempo de ejecución

---

## 🔬 Metodología Científica

### Diseño Experimental
1. **Variable independiente**: Número de features seleccionadas por Chi-cuadrado
2. **Variables dependientes**: Accuracy, F1-macro, Recall-macro, Precision-macro
3. **Controles**: Mismo preprocesamiento, misma división train/val/test, mismo random_state

### Validación de Resultados
- Múltiples ejecuciones con diferentes semillas
- Comparación estadística entre modelos
- Análisis de significancia de diferencias

---

## 📝 Conclusiones y Trabajo Futuro

### Conclusiones Principales
1. **Chi-cuadrado es efectivo** para reducir dimensionalidad manteniendo rendimiento
2. **MLP con 1000 features** ofrece el mejor balance F1-macro
3. **Random Forest con 500 features** es más eficiente computacionalmente
4. **Clasificación multilabel** es más apropiada que multiclase para este dominio

### Trabajo Futuro
- [ ] Experimentar con embeddings pre-entrenados (BERT, RoBERTa)
- [ ] Implementar ensemble methods combinando mejores modelos
- [ ] Evaluar con otros datasets de emociones (EmoBank, ISEAR)
- [ ] Análisis de interpretabilidad de features seleccionadas
- [ ] Optimización de hiperparámetros con Bayesian Optimization

---

## 🛠️ Dependencias Técnicas

```python
# requirements.txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-multilearn>=0.2.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Instalación Completa
```bash
# Clonar repositorio
git clone <repository-url>
cd Tesis_Chi2

# Instalar dependencias
pip install -r requirements.txt

# Descargar recursos NLTK
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 📚 Referencias y Contexto Académico

### Dataset
- **GoEmotions**: Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." ACL 2020.

### Metodología
- **Categorías Ekman**: Ekman, P. (1992). "An argument for basic emotions." Cognition & Emotion.
- **Chi-cuadrado**: Liu, H., & Setiono, R. (1995). "Chi2: Feature selection and discretization of numeric attributes."

### Algoritmos
- **Scikit-learn**: Pedregosa, F., et al. (2011). "Scikit-learn: Machine learning in Python." JMLR.
- **Scikit-multilearn**: Szymański, P., & Kajdanowicz, T. (2017). "A scikit-based Python environment for performing multi-label classification."

---

## 👥 Información del Proyecto

**Autor**: [Tu Nombre]  
**Institución**: [Tu Universidad]  
**Programa**: [Tu Programa de Estudios]  
**Director de Tesis**: [Nombre del Director]  
**Fecha**: [Fecha de Finalización]

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT/Apache 2.0/etc.] - ver el archivo `LICENSE` para más detalles.

---

*Para preguntas específicas sobre la implementación o resultados, consultar los notebooks individuales que contienen análisis detallados y comentarios explicativos.*