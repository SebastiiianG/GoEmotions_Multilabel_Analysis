import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# CARGA DE DATOS 
train_data = pd.read_csv("./Data/Chi2/train_2000_chi2.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

# LLenar los registros vacíos del train
train_data['Text'] = train_data['Text'].fillna('')

# Definir las clases de emociones
emotion_classes = train_data.columns[2:].tolist()

# TF-IDF VECTORIZACIÓN
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode", max_features=5000)
X_train = vectorizer.fit_transform(train_data['Text'].values)
X_test = vectorizer.transform(test_data['Text'].values)
y_train = np.asarray(train_data[emotion_classes])
y_test = np.asarray(test_data[emotion_classes])

#Spliteo del conjunto de entrenamiento para validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Random Forest Classifier para clasificación multi-etiqueta
base_model = RandomForestClassifier(random_state=42)
rf_model = MultiOutputClassifier(base_model)
rf_model.fit(X_train, y_train)

#Predicción
y_pred = rf_model.predict(X_val)

# EVALUACIÓN

print("\nResultados Random Forest")
print("Accuracy:", accuracy_score(y_val, y_pred))

print("\nReporte de clasificación:\n", classification_report(y_val, y_pred, target_names=emotion_classes, zero_division=0))
"""
param_grid = {
    'estimator__n_estimators': [10, 100, 200, 300],
    'estimator__max_depth': [3, 5, 10, 20, None],
    'estimator__max_features': [None, 'sqrt', 'log2'],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__bootstrap': [True, False]
}

rf_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf= grid_search.best_estimator_

print("\nMejores parámetros encontrados:", grid_search.best_params_)
print("\nMejor score de validación:", grid_search.best_score_)
"""