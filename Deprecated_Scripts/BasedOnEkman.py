#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:16:09 2025

@author: sebastiangarcia
"""


# Importación de la librería pandas para la manipulación de datos
import pandas as pd


#Import del json para el ekman mapping
import json


# ***** CARGA DE DATOS *****
# Cargar los conjuntos de datos desde archivos TSV (valores separados por tabulaciones)

# El archivo no tiene encabezado, por lo que se definen las columnas manualmente: 
# 'Text' (texto del comentario), 'Class' (emociones asociadas) y 'ID' (identificador del comentario)

# Cada archivo representa una parte del dataset: entrenamiento, validación y prueba

train_data = pd.read_csv("/Users/sebastiangarcia/Downloads/GoEmotions/train.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID']) 
valid_data = pd.read_csv("/Users/sebastiangarcia/Downloads/GoEmotions/dev.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])
test_data = pd.read_csv("/Users/sebastiangarcia/Downloads/GoEmotions/test.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])


# ***** PROCESAMIENTO DE LOS DATOS *****

# Para cada fila en la columna 'Class' (que contiene múltiples emociones separadas por comas), se divide el texto en una lista de emociones. Es decirm si se tiene 6, 4 dentro de un mismo objeto, se separan.

train_data['List of classes'] = train_data['Class'].apply(lambda x: x.split(','))
train_data['Len of classes'] = train_data['List of classes'].apply(lambda x: len(x))

valid_data['List of classes'] = valid_data['Class'].apply(lambda x: x.split(','))
valid_data['Len of classes'] = valid_data['List of classes'].apply(lambda x: len(x))

test_data['List of classes'] = test_data['Class'].apply(lambda x: x.split(','))
test_data['Len of classes'] = test_data['List of classes'].apply(lambda x: len(x))

"""
# Ver datos procesados
print(train_data)
print(valid_data)
print(test_data)


# ***** VERIFICACIÓN DE DATOS *****
#Comprobar que no hay datos nulos
print("Conteo de datos nulos")
print(train_data.isnull().sum())
print(valid_data.isnull().sum())
print(test_data.isnull().sum())


#Contar etiquetas por df
print("\nEtiquetas por DF")
print(train_data["Class"].value_counts())
print(valid_data["Class"].value_counts())
print(test_data["Class"].value_counts())
"""


# ***** CARGA DE EMOCIONES *****
#Abrir archivo de emociones y ponerlas en un array (posición del 0 al 27)
emotion_file = open("/Users/sebastiangarcia/Downloads/GoEmotions/emotions.txt", "r")
emotion_list = emotion_file.read()
emotion_list = emotion_list.split("\n")
#print(emotion_list)


def count_sentiment_categories(df, reverse_ekman_map):
    """
    Cuenta cuántos registros tienen emociones que recaen en una sola categoría Ekman
    y cuántos recaen en más de una.

    Parámetros:
    - df: DataFrame con la columna 'Emotions' ya procesada.
    - reverse_ekman_map: diccionario que mapea cada emoción a su categoría Ekman.

    Retorna:
    - (solo_una_categoria, mas_de_una_categoria)
    """
    solo_una, multiples = 0, 0

    for emotions in df['Emotions']:
        categorias_presentes = set()
        for e in emotions:
            if e in reverse_ekman_map:
                categorias_presentes.add(reverse_ekman_map[e])
        if len(categorias_presentes) <= 1:
            solo_una += 1
        else:
            multiples += 1

    return solo_una, multiples

# ***** ASIGNACIÓN DE EMOCIONES *****
#Función que toma la lista de etiquetas numéricas por registro
#Para cada número, se le asigna la clase de acuerdo al índice del array de emociones
def idx2class(idx_list):
    arr = []
    for i in idx_list:
        arr.append(emotion_list[int(i)])
    return arr

#Aplicación de la función a los datos
train_data['Emotions'] = train_data['List of classes'].apply(idx2class)
valid_data['Emotions'] = valid_data['List of classes'].apply(idx2class)
test_data['Emotions'] = test_data['List of classes'].apply(idx2class)

"""

print("\nTabla con emociones")
print(train_data)
print(valid_data)
print(test_data)

"""

# ***** CATEGORIZACIÓN DE SENTIMIENTOS *****
# Función para categorizar emociones en números basada en texto
# Cargar el mapeo de Ekman desde el archivo
with open("/Users/sebastiangarcia/Downloads/GoEmotions/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

# Crear un diccionario inverso: emoción específica → emoción Ekman

# Asegurarse de incluir la categoría 'neutral' en el mapeo
ekman_mapping["neutral"] = ["neutral"]
    
# Diccionario inverso
reverse_ekman_map = {e: cat for cat, emos in ekman_mapping.items() for e in emos}

# Contar registros
solo_una_train, multiples_train = count_sentiment_categories(train_data, reverse_ekman_map)
solo_una_valid, multiples_valid = count_sentiment_categories(valid_data, reverse_ekman_map)
solo_una_test, multiples_test = count_sentiment_categories(test_data, reverse_ekman_map)


# Mostrar resultados
print("\nTrain:")
print("Solo una categoría:", solo_una_train)
print("Más de una categoría:", multiples_train)

print("\nValidation:")
print("Solo una categoría:", solo_una_valid)
print("Más de una categoría:", multiples_valid)

print("\nTest:")
print("Solo una categoría:", solo_una_test)
print("Más de una categoría:", multiples_test)


# Asignar un número a cada emoción Ekman
ekman_labels = list(ekman_mapping.keys())
ekman_to_id = {label: idx for idx, label in enumerate(ekman_labels)}

# Nueva función de categorización
def categorize_emotions(row):
    emotions = row['Emotions']
    for emotion in emotions:
        if emotion in reverse_ekman_map:
            ekman_category = reverse_ekman_map[emotion]
            return ekman_to_id[ekman_category]
    return len(ekman_labels)  # categoría 'otros' si no hay coincidencia

# Aplicar la función a la columna "Emotions"
train_data['Sentiment'] = train_data.apply(categorize_emotions, axis=1)
valid_data['Sentiment'] = valid_data.apply(categorize_emotions, axis=1)
test_data['Sentiment'] = test_data.apply(categorize_emotions, axis=1)

"""
# Mostrar los DataFrames modificados
print("\nTabla con sentimientos")
print(train_data[['Text', 'Sentiment']])
print(valid_data[['Text', 'Sentiment']])
print(test_data[['Text', 'Sentiment']])
"""

# ***** VECTORIZACIÓN DE TEXTOS *****
# Importar TfidfVectorizer, Regresión Logística y SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Usar el 100% de train_data para entrenamiento
sentences_train = train_data['Text'].values
y_train = train_data['Sentiment'].values

# Usar test_data para evaluación
sentences_test = test_data['Text'].values
y_test = test_data['Sentiment'].values

# Convertir los textos a vectores numéricos usando TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=False,                # Convertir todo a minúsculas
    stop_words="english",          # Eliminar palabras vacías en inglés (puedes cambiarlo según el idioma)
    strip_accents="unicode",       # Normalizar acentos
    max_features=5000              # Limitar el número de características
)
X_train = vectorizer.fit_transform(sentences_train)
X_test = vectorizer.transform(sentences_test)



# ***** MÉTRICAS ****+

# ***** REGRESIÓN LOGÍSTICA *****
logistic_model = LogisticRegression(max_iter=1000)
# Entrenar el modelo
logistic_model.fit(X_train, y_train)
# Hacer predicciones
y_pred_logistic = logistic_model.predict(X_test)
print("\n- Precisión usando Regresión Logística:", accuracy_score(y_test, y_pred_logistic))
print("\n- Reporte de clasificación (Regresión Logística):\n", classification_report(y_test, y_pred_logistic))

# ***** SVM *****
#Si C = 1 la precision es de 0.6347 mientras que si se sube (10) es de  0.5935, después de 10 se tarda mucho en ejecutar
svm_model = SVC(kernel='linear', C=1)  # Kernel lineal y C=1.0 
svm_model.fit(X_train, y_train)
# Hacer predicciones
y_pred_svm = svm_model.predict(X_test)
print("\n- Precisión usando SVM:", accuracy_score(y_test, y_pred_svm))
print("\n- Reporte de clasificación (SVM):\n", classification_report(y_test, y_pred_svm))



# ***** ÁRBOL DE DECISIÓN *****
tree_model = DecisionTreeClassifier(
    criterion='entropy',  # O 'entropy' {“gini”, “entropy”, “log_loss”}
    max_depth=None,  # Profundidad máxima del árbol (ajustar si hay overfitting)
    min_samples_split=2,  # Mínimo de muestras para dividir un nodo, default: 2
    random_state=None #Se puede modificar
)
# Entrenar el modelo
tree_model.fit(X_train, y_train)
# Hacer predicciones
y_pred_tree = tree_model.predict(X_test)
# Evaluar el modelo
print("\n- Precisión usando Árbol de Decisión:", accuracy_score(y_test, y_pred_tree))
print("\n- Reporte de clasificación (Árbol de Decisión):\n", classification_report(y_test, y_pred_tree))



print("\nÍndices de emociones Ekman:")
for label, idx in ekman_to_id.items():
    print(f"{idx}: {label}")
    
 
