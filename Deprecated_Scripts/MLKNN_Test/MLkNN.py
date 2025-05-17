#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:09:46 2025

@author: sebastiangarcia
"""

# Importación de la librería pandas para la manipulación de datos
import pandas as pd
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


# ***** CARGA DE EMOCIONES *****
#Abrir archivo de emociones y ponerlas en un array (posición del 0 al 27)
emotion_file = open("/Users/sebastiangarcia/Downloads/GoEmotions/emotions.txt", "r")
emotion_list = emotion_file.read()
emotion_list = emotion_list.split("\n")
print(emotion_list)


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


print("\nTabla con emociones")
print(train_data)
print(valid_data)
print(test_data)



#Redefinir los grupos de sentimientos
positive = {'admiration', 'amusement', 'approval', 'caring', 'desire',
            'excitement', 'gratitude', 'joy', 'love', 'optimism', 
            'pride', 'relief'}
negative = {'anger', 'annoyance', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'fear', 'grief', 'nervousness',
            'remorse', 'sadness'}
ambiguous = {'confusion', 'curiosity', 'realization', 'surprise'}
neutral = {'neutral'}

#Función que asigna múltiples etiquetas de sentimiento
def multilabel_sentiment(emotion_list):
    sentiments = []
    if any(e in positive for e in emotion_list):
        sentiments.append("positive")
    if any(e in negative for e in emotion_list):
        sentiments.append("negative")
    if any(e in ambiguous for e in emotion_list):
        sentiments.append("ambiguous")
    if any(e in neutral for e in emotion_list):
        sentiments.append("neutral")
    return sentiments

# Aplicar a los tres datasets
train_data['Sentiment_labels'] = train_data['Emotions'].apply(multilabel_sentiment)
test_data['Sentiment_labels'] = test_data['Emotions'].apply(multilabel_sentiment)
valid_data['Sentiment_labels'] = valid_data['Emotions'].apply(multilabel_sentiment)

print(train_data)
print(valid_data)
print(test_data)



# Filtrar filas en las que la lista de sentimientos tenga dos o más elementos
multiple_sentiments = train_data[train_data['Sentiment_labels'].apply(lambda x: len(x) >= 2)]

# Imprimir los registros filtrados
print("Registros con dos o más sentimientos:")
print(multiple_sentiments)



from sklearn.preprocessing import MultiLabelBinarizer

# Definir las clases de sentimientos en el orden deseado
sentiment_classes = ["positive", "negative", "ambiguous", "neutral"]

# Crear una instancia del binarizador
mlb_sentiment = MultiLabelBinarizer(classes=sentiment_classes)

# Aplicar la transformación sobre la columna 'Sentiment_labels'
train_sentiment_binary = mlb_sentiment.fit_transform(train_data["Sentiment_labels"])
test_sentiment_binary = mlb_sentiment.fit_transform(test_data["Sentiment_labels"])


# Convertir la matriz binarizada en un DataFrame con nombres de columnas
train_sentiment_binary_df = pd.DataFrame(train_sentiment_binary, columns=mlb_sentiment.classes_, index=train_data.index)
test_sentiment_binary_df = pd.DataFrame(test_sentiment_binary, columns=mlb_sentiment.classes_, index=test_data.index)

# Combinar el DataFrame original con el DataFrame de binarización
train_data = pd.concat([train_data, train_sentiment_binary_df], axis=1)
test_data = pd.concat([test_data, test_sentiment_binary_df], axis=1)


# Imprimir las columnas de sentimientos originales y su versión binarizada
print(train_data[["Text", "positive", "negative", "ambiguous", "neutral"]])
print(test_data[["Text", "positive", "negative", "ambiguous", "neutral"]])



# --- Monkey patch para NearestNeighbors ---
from sklearn.neighbors import NearestNeighbors

def patched_get_param_names(self):
    return ['n_neighbors', 'algorithm', 'leaf_size', 'metric', 'p', 'metric_params', 'n_jobs']

NearestNeighbors._get_param_names = patched_get_param_names


from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score

# ***** EXTRACCIÓN DE ETIQUETAS COMO ARRAY DE NUMPY *****
import numpy as np
# Extraer las etiquetas usando np.asarray, seleccionando las columnas por nombre.
y_train = np.asarray(train_data[["positive", "negative", "ambiguous", "neutral"]])
y_test  = np.asarray(test_data[["positive", "negative", "ambiguous", "neutral"]])

# ***** VECTORIZACIÓN DE TEXTO CON TF-IDF *****
X_train_text = train_data["Text"].values
X_test_text  = test_data["Text"].values


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    strip_accents="unicode",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf  = vectorizer.transform(X_test_text)

# ***** ENTRENAMIENTO Y PREDICCIÓN CON MLkNN *****
classifier = MLkNN(ignore_first_neighbours=0, k=3, s=1.0)
classifier.fit(X_train_tfidf, y_train)
predictions = classifier.predict(X_train_tfidf)


new_sentences = ["I like the food but I hate the place"] 
new_sentence_tfidf = vectorizer.transform(new_sentences)
                                          
predicted_sentences = classifier.predict(new_sentence_tfidf)
print(predicted_sentences.toarray())

# ***** EVALUACIÓN *****
predicted = classifier.predict(X_test_tfidf)

print(accuracy_score(y_test, predicted))
print(hamming_loss(y_test, predicted))



#Probar un SVM multietiqueta, buscar algoritmos y cambiar la aplicacion pero con los 27 sentimientos (Aún no)


#Generar estadísticas y guardar en una tabla 

#hacer gráficas en excel también

#Sacar un conteo de cuantos difieren en sentimientos

#matriz de confusion para ekman y sentiment level

#Buscar algoritmo de clasificación multietiqueta basado en árboles de decisión







