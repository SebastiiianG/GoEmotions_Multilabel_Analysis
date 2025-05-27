#El objetivo de este experimento es tomar los registros que tienen solo una categoría de emoción y usarlos como datos de entrenamiento para un clasificador de sentimientos.
#Los registros que tienen más de una categoría se utilizarán para el testing y que así se determine una emoción dominante.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


#Carga de datos
train_data = pd.read_csv("./Data/OnlyOneEmotion/train_emotions.csv")
test_data = pd.read_csv("./Data/OnlyOneEmotion/test_emotions.csv")

#Llenar los Textos vacíos con un espacio
train_data['Text'] = train_data['Text'].fillna(" ")
test_data['Text'] = test_data['Text'].fillna(" ")

#Hacer un split de train_data para obtener un conjunto de validación
train, valid = train_test_split(train_data, test_size=0.2, random_state=42)

#Vectorizar los textos
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['Text'])
X_valid = vectorizer.transform(valid['Text'])
X_test = vectorizer.transform(test_data['Text'])

y_train = train['Emotion']
y_valid = valid['Emotion']
y_test = test_data['Emotion']

# Modelo (ejemplo con SVM)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Validación
y_pred_valid = clf.predict(X_valid)
print("Validación:")
print(classification_report(y_valid, y_pred_valid, zero_division=0))


#Contar las emociones en el conjunto de entrenamiento
print(train['Emotion'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix