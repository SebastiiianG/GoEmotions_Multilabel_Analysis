
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ***** CARGA DE DATOS *****


train_data = pd.read_csv("/Users/sebastiangarcia/Downloads/GoEmotions/train.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID']) 
test_data = pd.read_csv("/Users/sebastiangarcia/Downloads/GoEmotions/test.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])


# ***** PROCESAMIENTO DE LOS DATOS *****

# Para cada fila en la columna 'Class' (que contiene múltiples emociones separadas por comas), se divide el texto en una lista de emociones. Es decirm si se tiene 6, 4 dentro de un mismo objeto, se separan.

train_data['List of classes'] = train_data['Class'].apply(lambda x: x.split(','))
train_data['Len of classes'] = train_data['List of classes'].apply(lambda x: len(x))

test_data['List of classes'] = test_data['Class'].apply(lambda x: x.split(','))
test_data['Len of classes'] = test_data['List of classes'].apply(lambda x: len(x))


# ***** CARGA DE EMOCIONES *****
#Abrir archivo de emociones y ponerlas en un array (posición del 0 al 27)
emotion_file = open("/Users/sebastiangarcia/Downloads/GoEmotions/emotions.txt", "r")
emotion_list = emotion_file.read()
emotion_list = emotion_list.split("\n")


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
test_data['Emotions'] = test_data['List of classes'].apply(idx2class)


# Simplificar: solo tomar la primera emoción de la lista
train_data['Primary_Emotion'] = train_data['Emotions'].apply(lambda x: x[0])
test_data['Primary_Emotion'] = test_data['Emotions'].apply(lambda x: x[0])

# Actualizar y_train y y_test
y_train = train_data['Primary_Emotion'].values
y_test = test_data['Primary_Emotion'].values


# --- PREPROCESAMIENTO PARA LEMATIZACIÓN ---
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

# Datos
X_train_raw = train_data['Text'].values
X_test_raw = test_data['Text'].values
y_train = train_data['Primary_Emotion'].values
y_test = test_data['Primary_Emotion'].values

# Datos lematizados
X_train_lem = [preprocess_text(text) for text in X_train_raw]
X_test_lem = [preprocess_text(text) for text in X_test_raw]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Función para graficar matriz de confusión
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="bwr",  # Blue for acierto, Red for fallo
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(f'Matriz de Confusión - {title}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    

# --- FUNCION GENERAL PARA COMPARAR ---
def train_and_evaluate(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # Mostrar matriz de confusión
    emotions_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", 
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", 
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", 
    "surprise", "neutral"]
    
    plot_confusion_matrix(y_test, y_pred, emotions_labels, label)

# --- VECTORIZADORES ---
vectorizer_raw = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode", max_features=5000)
vectorizer_lem = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode", max_features=5000)

X_train_raw_tfidf = vectorizer_raw.fit_transform(X_train_raw)
X_test_raw_tfidf = vectorizer_raw.transform(X_test_raw)

X_train_lem_tfidf = vectorizer_lem.fit_transform(X_train_lem)
X_test_lem_tfidf = vectorizer_lem.transform(X_test_lem)

# --- MODELOS ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (linear)': SVC(kernel='linear', C=1),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy')
}



# --- COMPARATIVA FINAL ---
print("\n=== SIN LEMATIZAR ===")
for model_name, model in models.items():
    train_and_evaluate(model, X_train_raw_tfidf, y_train, X_test_raw_tfidf, y_test, f"{model_name} (sin lematizar)")

print("\n=== CON LEMATIZAR ===")
for model_name, model in models.items():
    train_and_evaluate(model, X_train_lem_tfidf, y_train, X_test_lem_tfidf, y_test, f"{model_name} (lematizado)")
