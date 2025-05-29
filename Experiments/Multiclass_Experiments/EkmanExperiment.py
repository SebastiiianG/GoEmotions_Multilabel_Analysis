
import pandas as pd
import json
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


#Carga de datos
train_data = pd.read_csv("./Data/BasedOnEkman/train_ekman.csv")
test_data = pd.read_csv("./Data/BasedOnEkman/test_ekman.csv")

#Llenar los Textos vacíos con un espacio
train_data['Text'] = train_data['Text'].fillna(" ")

# Cargar el mapeo de Ekman desde el archivo JSON
with open("./Data/GoEmotions/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)
ekman_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
ekman_to_id = {label: idx for idx, label in enumerate(ekman_labels)}

# --- PREPROCESAMIENTO PARA LEMATIZACIÓN ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

# Datos
X_train_raw = train_data['Text'].values
X_test_raw = test_data['Text'].values
y_train = train_data['Emotion'].values
y_test = test_data['Emotion'].values

# Datos lematizados
X_train_lem = [preprocess_text(text) for text in X_train_raw]
X_test_lem = [preprocess_text(text) for text in X_test_raw]

# --- DIRECTORIO DE SALIDA DE LOS PLOTS ---
output_dir = "./Plots/Experiment_Multiclass/Ekman/"


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Función para graficar matriz de confusión
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues", 
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
    # Guardar archivo
    filename = title.replace(" ", "_").replace("(", "").replace(")", "").lower() + ".png"
    plt.savefig(output_dir + filename)
    plt.close()
    
    

# --- FUNCION GENERAL PARA COMPARAR ---
def train_and_evaluate(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # Mostrar matriz de confusión
    
    plot_confusion_matrix(y_test, y_pred, ekman_labels, label)

# --- VECTORIZADORES ---
vectorizer_raw = TfidfVectorizer()
vectorizer_lem = TfidfVectorizer()

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
