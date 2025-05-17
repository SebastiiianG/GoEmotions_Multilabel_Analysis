import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix

# === CARGA DE DATOS ===
train_data = pd.read_csv("./Data/OnlyEmotions/train_emotions.csv")
test_data = pd.read_csv("./Data/OnlyEmotions/test_emotions.csv")

# Convertir strings de listas a listas reales

"""
Ejemplo:
Antes: "['admiration', 'gratitude']"
Después: ['admiration', 'gratitude']
"""
train_data['Emotions'] = train_data['Emotions'].apply(ast.literal_eval)
test_data['Emotions'] = test_data['Emotions'].apply(ast.literal_eval)

# TF-IDF VECTORIZACIÓN
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode", max_features=5000)
X_train = vectorizer.fit_transform(train_data['Text'].values)
X_test = vectorizer.transform(test_data['Text'].values)

# BINARIZACIÓN MULTILABEL
emotion_classes = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", 
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", 
    "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", 
    "surprise", "neutral"
]


mlb = MultiLabelBinarizer(classes=emotion_classes)
y_train = mlb.fit_transform(train_data['Emotions'])
y_test = mlb.transform(test_data['Emotions'])

#convertir a matriz dispersa de acuerdo con la documentación de skmultilearn
y_train_sparse = csr_matrix(y_train)

# MLkNN
mlknn = MLkNN(k=3)
mlknn.fit(X_train, y_train_sparse)
y_pred = mlknn.predict(X_test)

# EVALUACIÓN
print("\nResultados MLkNN")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación por emoción (Precision, Recall, F1):")
print(classification_report(y_test, y_pred, target_names=emotion_classes, zero_division=0))


# MATRICES DE CONFUSIÓN MULTIETIQUETA

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Directorio para los plots
output_dir = "./Plots/Experiment_Multilabel/MLKNN/" 

# FUNCIÓN PARA PLOT DE MATRIZ BINARIA POR EMOCIÓN
def plot_confusion_matrix_binary(y_true_col, y_pred_col, label):
    cm = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues", 
        xticklabels=["No", "Sí"],
        yticklabels=["No", "Sí"],
        cbar=True
    )
    plt.title(f"Matriz de Confusión - {label}")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    filename = output_dir + label.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.close()

# APLICAR MATRIZ DE CONFUSIÓN PARA CADA EMOCIÓN
for i, emotion in enumerate(emotion_classes):
    y_true_col = y_test[:, i].flatten()
    y_pred_col = y_pred[:, i].toarray().flatten()  
    plot_confusion_matrix_binary(y_true_col, y_pred_col, emotion)


"""
# === MATRICES DE CONFUSIÓN MULTIETIQUETA ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- DIRECTORIO DE SALIDA DE LOS PLOTS ---
output_dir = "./Plots/Experiment_Multilabel/MLKNN/" 

# --- FUNCIÓN PARA PLOT DE MATRIZ BINARIA POR EMOCIÓN ---
def plot_confusion_matrix_binary(y_true_col, y_pred_col, label):
    cm = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues", 
        xticklabels=["No", "Sí"],
        yticklabels=["No", "Sí"],
        cbar=True
    )
    plt.title(f"Matriz de Confusión - {label}")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    filename = output_dir + label.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.close()

# --- APLICAR MATRIZ DE CONFUSIÓN PARA CADA EMOCIÓN ---
for i, emotion in enumerate(emotion_classes):
    y_true_col = y_test[:, i].flatten()
    y_pred_col = y_pred[:, i].toarray().flatten()  
    plot_confusion_matrix_binary(y_true_col, y_pred_col, emotion)

# --- PLOT GENERAL CON TODAS LAS MATRICES ---
def plot_confusion_matrices_grid(y_true, y_pred, labels, ncols=5):
    n_labels = len(labels)
    nrows = (n_labels + ncols - 1) // ncols  # redondeo hacia arriba

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    for i, label in enumerate(labels):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        y_true_col = y_true[:, i].flatten()
        y_pred_col = y_pred[:, i].toarray().flatten()
        cm = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Sí"],
            yticklabels=["No", "Sí"],
            cbar=False,
            ax=ax
        )
        ax.set_title(label)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Verdadera")

    # Ocultar ejes vacíos si los hay
    total_plots = nrows * ncols
    for j in range(len(labels), total_plots):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.savefig(output_dir + "matriz_confusion_todas_emociones.png")
    plt.close()

# --- GENERAR PLOT GENERAL ---
plot_confusion_matrices_grid(y_test, y_pred, emotion_classes, ncols=5)
"""