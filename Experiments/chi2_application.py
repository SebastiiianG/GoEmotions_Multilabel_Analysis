import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# === CARGA DE DATOS ===
train_data = pd.read_csv("./Data/train_indexado.csv")
train_data['Emotions'] = train_data['Emotions'].apply(ast.literal_eval)

# === TF-IDF (Reducir a 10,000 características más frecuentes) ===
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    max_features=10000
)
X_tfidf = vectorizer.fit_transform(train_data['Text'])
feature_names = vectorizer.get_feature_names_out()

# === BINARIZACIÓN MULTILABEL ===
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(train_data['Emotions'])

# === CÁLCULO MANUAL DE CHI2 PARA CADA FEATURE Y ETIQUETA ===
# Acumulamos las puntuaciones para cada feature como el máximo chi² con cualquier etiqueta
chi2_scores = np.zeros(X_tfidf.shape[1])

for i in range(Y.shape[1]):
    y = Y[:, i].toarray().flatten() if hasattr(Y[:, i], 'toarray') else Y[:, i]
    for j in range(X_tfidf.shape[1]):
        x = X_tfidf[:, j].toarray().flatten()
        contingency = pd.crosstab(x > 0, y)
        if contingency.shape == (2, 2):
            chi2, _, _, _ = chi2_contingency(contingency, correction=False)
            chi2_scores[j] = max(chi2_scores[j], chi2)

# === SELECCIÓN DE K MEJORES Y EVALUACIÓN ===
k_values = [500, 1000, 2000, 3000, 5000]
scores = []

for k in k_values:
    top_k_indices = np.argsort(chi2_scores)[-k:]
    X_selected = X_tfidf[:, top_k_indices]

    clf = LogisticRegression(max_iter=1000)
    f1_macro = cross_val_score(clf, X_selected, Y, cv=3, scoring='f1_macro').mean()

    print(f"F1_macro con {k} features seleccionadas (scipy chi2): {f1_macro:.4f}")
    scores.append((k, f1_macro))

# === GRAFICAR ===
k_vals, f1_vals = zip(*scores)
plt.figure(figsize=(8, 5))
plt.plot(k_vals, f1_vals, marker='o', color='blue')
plt.title('F1_macro vs Número de Features (Chi² - scipy)')
plt.xlabel('Número de Features seleccionadas (k)')
plt.ylabel('F1_macro')
plt.grid(True)
plt.tight_layout()
plt.savefig('./Plots/chi2_f1_macro_curve_scipy.png')
plt.show()