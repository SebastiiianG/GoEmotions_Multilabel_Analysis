import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

# CARGA DE DATOS
train_data = pd.read_csv("./Data/train_indexado.csv")

# Separar texto y etiquetas
emotion_columns = train_data.columns.difference(['Text', 'ID'])
X_texts = train_data['Text']
Y = train_data[emotion_columns].values  # Matriz binaria (n_samples, n_emotions)


# TF-IDF (Reducir a 10,000 características más frecuentes)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', strip_accents='unicode', max_features=10000)
X_tfidf = vectorizer.fit_transform(X_texts)
X_tfidf = X_tfidf.tocsc()  
feature_names = vectorizer.get_feature_names_out()


# CÁLCULO MANUAL DE CHI2 PARA CADA FEATURE Y ETIQUETA 
# Acumulamos las puntuaciones para cada feature como el máximo chi2 con cualquier etiqueta
chi2_scores_all = np.zeros(X_tfidf.shape[1])

for i in range(Y.shape[1]):
    scores, _ = chi2(X_tfidf, Y[:, i])
    chi2_scores_all = np.maximum(chi2_scores_all, scores)

chi2_scores = chi2_scores_all


# === SELECCIÓN DE K MEJORES Y EVALUACIÓN ===
k_values = [0, 500, 1000, 2000, 3000, 5000]
scores = []

for k in k_values:
    top_k_indices = np.argsort(chi2_scores)[-k:]
    X_selected = X_tfidf[:, top_k_indices]
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    f1_macro = cross_val_score(clf, X_selected, Y, cv=3, scoring='f1_macro').mean()
    print(f"F1_macro con {k} features seleccionadas (scipy chi2): {f1_macro:.4f}")
    scores.append((k, f1_macro))

# GRAFICAR
k_vals, f1_vals = zip(*scores)
plt.figure(figsize=(8, 5))
plt.plot(k_vals, f1_vals, marker='o', color='blue')
plt.title('Tabla para evaluar el valor de chi2')
plt.xlabel('Número de Features seleccionadas (k)')
plt.ylabel('F-measure Macro')
plt.grid(True)
plt.tight_layout()
plt.savefig('./Plots/chi2_f1_macro_curve_scipy.png')
plt.close()



#Se puede usar SelectKBest para seleccionar las mejores features y asignar varias etiquetas por emoción, ya que
#actualmente, solo se asigna una etiqueta por emoción.


# GENERAR Y VER DATAFRAME CON 2000 MEJORES FEATURES
best_k = 2000
top_k_indices = np.argsort(chi2_scores)[-best_k:]
X_best_k = X_tfidf[:, top_k_indices]

# Obtener los nombres de las 2000 mejores features
best_feature_names = feature_names[top_k_indices]

# Reconstruir 'pseudo-textos' solo con las 2000 palabras más importantes
X_array = X_best_k.toarray()

reconstructed_texts = []
for row in X_array:
    words = [best_feature_names[i] for i, val in enumerate(row) if val > 0]
    reconstructed_texts.append(' '.join(words))

# DataFrame con los textos reconstruidos y etiquetas originales
df_texts = pd.DataFrame({'Text': reconstructed_texts, 'ID': train_data['ID']})
Y_df = pd.DataFrame(Y, columns=emotion_columns)

# Concatenar los textos reconstruidos con las etiquetas de emoción
final_df = pd.concat([df_texts, Y_df], axis=1)

# Mostrar ejemplo
print(final_df.head())

# Guardar a CSV
final_df.to_csv("./Data/Chi2/train_2000_chi2.csv", index=False)
