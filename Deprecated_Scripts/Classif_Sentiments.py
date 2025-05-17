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
    lowercase=True,                # Convertir todo a minúsculas
    stop_words="english",          # Eliminar palabras vacías en inglés (puedes cambiarlo según el idioma)
    strip_accents="unicode",       # Normalizar acentos
    max_features=5000              # Limitar el número de características para evitar overfitting
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



#alterar parámetros, primero contar el numero total de feature, si son 20,000, dejarlo en 5000 y si no en 10,000

#Aplicar lemantización, uppercase, lowercase y con y sin stopwords

#Imprimir matrices de confusión en colores tanto para los multiclase y multilabel

#leer el link de la doctora para aplicar en varios, no limpiamos el dataset, y también hacer expermientos con el dataset limpio (para multiclase cuando caen en dos clases)


#MLNN y foest para multilabel

#Empezar el campo de experimentos en el documento del proyecto
