# Contador de features basado solo en el texto
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar solo los datos de texto
train_data = pd.read_csv("./Data/GoEmotions/train.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])

# Extraer los comentarios
sentences_train = train_data['Text'].values

# Vectorizador temporal para contar las palabras únicas después de procesamiento (sin ecxluir stopwords)
temp_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents= None
)

# Ajustar el vectorizador
temp_vectorizer.fit(sentences_train)

# Obtener y contar los features
feature_names = temp_vectorizer.get_feature_names_out()
n_features = len(feature_names)

# Mostrar resultados
print(f"\nNúmero total de features detectados (sin max_features): {n_features}")

# Vectorizador temporal para contar las palabras únicas después de procesamiento (sin ecxluir stopwords)
temp_vectorizer_with_stopwords = TfidfVectorizer(
    lowercase=True,
    stop_words= "english",
    strip_accents= None
)

temp_vectorizer_with_stopwords.fit(sentences_train)
# Obtener y contar los features
second_feature_names = temp_vectorizer_with_stopwords.get_feature_names_out()
n_features_2 = len(second_feature_names)
# Mostrar resultados
print(f"\nNúmero total de features detectados sin stop words(sin max_features): {n_features_2}")



"""
Resultados con stop words y strip accents = 26369
Resultados sin stop words y strip accents = 26070


Resultados con stop words y sin strip accents = 26379
Resultados sin stop words y sin strip accents = 26080



Por lo tanto se queda en 5,000
"""

