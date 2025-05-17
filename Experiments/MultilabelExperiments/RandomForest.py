import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix

# CARGA DE DATOS 
train_data = pd.read_csv("./Data/OnlyEmotions/train_emotions.csv")
test_data = pd.read_csv("./Data/OnlyEmotions/test_emotions.csv")

# Convertir strings de listas a listas reales
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
