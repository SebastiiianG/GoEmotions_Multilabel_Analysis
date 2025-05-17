import pandas as pd
import ast

# Carga de datos
train_data = pd.read_csv("./Data/train_indexado.csv")
valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

# Convertir la cadena de lista a lista real
train_data['Emotions'] = train_data['Emotions'].apply(ast.literal_eval)
valid_data['Emotions'] = valid_data['Emotions'].apply(ast.literal_eval)
test_data['Emotions'] = test_data['Emotions'].apply(ast.literal_eval)

# Definición de grupos emocionales
positive = ['admiration', 'amusement', 'approval', 'caring', 'desire',
            'excitement', 'gratitude', 'joy', 'love', 'optimism', 
            'pride', 'relief']
negative = ['anger', 'annoyance', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'fear', 'grief', 'nervousness',
            'remorse', 'sadness']
ambiguous = ['confusion', 'curiosity', 'realization', 'surprise']
neutral = ['neutral']

# Mapeo emoción → categoría
sentiment_map = {}
for e in positive:
    sentiment_map[e] = 'positive'
for e in negative:
    sentiment_map[e] = 'negative'
for e in ambiguous:
    sentiment_map[e] = 'ambiguous'
for e in neutral:
    sentiment_map[e] = 'neutral'

# Función para obtener índices de registros con múltiples categorías
def get_conflicting_indices(df):
    conflicting = []
    for idx, emotions in enumerate(df['Emotions']):
        categorias = set()
        for e in emotions:
            if e in sentiment_map:
                categorias.add(sentiment_map[e])
        if len(categorias) > 1:
            conflicting.append(idx)
    return conflicting

# Función para contar registros por tipo
def count_sentiment_categories(df):
    solo_una, multiples = 0, 0
    for emotions in df['Emotions']:
        categorias = set()
        for e in emotions:
            if e in sentiment_map:
                categorias.add(sentiment_map[e])
        if len(categorias) <= 1:
            solo_una += 1
        else:
            multiples += 1
    return solo_una, multiples

# Obtener y contar registros conflictivos
conflict_train_idx = get_conflicting_indices(train_data)
conflict_valid_idx = get_conflicting_indices(valid_data)
conflict_test_idx  = get_conflicting_indices(test_data)

solo_una_train, multiples_train = count_sentiment_categories(train_data)
solo_una_valid, multiples_valid = count_sentiment_categories(valid_data)
solo_una_test, multiples_test = count_sentiment_categories(test_data)

# Imprimir resultados para documentación
print("\n>>> CONTEO DE REGISTROS POR CATEGORÍAS DE SENTIMIENTO:")
print("Train - Solo una categoría:", solo_una_train, "| Más de una:", multiples_train)
print("Valid - Solo una categoría:", solo_una_valid, "| Más de una:", multiples_valid)
print("Test  - Solo una categoría:", solo_una_test,  "| Más de una:", multiples_test)

# Eliminar registros conflictivos
train_data_clean = train_data.drop(index=conflict_train_idx).reset_index(drop=True)
valid_data_clean = valid_data.drop(index=conflict_valid_idx).reset_index(drop=True)
test_data_clean  = test_data.drop(index=conflict_test_idx).reset_index(drop=True)

# Función para categorizar una lista de emociones (ahora ya limpia)
def categorize_emotions(row):
    emotions = row['Emotions']
    for emotion in emotions:
        if emotion in positive:
            return 1
        elif emotion in negative:
            return 0
        elif emotion in ambiguous:
            return 2
        elif emotion in neutral:
            return 3
    return 4  # Unknown

# Aplicar categorización final a los registros limpios
train_data_clean['Sentiment'] = train_data_clean.apply(categorize_emotions, axis=1)
valid_data_clean['Sentiment'] = valid_data_clean.apply(categorize_emotions, axis=1)
test_data_clean['Sentiment']  = test_data_clean.apply(categorize_emotions, axis=1)

# Mostrar una vista previa
print("\nTabla limpia con sentimientos:")
print(train_data_clean[['Text', 'Sentiment']].head())

# Guardar archivos limpios
train_data_clean[["Text", "Sentiment", "ID"]].to_csv('./Data/BasedOnSentiments/train_sentiments.csv', index=False)
valid_data_clean[["Text", "Sentiment", "ID"]].to_csv('./Data/BasedOnSentiments/valid_sentiments.csv', index=False)
test_data_clean[["Text", "Sentiment", "ID"]].to_csv('./Data/BasedOnSentiments/test_sentiments.csv', index=False)

print("\nArchivos guardados correctamente.")