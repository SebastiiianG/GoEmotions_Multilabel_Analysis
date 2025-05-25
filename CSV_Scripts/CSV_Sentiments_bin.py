import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Carga de datos
train_data = pd.read_csv("./Data/Chi2/train_2000_chi2.csv")
#valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

# Definición de grupos emocionales
positive = ['admiration', 'amusement', 'approval', 'caring', 'desire',
            'excitement', 'gratitude', 'joy', 'love', 'optimism', 
            'pride', 'relief']
negative = ['anger', 'annoyance', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'fear', 'grief', 'nervousness',
            'remorse', 'sadness']
ambiguous = ['confusion', 'curiosity', 'realization', 'surprise']
neutral = ['neutral']

sentiment_map = {e: "positive" for e in positive}
sentiment_map.update({e: "negative" for e in negative})
sentiment_map.update({e: "ambiguous" for e in ambiguous})
sentiment_map.update({e: "neutral" for e in neutral})
sentiment_labels = ["positive", "negative", "ambiguous", "neutral"]

# Determinar categorías de emociones
def get_sentiment_emotions(row):
    active = set()
    for emotion in row.index:
        if row[emotion] == 1 and emotion in sentiment_map:
            active.add(sentiment_map[emotion])
    return list(active)

# Aplicar la función a los datos
train_data["Sentiments"] = train_data.apply(get_sentiment_emotions, axis=1)
#valid_data["Sentiments"] = valid_data.apply(get_sentiment_emotions, axis=1)
test_data["Sentiments"]  = test_data.apply(get_sentiment_emotions, axis=1)


# Función para obtener índices conflictivos
def get_conflicting_indices(df):
    return [i for i, s in enumerate(df["Sentiments"]) if len(s) > 1]

# Aplciar la función para obtener índices conflictivos
conflict_train_idx = get_conflicting_indices(train_data)
#conflict_valid_idx = get_conflicting_indices(valid_data)
conflict_test_idx  = get_conflicting_indices(test_data)


solo_una_train = len(train_data) - len(conflict_train_idx)
#solo_una_valid = len(valid_data) - len(conflict_valid_idx)
solo_una_test  = len(test_data) - len(conflict_test_idx)


print("\nCONTEO DE REGISTROS POR CATEGORÍAS DE SENTIMIENTO:")
print("Train - Solo una categoría:", solo_una_train, "| Más de una:", len(conflict_train_idx))
#print("Valid - Solo una categoría:", solo_una_valid, "| Más de una:", len(conflict_valid_idx))
print("Test  - Solo una categoría:", solo_una_test,  "| Más de una:", len(conflict_test_idx))

# Eliminar registros conflictivos
train_clean = train_data.drop(index=conflict_train_idx).reset_index(drop=True)
#valid_clean = valid_data.drop(index=conflict_valid_idx).reset_index(drop=True)
test_clean  = test_data.drop(index=conflict_test_idx).reset_index(drop=True)

#Binarización de las emociones
mlb = MultiLabelBinarizer(classes=sentiment_labels)
train_sentiments = pd.DataFrame(mlb.fit_transform(train_clean["Sentiments"]), columns=mlb.classes_)
#valid_sentiments = pd.DataFrame(mlb.transform(valid_clean["Sentiments"]), columns=mlb.classes_)
test_sentiments  = pd.DataFrame(mlb.transform(test_clean["Sentiments"]),  columns=mlb.classes_)


# Combinar y guardar
train_final = pd.concat([train_clean[["Text", "ID"]], train_sentiments], axis=1)
#valid_final = pd.concat([valid_clean[["Text", "ID"]], valid_sentiments], axis=1)
test_final  = pd.concat([test_clean[["Text", "ID"]], test_sentiments], axis=1)


# Guardar los archivos finales
train_final.to_csv("./Data/BasedOnSentiments/train_sentiments_bin.csv", index=False)
#valid_final.to_csv("./Data/BasedOnSentiments/valid_sentiments_bin.csv", index=False)
test_final.to_csv("./Data/BasedOnSentiments/test_sentiments_bin.csv", index=False)