#El objetivo de este script es generar dos CSV, uno para entrenamiento y otro para testeo.
#El CSV de entrenamiento contendrá solo los registros que tienen una sola categoría de emoción pero del conjunto train
#El CSV de testeo contendrá los registros que tienen más de una categoría de emoción en el conjunto train y test
import pandas as pd

# Carga de datos
train_data = pd.read_csv("./Data/Chi2/train_2000_chi2.csv")
valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

with open("./Data/GoEmotions/emotions.txt", "r") as f:
    emotion_list = f.read().splitlines()

emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_list)}

# Función para extraer emociones activas de cada fila
def get_sentiment_emotions(row):
    return [emotion for emotion in row.index if row[emotion] == 1 and emotion in emotion_list]

# Aplicar la función a los datos
train_data["Emotions"] = train_data.apply(get_sentiment_emotions, axis=1)
valid_data["Emotions"] = valid_data.apply(get_sentiment_emotions, axis=1)
test_data["Emotions"]  = test_data.apply(get_sentiment_emotions, axis=1)

# Función para obtener índices conflictivos
def get_conflicting_indices(df):
    return [i for i, s in enumerate(df["Emotions"]) if len(s) > 1]
# Aplicar la función para obtener índices conflictivos

conflict_train_idx = get_conflicting_indices(train_data)
conflict_valid_idx = get_conflicting_indices(valid_data)
conflict_test_idx  = get_conflicting_indices(test_data)


#Append de los indices conflictivos para crear el CSV de testeo
conflict_idx = conflict_train_idx + conflict_valid_idx + conflict_test_idx
#El CSV de testeo contendrá los registros que tienen más de una categoría de emoción en el conjunto train y test

train_clean = train_data.drop(index=conflict_train_idx).reset_index(drop=True)

# Construir el conjunto de test a partir de los registros conflictivos
test_conflicts = pd.concat([train_data.loc[conflict_train_idx], test_data.loc[conflict_test_idx]]).reset_index(drop=True)


# Obtener siempre la primera emoción y su ID correspondiente (Del documento Emotins.txt)
def extract_first_sentiment_id(row):
    if row["Emotions"]:
        return emotion_to_id[row["Emotions"][0]]
    return None

train_clean["Emotion"] = train_clean.apply(extract_first_sentiment_id, axis=1)
test_conflicts["Emotion"] = test_conflicts.apply(extract_first_sentiment_id, axis=1)

# Guardar ambos archivos
train_clean[["Text", "Emotion", "ID"]].to_csv("./Data/OnlyOneEmotion/train_emotions.csv", index=False)
test_conflicts[["Text", "Emotion", "ID"]].to_csv("./Data/OnlyOneEmotion/test_emotions.csv", index=False)

print("Archivos guardados correctamente.")






