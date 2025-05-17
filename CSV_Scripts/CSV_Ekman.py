import pandas as pd
import ast
import json

# Carga de datos
train_data = pd.read_csv("./Data/train_indexado.csv")
valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

# Convertir las emociones de string a lista real
train_data['Emotions'] = train_data['Emotions'].apply(ast.literal_eval)
valid_data['Emotions'] = valid_data['Emotions'].apply(ast.literal_eval)
test_data['Emotions'] = test_data['Emotions'].apply(ast.literal_eval)

# Cargar el mapeo de Ekman desde el archivo JSON
with open("./Data/GoEmotions/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

# Asegurar inclusión de 'neutral'
ekman_mapping["neutral"] = ["neutral"]

# Crear mapeo inverso: emoción específica -> categoría Ekman
reverse_ekman_map = {e: cat for cat, emos in ekman_mapping.items() for e in emos}

# Función para obtener índices conflictivos
def get_conflicting_indices(df, reverse_ekman_map):
    conflicting = []
    for idx, emotions in enumerate(df['Emotions']):
        categorias = set()
        for e in emotions:
            if e in reverse_ekman_map:
                categorias.add(reverse_ekman_map[e])
        if len(categorias) > 1:
            conflicting.append(idx)
    return conflicting

# Obtener índices conflictivos
conflict_train_idx = get_conflicting_indices(train_data, reverse_ekman_map)
conflict_valid_idx = get_conflicting_indices(valid_data, reverse_ekman_map)
conflict_test_idx  = get_conflicting_indices(test_data,  reverse_ekman_map)

# Conteo para documentación
solo_una_train = len(train_data) - len(conflict_train_idx)
solo_una_valid = len(valid_data) - len(conflict_valid_idx)
solo_una_test  = len(test_data) - len(conflict_test_idx)

print("\nTrain:")
print("Solo una categoría:", solo_una_train)
print("Más de una categoría:", len(conflict_train_idx))

print("\nValidation:")
print("Solo una categoría:", solo_una_valid)
print("Más de una categoría:", len(conflict_valid_idx))

print("\nTest:")
print("Solo una categoría:", solo_una_test)
print("Más de una categoría:", len(conflict_test_idx))

# Eliminar registros conflictivos
train_data_clean = train_data.drop(index=conflict_train_idx).reset_index(drop=True)
valid_data_clean = valid_data.drop(index=conflict_valid_idx).reset_index(drop=True)
test_data_clean  = test_data.drop(index=conflict_test_idx).reset_index(drop=True)

# Asignar un número a cada emoción Ekman
ekman_labels = list(ekman_mapping.keys())
ekman_to_id = {label: idx for idx, label in enumerate(ekman_labels)}

# Categorización usando Ekman
def categorize_emotions(row):
    for emotion in row['Emotions']:
        if emotion in reverse_ekman_map:
            ekman_category = reverse_ekman_map[emotion]
            return ekman_to_id[ekman_category]
    return len(ekman_labels)  # "otros" si no encuentra

# Aplicar categorización a los datos limpios
train_data_clean['Emotion'] = train_data_clean.apply(categorize_emotions, axis=1)
valid_data_clean['Emotion'] = valid_data_clean.apply(categorize_emotions, axis=1)
test_data_clean['Emotion']  = test_data_clean.apply(categorize_emotions, axis=1)

# Guardar resultados
train_data_clean[['Text', 'Emotion', 'ID']].to_csv('./Data/BasedOnEkman/train_ekman.csv', index=False)
valid_data_clean[['Text', 'Emotion', 'ID']].to_csv('./Data/BasedOnEkman/valid_ekman.csv', index=False)
test_data_clean[['Text', 'Emotion', 'ID']].to_csv('./Data/BasedOnEkman/test_ekman.csv', index=False)

print("\nArchivos guardados correctamente.")