import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

# CARGA DE DATOS
train_data = pd.read_csv("./Data/Chi2/train_2000_chi2.csv")
valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")

# CARGA DEL MAPEADO EKMAN
with open("./Data/GoEmotions/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

# Asegurar inclusión de 'neutral'
ekman_mapping["neutral"] = ["neutral"]
reverse_ekman_map = {e: cat for cat, emos in ekman_mapping.items() for e in emos}
ekman_labels = list(ekman_mapping.keys())

# IDENTIFICAR EMOCIONES ORIGINALES (GoEmotions)
emotion_columns = [col for col in train_data.columns if col not in ["Text", "ID"]]
# FUNCIÓN PARA OBTENER EMOCIONES EKMAN POR FILA
def get_ekman_emotions(row):
    active_ekman = set()
    for emo in emotion_columns:
        if row[emo] == 1 and emo in reverse_ekman_map:
            active_ekman.add(reverse_ekman_map[emo])
    return list(active_ekman)

# APLICAR FUNCIÓN 
train_data["Ekman"] = train_data.apply(get_ekman_emotions, axis=1)
valid_data["Ekman"] = valid_data.apply(get_ekman_emotions, axis=1)
test_data["Ekman"]  = test_data.apply(get_ekman_emotions, axis=1)

# FUNCIÓN PARA OBTENER ÍNDICES CONFLICTIVOS
def get_conflicting_indices(df):
    return [i for i, x in enumerate(df["Ekman"]) if len(x) > 1]

# OBTENER Y CONTAR REGISTROS CONFLICTIVOS
conflict_train_idx = get_conflicting_indices(train_data)
conflict_valid_idx = get_conflicting_indices(valid_data)
conflict_test_idx  = get_conflicting_indices(test_data)

solo_una_train = len(train_data) - len(conflict_train_idx)
#solo_una_valid = len(valid_data) - len(conflict_valid_idx)
solo_una_test  = len(test_data) - len(conflict_test_idx)

print("\nTrain:")
print("Solo una categoría:", solo_una_train)
print("Más de una categoría:", len(conflict_train_idx))

"""
print("\nValidation:")
print("Solo una categoría:", solo_una_valid)
print("Más de una categoría:", len(conflict_valid_idx))
"""
print("\nTest:")
print("Solo una categoría:", solo_una_test)
print("Más de una categoría:", len(conflict_test_idx))

# ELIMINAR REGISTROS CON MÁS DE UNA CATEGORÍA EKMAN
train_data_clean = train_data.drop(index=conflict_train_idx).reset_index(drop=True)
#valid_data_clean = valid_data.drop(index=conflict_valid_idx).reset_index(drop=True)
test_data_clean  = test_data.drop(index=conflict_test_idx).reset_index(drop=True)

# BINARIZACIÓN MULTILABEL (EKMAN)
mlb = MultiLabelBinarizer(classes=ekman_labels)
train_ekman = pd.DataFrame(mlb.fit_transform(train_data_clean["Ekman"]), columns=mlb.classes_)
#valid_ekman = pd.DataFrame(mlb.transform(valid_data_clean["Ekman"]), columns=mlb.classes_)
test_ekman  = pd.DataFrame(mlb.transform(test_data_clean["Ekman"]),  columns=mlb.classes_)

# COMBINAR Y GUARDAR 
train_final = pd.concat([train_data_clean[["Text", "ID"]], train_ekman], axis=1)
#valid_final = pd.concat([valid_data_clean[["Text", "ID"]], valid_ekman], axis=1)
test_final  = pd.concat([test_data_clean[["Text", "ID"]], test_ekman], axis=1)

train_final.to_csv("./Data/BasedOnEkman/train_ekman_bin.csv", index=False)
#valid_final.to_csv("./Data/BasedOnEkman/valid_ekman_bin.csv", index=False)
test_final.to_csv("./Data/BasedOnEkman/test_ekman_bin.csv", index=False)

print("\nArchivos binarizados y guardados correctamente con emociones de Ekman.")