import pandas as pd

# Carga de datos
train_data = pd.read_csv("./Data/train_indexado.csv")
valid_data = pd.read_csv("./Data/valid_indexado.csv")
test_data = pd.read_csv("./Data/test_indexado.csv")



train_data[["Text", "Emotions"]].to_csv("./Data/OnlyEmotions/train_emotions.csv", index=False)
valid_data[["Text", "Emotions"]].to_csv("./Data/OnlyEmotions/valid_emotions.csv", index=False)
test_data[["Text", "Emotions"]].to_csv("./Data/OnlyEmotions/test_emotions.csv", index=False)

