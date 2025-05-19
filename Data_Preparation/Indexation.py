#Archivo hecho para asignar las emociones a los datos
#Convierte los índices numéricos en etiquetas de emociones



# Importación de la librería pandas para la manipulación de datos
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# ***** CARGA DE DATOS *****
# Cargar los conjuntos de datos desde archivos TSV (valores separados por tabulaciones)

# El archivo no tiene encabezado, por lo que se definen las columnas manualmente: 
# 'Text' (texto del comentario), 'Class' (emociones asociadas) y 'ID' (identificador del comentario)

# Cada archivo representa una parte del dataset: entrenamiento, validación y prueba

train_data = pd.read_csv("./Data/GoEmotions/train.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID']) 
valid_data = pd.read_csv("./Data/GoEmotions/dev.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])
test_data = pd.read_csv("./Data/GoEmotions/test.tsv", sep='\t', header=None, names=['Text', 'Class', 'ID'])


# ***** PROCESAMIENTO DE LOS DATOS *****

# Para cada fila en la columna 'Class' (que contiene múltiples emociones separadas por comas), se divide el texto en una lista de emociones. Es decirm si se tiene 6, 4 dentro de un mismo objeto, se separan.

train_data['List of classes'] = train_data['Class'].apply(lambda x: x.split(','))
train_data['Len of classes'] = train_data['List of classes'].apply(lambda x: len(x))

valid_data['List of classes'] = valid_data['Class'].apply(lambda x: x.split(','))
valid_data['Len of classes'] = valid_data['List of classes'].apply(lambda x: len(x))

test_data['List of classes'] = test_data['Class'].apply(lambda x: x.split(','))
test_data['Len of classes'] = test_data['List of classes'].apply(lambda x: len(x))

# Ver datos procesados
print(train_data)
print(valid_data)
print(test_data)


# ***** VERIFICACIÓN DE DATOS *****
#Comprobar que no hay datos nulos
print("Conteo de datos nulos")
print(train_data.isnull().sum())
print(valid_data.isnull().sum())
print(test_data.isnull().sum())


#Contar etiquetas por df
print("\nEtiquetas por DF")
print(train_data["Class"].value_counts())
print(valid_data["Class"].value_counts())
print(test_data["Class"].value_counts())


# ***** CARGA DE EMOCIONES *****
#Abrir archivo de emociones y ponerlas en un array (posición del 0 al 27)
emotion_file = open("./Data/GoEmotions/emotions.txt", "r")
emotion_list = emotion_file.read()
emotion_list = emotion_list.split("\n")
print(emotion_list)


# ***** ASIGNACIÓN DE EMOCIONES *****
#Función que toma la lista de etiquetas numéricas por registro
#Para cada número, se le asigna la clase de acuerdo al índice del array de emociones
def idx2class(idx_list):
    arr = []
    for i in idx_list:
        arr.append(emotion_list[int(i)])
    return arr

#Aplicación de la función a los datos
train_data['Emotions'] = train_data['List of classes'].apply(idx2class)
valid_data['Emotions'] = valid_data['List of classes'].apply(idx2class)
test_data['Emotions'] = test_data['List of classes'].apply(idx2class)


print("\nTabla con emociones")
print(train_data)
print(valid_data)
print(test_data)

#Binarización de las emociones
mlb = MultiLabelBinarizer(classes=emotion_list)
mlb.fit(train_data['Emotions']) 
mlb.fit(valid_data['Emotions'])
mlb.fit(test_data['Emotions'])


# Transformar las emociones en variables binarias (0 o 1) para cada emoción
# Se crea un nuevo DataFrame con las emociones binarizadas
# Se eliminan las columnas originales de 'Emotions' y 'List of classes'
# Se concatenan las columnas originales con las nuevas columnas binarizadas
# Guardar una copia para conservar la columna 'Emotions'
train_data_with_emotions = train_data.copy()
valid_data_with_emotions = valid_data.copy()
test_data_with_emotions = test_data.copy()

# Binarización y reemplazo
train_binarized = pd.DataFrame(mlb.transform(train_data_with_emotions['Emotions']), columns=mlb.classes_)
valid_binarized = pd.DataFrame(mlb.transform(valid_data_with_emotions['Emotions']), columns=mlb.classes_)
test_binarized = pd.DataFrame(mlb.transform(test_data_with_emotions['Emotions']), columns=mlb.classes_)

train_data = pd.concat([train_data[['Text', 'ID']], train_binarized], axis=1)
valid_data = pd.concat([valid_data[['Text', 'ID']], valid_binarized], axis=1)
test_data  = pd.concat([test_data[['Text', 'ID']], test_binarized], axis=1)

# Guardar el archivo de emociones
train_data.to_csv('./Data/train_indexado.csv', index=False)
valid_data.to_csv('./Data/valid_indexado.csv', index=False)
test_data.to_csv('./Data/test_indexado.csv', index=False)

print("\nArchivos guardados")

