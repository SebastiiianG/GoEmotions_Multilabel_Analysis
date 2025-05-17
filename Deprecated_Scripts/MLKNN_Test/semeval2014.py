
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from skmultilearn.adapt import MLkNN 
from sklearn.metrics import hamming_loss, accuracy_score 

aspects_df = pd.read_csv('./Deprecated_Scripts/MLkNN_Test/semeval2014.csv') 

X = aspects_df["text"] 
y = np.asarray(aspects_df[aspects_df.columns[1:]]) 

# initializing TfidfVectorizer 
vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85) 
# fitting the tf-idf on the given data 
vetorizar.fit(X) 

# splitting the data to training and testing data set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 

# transforming the data 
X_train_tfidf = vetorizar.transform(X_train) 
X_test_tfidf = vetorizar.transform(X_test) 

# using Multi-label kNN classifier 
mlknn_classifier = MLkNN() 
mlknn_classifier.fit(X_train_tfidf, y_train) 

new_sentences = ["I like the food but I hate the place"] 
new_sentence_tfidf = vetorizar.transform(new_sentences) 

predicted_sentences = mlknn_classifier.predict(new_sentence_tfidf) 
print(predicted_sentences.toarray()) 

predicted = mlknn_classifier.predict(X_test_tfidf) 

print(accuracy_score(y_test, predicted)) 
print(hamming_loss(y_test, predicted)) 
