import requests
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

def cosine_similarity(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

Sentence_1 = input("Enter your the first Sentence: ")
Sentence_2 = input("Enter your second Sentence: ")

Sentence_1 =  Sentence_1.lower()
Sentence_2 = Sentence_2.lower()
Sentence_1 = word_tokenize(Sentence_1)
stop_w = stopwords.words('english')
stop_w.extend('.') 
Sentence_1 = [word for word in Sentence_1 if word not in stop_w]
ps = PorterStemmer()
Sentence_1 = [ps.stem(w) for w in Sentence_1]
Sentence_2 = word_tokenize(Sentence_2)
Sentence_2 = [word for word in Sentence_2 if word not in stop_w]
Sentence_2 = [ps.stem(w) for w in Sentence_2]

vectorizer = CountVectorizer()

docs = [
    ' '.join(Sentence_1),
    ' '.join(Sentence_2)
]

x = vectorizer.fit_transform(docs)
df = pd.DataFrame(x.toarray(), index = ['Sentence_1', 'Sentence_2'], columns = vectorizer.get_feature_names_out())
array = df.values 
Sentence_1 = array[0]
Sentence_2 = array[1]
q1 = cosine_similarity(Sentence_1,Sentence_1)
q2 = cosine_similarity(Sentence_1,Sentence_2)
q3 = cosine_similarity(Sentence_2,Sentence_2) 
output = [[q1,q2],[q2,q3]]
output = pd.DataFrame(output, index = ['Sentence_1','Sentence_2'], columns = ['Sentence_1','Sentence_2'])
print(output)


  