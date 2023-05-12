# # Proyecto Inteligencia Artificial
# Stefano Aragoni, Luis Diego Santos, Carol Arevalo
# 
# ______________________
# 
# El objetivo del presente proyecto es detectar los tweets que puedan estar relacionados con cyberbulling. Para ello se descargo una base de datos de tweets del 2020. 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import re
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from nltk.stem import SnowballStemmer
from copy import deepcopy
from pandas.plotting import scatter_matrix
import pandas_profiling
from prettytable import PrettyTable
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('cyberbullying_tweets_clean.csv')

data.head()

data.tail()

sns.heatmap(data.isnull(), cbar=False)

data.columns[data.isnull().any()]

data['tweet_text'] = data['tweet_text'].str.lower()

data.duplicated().sum()

data.drop_duplicates(inplace=True)
data.duplicated().sum()

frecuencias = data['cyberbullying_type'].value_counts()

print(frecuencias)

class_counts = data['cyberbullying_type'].value_counts()
class_counts.plot(kind='bar')
plt.title('Class Distribution of Cyberbullying Types')
plt.xlabel('Labels')
plt.ylabel('Number of Tweets')
plt.show()

new_tweets = []
new_types = []

def remove_urls_mentions_hashtags(text, type):
    text = re.sub(r'http\S+', '', text) # URLs
    text = re.sub(r'@\S+', '', text) # Menciones
    text = re.sub(r'#\S+', '', text) # Hashtags

    text = re.sub('<.*?>', '', text) # HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text) # Puntuación y números

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text, type

for i in tqdm(range(len(data))):
    try:
        text, type = remove_urls_mentions_hashtags(data['tweet_text'][i], data['cyberbullying_type'][i])
        new_tweets.append(text)
        new_types.append(type)
    except:
        pass

data = pd.DataFrame({
    'tweet_text': new_tweets,
    'cyberbullying_type': new_types
})

tweets = ' '.join(data['tweet_text'].values)
words = tweets.split()

frecuencia = Counter(words).most_common(10)

frecuencia

plt.bar(*zip(*frecuencia))
plt.title('Top palabras más frecuentes')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.show()

modelo1 = data.__deepcopy__()
modelo2 = data.__deepcopy__()

not_cyberbullying = modelo1[modelo1['cyberbullying_type'] == 'not_cyberbullying']
age = modelo1[modelo1['cyberbullying_type'] == 'age']
gender = modelo1[modelo1['cyberbullying_type'] == 'gender']
religion = modelo1[modelo1['cyberbullying_type'] == 'religion']
ethnicity = modelo1[modelo1['cyberbullying_type'] == 'ethnicity']

sample_size = int(len(not_cyberbullying)/4)
sampled_age = age.sample(n=sample_size, replace=True, random_state=42)
sampled_gender = gender.sample(n=sample_size, replace=True, random_state=42)
sampled_religion = religion.sample(n=sample_size, replace=True, random_state=42)
sampled_ethnicity = ethnicity.sample(n=sample_size, replace=True, random_state=42)

modelo1 = pd.concat([not_cyberbullying, sampled_gender, sampled_age, sampled_ethnicity, sampled_religion])

modelo1['cyberbullying_type'] = modelo1['cyberbullying_type'].replace(['age', 'gender', 'religion', 'ethnicity'], 'cyberbullying')

frecuencias = modelo1['cyberbullying_type'].value_counts()

print("\nModelo 1: Cyberbullying vs. Not Cyberbullying\n")
print(frecuencias)

age = modelo2[modelo2['cyberbullying_type'] == 'age']
gender = modelo2[modelo2['cyberbullying_type'] == 'gender']
religion = modelo2[modelo2['cyberbullying_type'] == 'religion']
ethnicity = modelo2[modelo2['cyberbullying_type'] == 'ethnicity']

modelo2 = pd.concat([gender, age, ethnicity, religion])

frecuencias = modelo2['cyberbullying_type'].value_counts()

print("\nModelo 2: Cyberbullying Type\n")
print(frecuencias)

vectorizer1 = CountVectorizer()
X1 = vectorizer1.fit_transform(modelo1['tweet_text']).toarray()
y1 = modelo1['cyberbullying_type']

vectorizer2 = CountVectorizer()
X2 = vectorizer2.fit_transform(modelo2['tweet_text']).toarray()
y2 = modelo2['cyberbullying_type']


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_val1, X_test1, y_val1, y_test1 = train_test_split(X_test1, y_test1, test_size=0.5, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_test2, y_test2, test_size=0.5, random_state=42)


best_models1 = []
best_models2 = []

param_grid = {
    'C': [0.0001, 0.001, 0.1, 1, 10],
    'max_iter': [1000, 10000],
}

lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid, cv=5)
clf.fit(X_val1, y_val1)

best_lr1 = LogisticRegression(**clf.best_params_)
best_lr1.fit(X_train1, y_train1)

accuracy = accuracy_score(y_test1, best_lr1.predict(X_test1))
best_models1.append(('Logistic Regression M1', clf.best_params_, accuracy))

print("Regresion Logistica (Bullying vs. Not Bullying): ", accuracy, "\n")
print("Best parameters: ", clf.best_params_, "\n")
print(classification_report(y_test1, best_lr1.predict(X_test1)))

lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid, cv=5)
clf.fit(X_val2, y_val2)

best_lr2 = LogisticRegression(**clf.best_params_)
best_lr2.fit(X_train2, y_train2)

accuracy = accuracy_score(y_test2, best_lr2.predict(X_test2))
best_models2.append(('Logistic Regression M2', clf.best_params_, accuracy))

print("Regresion Logistica (Type of Bullying) : ", accuracy, "\n")
print("Best parameters: ", clf.best_params_, "\n")
print(classification_report(y_test2, best_lr2.predict(X_test2)))

# save the model to disk
filename = 'best_lr1.sav'
pickle.dump(best_lr1, open(filename, 'wb'))
filename = 'best_lr2.sav'
pickle.dump(best_lr2, open(filename, 'wb'))

# save the vectorizer to disk
filename = 'vectorizer1.sav'
pickle.dump(vectorizer1, open(filename, 'wb'))
filename = 'vectorizer2.sav'
pickle.dump(vectorizer2, open(filename, 'wb'))