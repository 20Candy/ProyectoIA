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

from pandas.plotting import scatter_matrix
import pandas_profiling

from prettytable import PrettyTable

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

data = pd.read_csv('cyberbullying_tweets.csv')
data2 = pd.read_csv('FinalBalancedDataset.csv')

# to add 31292
contador = 0

# iterate over the rows of the data
for index, element in data2.iterrows():

    tweet = element['tweet']
    class_ = element['Toxicity']

    if class_ == 0:
        class_ = 'not_cyberbullying'

        data = data.append({'tweet_text': tweet, 'cyberbullying_type': class_}, ignore_index=True)

        contador += 1

    if contador == 27000:
        break

# save the new data
data.to_csv('cyberbullying_tweets.csv', index=False)
