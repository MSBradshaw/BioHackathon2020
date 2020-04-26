from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.decomposition import PCA
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

train = pd.read_csv('train.csv')

abstracts = [BeautifulSoup(x).get_text() for x in train['abstract']]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(abstracts)
y = train['type'].to_numpy()


clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)
with open('svm-tfidf-results.txt','w') as outfile:
    outfile.write('Cross Val scores: ' + str(scores) + '\n')
    outfile.write('SVM SCore: ' + str(clf.score(X_test,y_test)) + '\n')
    preds = clf.predict(X_test)
    outfile.write('Predictions: ')
    for p in preds:
        outfile.write(',' + str(p))
