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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import sys
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')

abstracts = [BeautifulSoup(x).get_text() for x in train['abstract']]
test_abstracts = [BeautifulSoup(x).get_text() for x in test['abstract']]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(abstracts)
y = train['type'].to_numpy()
support_vec = svm.SVC(kernel='rbf', C=1000, gamma=0.001)
rf = RandomForestClassifier(criterion='gini', max_features='sqrt',n_estimators=700)
sgd = SGDClassifier(alpha=0.0001, fit_intercept=True, loss='modified_huber', penalty='l2')
pac = PassiveAggressiveClassifier(C=1.0, early_stopping=True, fit_intercept=True, max_iter=2000)

learners = [support_vec, rf, sgd, pac]
learner_names = ['SupportVector', 'RandomForest', 'StochasticGradientDescent', 'PassiveAggressive']

print('Lengths')
print(sys.argv)
print(len(sys.argv))
print(len(learners))
print(len(learner_names))

learner = learners[int(sys.argv[1])]
learner_final = learners[int(sys.argv[1])]
name = learner_names[int(sys.argv[1])]

scores = cross_val_score(learner, X, y, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

learner.fit(X_train, y_train)

with open('results-optimized-' + name + '.txt', 'w') as outfile:
    outfile.write('Cross Val scores: ' + str(scores) + '\n')
    outfile.write('Prediction Score: ' + str(learner.score(X_test, y_test)) + '\n')
    preds = learner.predict(X_test)
    outfile.write(
        'Predictions fake:' + str(list(preds).count('fake')) + ' real: ' + str(list(preds).count('real')) + '\n')

# train a learner with all the data
learner_final.fit(X, y)
tfidf_final = TfidfVectorizer(vocabulary=tfidf.vocabulary_)

# make a final prediction on the held out test data
test_X = tfidf_final.fit_transform(test_abstracts)
test_y = test['type'].to_numpy()

score = learner_final.score(test_X, test_y)
with open('results-optimized-' + name + '.txt', 'a') as outfile:
    outfile.write('Test set score:' + str(score) + '\n')
