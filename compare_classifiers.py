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

abstracts = [BeautifulSoup(x).get_text() for x in train['abstract']]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(abstracts)
y = train['type'].to_numpy()


support_vec = svm.SVC(kernel='linear', C=1)

rf = RandomForestClassifier()
gnb = GaussianNB()
lr = LinearRegression()
knn = KNeighborsClassifier()
sgd = SGDClassifier()
pac = PassiveAggressiveClassifier()
learners = [support_vec,rf,gnb,lr,knn,sgd,pac]
learner_names = ['SupportVector','RandomForest','GradientBoosting','LinearRegression','KNN','StochasticGradientDescent','PassiveAggressive']

print('Lengths')
print(sys.argv)
print(len(sys.argv))
print(len(learners))
print(len(learner_names))
learner = learners[int(sys.argv[1])]
name = learner_names[int(sys.argv[1])]

scores = cross_val_score(learner, X, y, cv=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

learner.fit(X_train, y_train)

with open('results-'+name+'.txt','w') as outfile:
    outfile.write('Cross Val scores: ' + str(scores) + '\n')
    outfile.write('Prediction Score: ' + str(learner.score(X_test,y_test)) + '\n')
    preds = learner.predict(X_test)
    outfile.write('Predictions fake:' + str(list(preds).count('fake')) + ' real: ' + str(list(preds).count('real')))


