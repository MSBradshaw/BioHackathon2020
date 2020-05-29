from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfTransformer

data = pd.read_csv('dataset.csv')
abstracts = [BeautifulSoup(x).get_text() for x in data['abstract']]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(abstracts)
y = data['type'].to_numpy()

support_vec = svm.SVC(kernel='rbf', C=1000, gamma=0.001)
rf = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=700)
sgd = SGDClassifier(alpha=0.0001, fit_intercept=True, loss='modified_huber', penalty='l2')
pac = PassiveAggressiveClassifier(C=1.0, early_stopping=True, fit_intercept=True, max_iter=2000)

support_vec.fit(X, y)
rf.fit(X, y)
sgd.fit(X, y)
pac.fit(X, y)

# p_data = pd.read_csv('potentially_fake.tsv', sep='\t')
p_data = pd.read_csv('potentially_fake-8000.tsv', sep='\t')

p_abstracts = [BeautifulSoup(x).get_text() for x in p_data['abstract']]
fake_indexes = []
for index in range(len(p_abstracts)):
    tfidf_pred = TfidfVectorizer(vocabulary=tfidf.vocabulary_)
    p_x = tfidf_pred.fit_transform([p_abstracts[index]])
    predictions = [support_vec.predict(p_x)[0], rf.predict(p_x)[0], sgd.predict(p_x)[0], pac.predict(p_x)[0]]
    # if there is a majority saying it is fake
    if predictions.count('fake') > 3:
        fake_indexes.append(index)
        print(str(index) + ' Fake!')
    else:
        print(index)

# p_data.loc[fake_indexes].to_csv('115_predicted_fake.csv')


ids = []
for line in open('download_data/fake_pmids.txt'):
    ids.append(line[5:].strip())
    print(line[5:])

newids = [x for x in list(p_data.loc[fake_indexes]['pmid']) if str(x) not in ids]

print('Potentially Fake PMIDs:')
with open('new_potentially_fake_pmids.txt','w') as outfile:
    for i in newids:
        outfile.write(str(newids[i]) + '\n')
        print(i)