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
        print('Fake!')

p_data.loc[fake_indexes].to_csv('115_predicted_fake.csv')


ids = []
for line in open('download_data/fake_pmids.txt'):
    ids.append(line[5:].strip())
    print(line[5:])

newids = [x for x in list(p_data.loc[fake_indexes]['pmid']) if str(x) not in ids]

p_data.index = p_data['pmid']

p_data[newids]

oldids = [x for x in list(p_data.loc[fake_indexes]['pmid']) if x in ids]

#-----------
#-----------
#-----------
#-----------
#-----------
# get the terms and weights of those terms
index_value = {i[1]: i[0] for i in tfidf.vocabulary_.items()}
fully_indexed = []
for row in X:
    fully_indexed.append({index_value[column]: value for (column, value) in zip(row.indices, row.data)})

s_dic = {k: v for k, v in sorted(fully_indexed.items(), key=lambda item: item[1])}

terms = {}
for d in fully_indexed:
    for k in d:
        if k in terms:
            terms[k].append(d[k])
        else:
            terms[k] = [d[k]]


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)
term_counts = {}
for abstract in abstracts:
    tf_idf_vector = tfidf_transformer.transform(tfidf.transform([abstract]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    feature_names = tfidf.get_feature_names()
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    for k in keywords:
        if k in term_counts:
            term_counts[k] += keywords[k]
        else:
            term_counts[k] = keywords[k]

sd = {k: v for k, v in sorted(term_counts.items(), key=lambda item: item[1])}
