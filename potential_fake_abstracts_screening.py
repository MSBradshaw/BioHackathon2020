import wordcloud
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
"""
This script is actually for making word clouds
"""

data = pd.read_csv('dataset.csv')
abstracts = [BeautifulSoup(x).get_text() for x in data['abstract']]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(abstracts)
y = data['type'].to_numpy()

# get the terms and weights of those terms
index_value = {i[1]: i[0] for i in tfidf.vocabulary_.items()}
fully_indexed = []
for row in X:
    fully_indexed.append({index_value[column]: value for (column, value) in zip(row.indices, row.data)})


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

with open('wordcloud.txt','w') as outfile:
    for k in sd:
        string = str(k) + ' '
        outfile.write(string * int(sd[k] * 100))
        outfile.write(' ')

with open('wordcloud.txt', 'r') as file:
    words = file.read()
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                      stopwords = stopwords,
                min_font_size = 10).generate(words)
# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()