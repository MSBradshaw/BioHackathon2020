import re
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import pickle
import os


def date_to_unix_time(date):
    if date is None or date == '':
        return None
    dt = datetime.datetime.strptime(date, '%B %d, %Y')
    return int(time.mktime(dt.timetuple()))


def get_n_grams(_text, _n, _gram_dict={}):
    # if a special character is being used as punctuation (not in a name) add a space
    _text = re.sub('(: )', ' \\g<1>', _text)
    _text = re.sub('(- )', ' \\g<1>', _text)
    _text = re.sub('(, )', ' \\g<1>', _text)
    _text = re.sub('(\\. )', ' \\g<1>', _text)
    _text = re.sub('(- )', ' \\g<1>', _text)
    _text = re.sub('(\\? )', ' \\g<1>', _text)
    _text = re.sub('(; )', ' \\g<1>', _text)
    _text = re.sub('(! )', ' \\g<1>', _text)
    # remove paranthesis arounda  single word
    _text = re.sub(' \\(([^ ])\\) ', ' \\g<1> ', _text)
    # remove leading and trailing parenthesis
    _text = re.sub(' \\(', ' ', _text)
    _text = re.sub('\\) ', ' ', _text)
    _text_list = _text.split(' ')

    # create the n-grams
    _done = False
    # gram_dict = {}
    for _i in range(len(_text_list)):
        _gram = ''
        _skip = False
        for _j in range(_n):
            if _i + _j >= len(_text_list):
                _done = True
                break
            # check if the current item is punctuation, if so skip this gram
            if _text_list[_i + _j] in ['.', ',', '?', ';', '!', ':', '-']:
                _skip = True
                break
            _gram += _text_list[_i + _j] + ' '
        if not _done and not _skip:
            # remove trailing space
            _gram = _gram[:-1]
            # if gram has already been made
            if _gram in _gram_dict:
                # increment count
                _gram_dict[_gram] += 1
            else:
                # else create new entry
                _gram_dict[_gram] = 1
    _gram_df = pd.DataFrame({'gram': list(_gram_dict.keys()), 'count': list(_gram_dict.values())})
    return _gram_df, _gram_dict


def get_df_of_n_grams(_texts, _n):
    _dic = {}
    _final_df = None
    for _ab in _texts:
        _final_df, _dic = get_n_grams(BeautifulSoup(_ab).get_text(), _n, _dic)

    _grams = list(set(_final_df['gram']))
    _article_n_grams = {_x: [] for _x in _grams}

    for _ab in _texts:
        _final_df, _dic = get_n_grams(BeautifulSoup(_ab).get_text(), _n,{})
        for _key in _grams:
            if _key in _dic:
                _article_n_grams[_key].append(_dic[_key])
            else:
                _article_n_grams[_key].append(0)

    fake_df_n_grams = pd.DataFrame(_article_n_grams)
    return fake_df_n_grams


train = pd.read_csv('train.csv')

pickle_cache = 'grams_2_df.pickle'

if os.path.exists(pickle_cache):
    grams_2 = pickle.load(open(pickle_cache,'rb'))
else:
    grams_2 = get_df_of_n_grams(list(train['abstract']),2)
    pickle.dump(grams_2,open(pickle_cache,'wb'),protocol=4)

X = grams_2.to_numpy()
y = train['type'].to_numpy()


clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)
with open('svm-results.txt','w') as outfile:
    outfile.write('Cross Val scores: ' + str(scores) + '\n')
    outfile.write('SVM SCore: ' + str(clf.score(X_test,y_test)) + '\n')
    preds = clf.predict(X_test)
    outfile.write('Predictions: ')
    for p in preds:
        outfile.write(',' + str(p))