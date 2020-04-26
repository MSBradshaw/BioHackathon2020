import re
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import time
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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



pulse_sub = pd.read_csv('biorvix-post-2018.tsv', sep='\t')


# choose 200 from each subcategory:
abstracts = []
subcats = []
for cat in list(set(pulse_sub['subcategory'])):
    sub = pulse_sub[pulse_sub['subcategory'] == cat]
    for i in range(50):
        rand_index = random.randint(0, sub.shape[0] - 1)
        sel = sub.iloc[rand_index, -2]
        abstracts.append(sel)
        subcats.append(cat)

grams2 = get_df_of_n_grams(abstracts,2)

pca = PCA()
pca.fit(grams2.to_numpy().transpose())

sns.scatterplot(pca.components_[0,:],pca.components_[1,:],hue=subcats)
plt.savefig('pca-50.png', figsize=(8, 6))

sns.scatterplot(pca.components_[0,:],pca.components_[1,:],hue=subcats)
plt.legend(None)
plt.savefig('pca-50-no-legend.png', figsize=(8, 6))


