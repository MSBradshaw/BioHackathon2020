import re
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

df = pd.read_csv('dataset.csv')

df_copy = df

grams_2 = get_df_of_n_grams(list(df['abstract']),2)



pca = PCA(n_components=2)
pca.fit(grams_2.to_numpy())

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=list(grams_2.columns))
sorted_loadings = loadings.sort_values(['PC1', 'PC2'],ascending=False)
# top 10 most important 2-grams
print(sorted_loadings.iloc[:10,:])
sorted_loadings.iloc[:10,:].to_csv('top-10-pca-term.csv')
# bottom 10  2-grams
print(sorted_loadings.iloc[-10:,:])
sorted_loadings.iloc[-10:,:].to_csv('bottom-10-pca-term.csv')

pca2 = PCA(n_components=2)
pca2.fit(grams_2.to_numpy().transpose())

pickle.dump(pca2, open('real_fake_pca.pickle','wb'))

sns.scatterplot(pca2.components_[0,:],pca2.components_[1,:],hue=list(df['type']))
plt.savefig('real_vs_fake_pca.png')

pca = pickle.load(open('real_fake_pca.pickle','rb'))

fakes0 = list(pca.components_[0][:list(df['type']).index('real')])
fakes1 = list(pca.components_[1][:list(df['type']).index('real')])

real0 = list(pca.components_[0][list(df['type']).index('real'):])
real1 = list(pca.components_[1][list(df['type']).index('real'):])

pc_1 = real0 + fakes0
pc_2 = real1 + fakes1

labels = (['real'] * len(real0)) + (['fake'] * len(fakes0))

sns.scatterplot(pc_1,pc_2,hue=labels)
