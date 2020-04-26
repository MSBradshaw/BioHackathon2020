import re
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import time
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import datashader
import colorcet
import holoviews
import umap.plot
import numpy as np
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


df = pd.read_csv('422-abstracts.tsv', sep='\t')
fake_df_2_grams = get_df_of_n_grams(list(df['abstract']), 2)

grams = list(fake_df_2_grams.columns)


pulse_sub = pd.read_csv('biorvix-post-2018.tsv', sep='\t')
# choose 4 n grams at random
random.shuffle(grams)
index = 0
count = 0
selected_from_pusle = []
while count < 400:
    # get all articles from biorxiv that mention this ngram
    try:
        sub = pulse_sub.loc[pulse_sub['abstract'].str.contains(grams[index],na=False),]
    except re.error:
        # this occurs if there are in balanced parenthesis in the abstract
        index += 1
        continue
    index += 1
    if sub.shape[0] > 0:
        # choose a random index
        rand_index = random.randint(0,sub.shape[0]-1)
        sel = sub.iloc[rand_index,-2]
        selected_from_pusle.append(sel)
        count += 1

combind_abstracts = list(df['abstract']) + selected_from_pusle

combine_2_grams = get_df_of_n_grams(combind_abstracts,2)

labels = (['fake'] * len(list(df['abstract']))) + (['real'] * len(selected_from_pusle))

pca = PCA(n_components=2)
pca.fit(combine_2_grams.to_numpy())

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=list(combine_2_grams.columns))
sorted_loadings = loadings.sort_values(['PC1', 'PC2'],ascending=False)
# top 10 most important 2-grams
sorted_loadings.iloc[:10,:]
# bottom 10  2-grams
sorted_loadings.iloc[-10:,:]



colors = (['fake'] * len(list(df['abstract']))) + (['real'] * len(selected_from_pusle))
sns.scatterplot(pca.components_[0,:],pca.components_[1,:],hue=colors)


um = umap.UMAP()
a = um.fit(combine_2_grams.to_numpy())

umap.plot.points(a, labels=np.array(labels))

# remove columns that are only found in the real or only found in the fake
good_cols = []
bad_cols = []
count = 0
abs_dif = []
for col in list(combine_2_grams.columns):
    if count % 500 == 0:
        print(str(count) + ' / ' + str(combine_2_grams.shape[1]))
    count += 1
    if col == 'label':
        continue
    if sum(combine_2_grams[col]) == 1:
        continue
    data = list(combine_2_grams[col])
    fake_sub = data[:labels.count('fake')]
    real_sub = data[labels.count('fake'):]
    fake_sum = sum(fake_sub)
    real_sum = sum(real_sub)
    abs_dif.append(abs(fake_sum - real_sum))
    if fake_sum * 10 < real_sum or real_sum * 10 < fake_sum:
        bad_cols.append(col)
    else:
        good_cols.append(col)

col_sums = [sum(list(combine_2_grams.iloc[:,x])) for x in range(combine_2_grams.shape[1]-1)]
row_sums = [sum(list(combine_2_grams.iloc[x,:-2])) for x in range(combine_2_grams.shape[0])]
