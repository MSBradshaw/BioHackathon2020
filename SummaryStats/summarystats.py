import pandas as pd
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

os.chdir('SummaryStats')
known = pd.read_csv('known_fake_abstract_info.tsv', sep='\n')
predicted = pd.read_csv('predicted_fake_abstract_info.tsv', sep='\n')


def sep_df(lines_df):
    sections = ['pmid', 'authors', 'journal', 'year', 'title', 'abstract']
    dictionary = {x: [] for x in sections}
    for i in range(lines_df.shape[0]):
        split = lines_df.iloc[i, 0].split('\t')
        if len(split) != 6:
            continue
        for j in range(len(split)):
            if j == 2:
                dictionary[sections[j]].append(re.sub('\\..*', '', split[j]))
            else:
                dictionary[sections[j]].append(split[j])
    df = pd.DataFrame(dictionary)
    return df


kdf = sep_df(known)
pdf = sep_df(predicted)


def summary_stats(pdf, kdf, col, figname, ignore_pdf_0s, title):
    vals = list(pdf[col])
    vals2 = list(kdf[col])
    counts = Counter(vals)
    counts2 = Counter(vals2)
    df = pd.DataFrame.from_dict(counts, orient='index')
    df2 = pd.DataFrame.from_dict(counts2, orient='index')
    df[1] = df2[0]
    df.fillna(0)
    df.columns = ['Predicted', 'Known']
    df = df.sort_index()
    if ignore_pdf_0s:
        df = df[df['Predicted'] != 0]
        df = df.sort_values('Predicted')
        df.plot(kind='barh', figsize=(10, 20))
    else:
        df.plot(kind='bar', figsize=(20, 20))
    plt.title(title)
    plt.savefig(figname, bbox_inches='tight')


summary_stats(pdf, kdf, 'year', 'summary_stats_year.png', False, 'Publication Years')
summary_stats(pdf, kdf, 'journal', 'summary_stats_journal.png', True, 'Journals')

aff_df = pd.read_csv('PMIDS_Affiliations_single.tsv', sep='\n')
affs = []
for i in range(aff_df.shape[0]):
    split = str(aff_df.iloc[i, 0]).split('\t')
    if len(split) > 0:
        af = split[-1]
        af = re.sub('^a ', '', af)
        af = re.sub('^ ', '', af)
        # if len(af) > 0 and af[0] == ' ':
        #     af = af[1:]
        try:
            af = re.match('^[a-zA-Z 0-9\.\'\-&\(\)]*,[a-zA-Z 0-9\.\'\-&\(\)]*', af).group()
            hospital = re.sub('^[a-zA-Z 0-9\.\'\-&\(\)]*, ', '', af)
            affs.append(hospital.lower())
        except AttributeError:
            print('Error:' + af)
print(len(set(affs)))
aff_df = pd.DataFrame.from_dict(Counter(affs), orient='index')
aff_df = aff_df.sort_values(0, ascending=False)
og_dim = aff_df.shape
aff_df = aff_df[aff_df[0] > 1]
print('Number of stand alone affiliations:' + str(og_dim[0] - aff_df.shape[0]))
aff_df.plot(kind='bar', legend=None)
plt.title('First Author Affiliations')
plt.savefig('affiliation_plot.png', bbox_inches='tight')
