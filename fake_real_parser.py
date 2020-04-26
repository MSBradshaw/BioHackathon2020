import pandas as pd
import re
import os

def get_field_text(text, field):
    pattern = '<FIELD>[^<]*</FIELD>'.replace('FIELD', field)
    res = re.findall(pattern, text)
    if len(res) == 0:
        return ''
    final = res[0].replace('<FIELD>'.replace('FIELD', field), '')
    final = final.replace('</FIELD>'.replace('FIELD', field), '')
    return final


def get_title(text):
    pattern = '<ArticleTitle>.*</ArticleTitle>'
    res = re.findall(pattern, text)
    if len(res) == 0:
        print('NO TITLE')
        return ''
    final = res[0].replace('<ArticleTitle>', '')
    final = final.replace('</ArticleTitle>', '')
    return final


def get_abstract(text):
    pattern = '<AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">((?!AbstractText).)*</AbstractText>'
    res = re.findall(pattern, text)
    if len(res) != 0:
        string = ''
        for s in res:
            string += s
        return string
    else:
        pattern = '<AbstractText[^<]*>.*</AbstractText>'
        res = re.findall(pattern, text)
        if len(res) == 0:
            print('NO ABSTRACT')
            return ''
        final = re.sub('<AbstractText[^<]*>','',res[0])
        final = final.replace('</AbstractText>', '')
        return final
    return('')


def get_authors(text):
    lastnames = re.findall('<LastName>[^<]*</LastName>', text)
    firstnames = re.findall('<ForeName>[^<]*</ForeName>', text)
    midnames = re.findall('<Initials>[^<]*</Initials>', text)
    lastnames = [ re.sub('</?LastName>','',x) for x in lastnames]
    firstnames = [re.sub('</?ForeName>', '', x) for x in firstnames]
    midnames = [re.sub('</?Initials>', '', x) for x in midnames]
    if len(lastnames) == 0 or len(firstnames) == 0 or len(midnames) == 0:
        return ''
    names = lastnames[0] + ', ' + firstnames[0] + ' ' + midnames[0]
    smallest = min([len(lastnames),len(firstnames),len(midnames)])
    for i in range(1,smallest):
        names += '; ' + lastnames[i] + ', ' + firstnames[i] + ' ' + midnames[i]
    return names


def get_df(path):
    data = {'title': [], 'authors': [], 'year': [], 'month': [], 'day': [], 'month': [], 'journal': [], 'abstract': []}

    for file in os.listdir(path):
        text = ''
        for line in open(path + file, 'r'):
            text += line.strip()
        fields = ['Year', 'Month', 'Day', 'Title']
        keys = ['year', 'month', 'day', 'journal']
        temp_dic = {}
        all_fields_found = True
        for i in range(len(fields)):
            info = get_field_text(text, fields[i])
            if info == '':
                all_fields_found = False
            if keys[i] not in temp_dic:
                temp_dic[keys[i]] = info
            else:
                temp_dic[keys[i]].append(info)
        title = get_title(text)
        abstract = get_abstract(text)
        authors = get_authors(text)
        if title != '' and abstract != '' and authors != '' and all_fields_found:
            data['title'].append(title)
            data['abstract'].append(abstract)
            data['authors'].append(authors)
            for key in temp_dic:
                data[key].append(temp_dic[key])
    return pd.DataFrame(data)

df_fake = get_df('/Users/michael/Downloads/PMID/')
df_real = get_df('/Users/michael/Downloads/random_dataset/')

df_fake['type'] = 'fake'
df_real['type'] = 'real'
df = pd.concat([df_real,df_fake])
df.to_csv('real_and_fake.tsv', sep='\t')

potential = get_df('/Users/michael/Downloads/20200426_potential_fakes_dataset/')
potential.to_csv('potentially_fake.tsv', sep='\t')