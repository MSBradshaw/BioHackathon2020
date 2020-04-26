import pandas as pd
import re
import os

def get_field_text(text, field):
    pattern = '<FIELD>[^<]*</FIELD>'.replace('FIELD', field)
    res = re.findall(pattern, text)
    final = res[0].replace('<FIELD>'.replace('FIELD', field), '')
    final = final.replace('</FIELD>'.replace('FIELD', field), '')
    return final


def get_title(text):
    pattern = '<ArticleTitle>.*</ArticleTitle>'
    res = re.findall(pattern, text)
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
    names = lastnames[0] + ', ' + firstnames[0] + ' ' + midnames[0]
    smallest = min([len(lastnames),len(firstnames),len(midnames)])
    for i in range(1,smallest):
        names += '; ' + lastnames[i] + ', ' + firstnames[i] + ' ' + midnames[i]
    return names


path = '/Users/michael/Downloads/PMID/'

data = {'title': [], 'authors': [], 'year': [], 'month': [], 'day': [], 'month': [], 'journal': [], 'abstract': []}

for file in os.listdir(path):
    text = ''
    for line in open(path + file, 'r'):
        text += line.strip()
    print(file)
    fields = ['Year', 'Month', 'Day', 'Title']
    keys = ['year', 'month', 'day', 'journal']
    for i in range(len(fields)):
        res = get_field_text(text, fields[i])
        data[keys[i]].append(res)
    data['title'].append(get_title(text))
    data['abstract'].append(get_abstract(text))
    data['authors'].append(get_authors(text))

df = pd.DataFrame(data)
df.to_csv('422-abstracts.tsv', sep='\t')
