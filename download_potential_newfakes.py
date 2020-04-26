import pandas as pd
import os

data=pd.read_csv('../TSV/422-abstracts.tsv', sep='\t')

authors=data['authors']

author_list=[]
for i in range(0,len(authors)):
    for author in authors[i].split(';'):
        if author in author_list:
            continue
        else:
            author_list.append(author)

### ESEARCH FOR PMIDS OF DIFFERENT PAPERS OF SAME AUTHORS
from Bio import Entrez
Entrez.email = "sven.klumpe@hotmail.de"

PMIDs=[]
with open('potentially_fake_IDs.txt','w') as output_file:
    for author in author_list:
        if len(author.split(' '))>=3:
            author=author[:-1]

        handle = Entrez.esearch(db="pubmed", retmax=10, rettype='uilist', term=str(author)+'[Auth]')#+2008[pdat]")#, idtype="acc")
        record = Entrez.read(handle)
        handle.close()

        idList=record['IdList']
        for ID in idList:
            if ID in PMIDs:
                continue
            else:
                PMIDs.append(ID)
                output_file.write('PMID:'+str(ID)+'\n')
        
print(PMIDs)

for filename in PMIDs:
    if not os.path.isfile(filename):
        # Downloading...
        net_handle = Entrez.efetch(
            db='pubmed', id=filename, rettype='full',retmode='xml'
        )
        out_handle = open(filename, 'w')
        out_handle.write(net_handle.read())
        out_handle.close()
        net_handle.close()
        print('Saved')