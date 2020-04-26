import os
from Bio import Entrez
Entrez.email = 'sven.klumpe@hotmail.de'
#filenames = ['28402151']
filenames=[]
with open('fake_pmids.txt','r') as input_file:
    lines=input_file.readlines()
    for line in lines:
        pmid=line.split(':')[1]
        filenames.append(pmid)


for filename in filenames:
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
#'PMC5806795'

#print(open_access_ids)