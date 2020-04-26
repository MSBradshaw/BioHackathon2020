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

#print(filenames)
pmc_ids=[]
open_access_ids=[]
for filename in filenames:
    if not os.path.isfile(filename):
        # Downloading...
        net_handle = Entrez.efetch(
            db='pubmed', id=filename, rettype='full',retmode='xml'
        )
        #out_handle = open(filename, 'w')
        #out_handle.write(net_handle.read())
        #out_handle.close()
        #net_handle.close()
        x=net_handle.readlines()
        #out_handle=Entrez.read(net_handle)

        for i in x:
            #print(i)
            if '<ArticleId IdType="pmc">' in i:
                #print(i.split('>')[1].split('<')[0])
                pmc_id=i.split('>')[1].split('<')[0]
                pmc_ids.append(pmc_id)
                open_access_ids.append(filename)


    #print(pmc_ids)
#print('Parsing...')



for filename in pmc_ids:
    if not os.path.isfile(filename):
        # Downloading...
        net_handle = Entrez.efetch(
            db='pmc', id=filename, rettype='full',retmode='xml'
        )
        out_handle = open(filename, 'w')
        out_handle.write(net_handle.read())
        out_handle.close()
        net_handle.close()
        #print('Saved')
#'PMC5806795'

print(open_access_ids)