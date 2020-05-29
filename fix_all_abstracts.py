infile = open('all_abstracts.tsv','r')
outfile_5 = open('abstracts-5-cols.tsv','w')
outfile_6 = open('abstracts-6-cols.tsv','w')
outfile_other = open('abstracts-other-cols.tsv','w')


for line in infile:
    count = line.count('\t')
    print(count)
    if count == 5:
        outfile_5.write(line)
    elif count == 6:
        outfile_6.write(line)
    else:
        outfile_other.write(line)

infile.close()
outfile_5.close()
outfile_6.close()
outfile_other.close()