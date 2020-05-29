lines_per_file = 10000
count = 0
outfile = open('delete-this.csv','w')
current_file = None
with open('all_abstracts.tsv','r') as infile:
    for line in infile:
        if count % lines_per_file == 0:
            current_file = 'Subsets/subset_'+str(count)+'.tsv'
            outfile.close()
            outfile = open(current_file,'w')
        outfile.write(line)
        count += 1
    outfile.close()