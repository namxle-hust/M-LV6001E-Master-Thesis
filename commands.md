```bash
awk -F"," 'FNR==NR{ a[$1]=1;next; }{ if(a[$1]==1){print}; }' mirna.list.txt processed/mirtarbase_processed.csv > processed/mirtarbase.filtered.csv

awk -F"," 'FNR==NR{ a[$1]=1;next; }{ if(a[$1]==1){print}; }' mirna.list.txt processed/targetscan_processed.csv > processed/targetscan.filtered.csv
```