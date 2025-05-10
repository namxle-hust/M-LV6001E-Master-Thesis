# LV6001E

## Build & Run docker

```bash
# Build
docker build -t namxle/python:3.12.10-slim .

# Run
docker run -itv .:/workspace namxle/python:3.12.10-slim bash
```

## Download data

```bash
# Multi-omics data
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.gene-level_ascat2.tsv.gz && gunzip -ck TCGA-LUAD.gene-level_ascat2.tsv.gz > cnv.tsv

wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.methylation450.tsv.gz && gunzip -ck TCGA-LUAD.methylation450.tsv.gz > dnameth.tsv

# Log transformed
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_fpkm-uq.tsv.gz && gunzip -ck TCGA-LUAD.star_fpkm-uq.tsv.gz > mrna.tsv

# Log transformed
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.mirna.tsv.gz && gunzip -ck TCGA-LUAD.mirna.tsv.gz > mirna.tsv

# Survival data
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.survival.tsv.gz && gunzip -ck TCGA-LUAD.survival.tsv.gz > surival.tsv
```

## Run preprocess

```bash
# No missing values
python3 preprocess.py --input data/mrna.tsv --output mrna.clean.tsv --type mrna --num-features 2000 -s data/survival.tsv

# CNV gene level can fill missing values = 2 (For normal people)
python3 preprocess.py --input data/cnv.tsv --output cnv.clean.tsv --type cnv --num-features 1500 -s data/survival.tsv

# DNA methylation have 2 approaches for fill out missing values
python3 preprocess.py --input data/dnameth.tsv --output dnameth.knnimpute.clean.tsv --type dnameth --num-features 500 --fill-missing-method knnimpute
python3 preprocess.py --input data/dnameth.tsv --output dnameth.mean.clean.tsv --type dnameth --num-features 500 --fill-missing-method mean

# No missing values
python3 preprocess.py --input data/mirna.tsv --output mirna.clean.tsv --type mirna --num-features 300
```
