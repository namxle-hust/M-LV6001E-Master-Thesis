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

## CNV
# wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.gene-level_absolute.tsv.gz
gunzip -ck TCGA-LUAD.gene-level_absolute.tsv.gz > cnv.tsv
# gunzip -ck TCGA-LUAD.gene-level_ascat3.tsv.gz > cnv.tsv
# gunzip -ck TCGA-LUAD.gene-level_ascat2.tsv.gz > cnv.tsv

## DNA methylation
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.methylation450.tsv.gz && gunzip -ck TCGA-LUAD.methylation450.tsv.gz > dnameth.tsv

## mRNA
# Log transformed
# wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_tpm.tsv.gz && gunzip -ck TCGA-LUAD.star_tpm.tsv.gz > mrna.tsv
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_fpkm-uq.tsv.gz && gunzip -ck TCGA-LUAD.star_fpkm-uq.tsv.gz > mrna.tsv

## miRNA
# Log transformed
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.mirna.tsv.gz && gunzip -ck TCGA-LUAD.mirna.tsv.gz > mirna.tsv

# Survival data
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.survival.tsv.gz && gunzip -ck TCGA-LUAD.survival.tsv.gz > surival.tsv
```

## Get common samples

```bash
python3 getsamples.py && \
awk -F"\t" 'FNR==NR{ a[$1] = 1;next}{ if(FNR==1 || a[$1] == 1){ print; } }' data/sample_ids.txt data/survival.tsv > data/survival.filtered.tsv
```

## Run preprocess

```bash
outdir=ppv6

rm -rf ${outdir} && mkdir -p ${outdir}

# No missing values
python3 preprocessv3.py --input data/mrna.tsv --output ${outdir}/mrna.clean.tsv --type mrna --num-features 2000 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

# CNV gene level
python3 preprocessv3.py --input data/cnv.tsv --output ${outdir}/cnv.clean.tsv --type cnv --num-features 1500 --sample-ids data/sample_ids.txt --fill-missing-method knnimpute --feature-selection genewise --no-normalization

# DNA methylation
python3 preprocessv3.py --input data/dnameth.tsv --output ${outdir}/dnameth.clean.tsv --type dnameth --num-features 1000 --fill-missing-method knnimpute --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

# No missing values
python3 preprocessv3.py --input data/mirna.tsv --output ${outdir}/mirna.clean.tsv --type mirna --num-features 300 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization
```

## Run Cox

```bash
python3 cox.py
```

## Training

### CAE

```bash
rm -rf models && mkdir -p models

python3 cae_model_v2.py -i ppv6/mrna.clean.tsv -o models/mrna.out

python3 cae_model_v2.py -i ppv6/cnv.clean.tsv -o models/cnv.out

python3 cae_model_v2.py -i ppv6/dnameth.clean.tsv -o models/dnameth.out

python3 cae_model_v2.py -i ppv6/mirna.clean.tsv -o models/mirna.out
```

## Subtype Model

```bash
python3 main.py
```

## Get test sample
```bash

```