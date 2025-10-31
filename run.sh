#!/bin/bash

ppdir=pp

rm -rf ${ppdir} && mkdir -p ${ppdir}

# No missing values
python3 preprocessv3.py --input data/mrna.tsv --output ${ppdir}/mrna.clean.tsv --type mrna --num-features -1 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

# CNV gene level
python3 preprocessv3.py --input data/cnv.tsv --output ${ppdir}/cnv.clean.tsv --type cnv --num-features -1 --sample-ids data/sample_ids.txt --fill-missing-method knnimpute --feature-selection genewise --no-normalization

# DNA methylation
python3 preprocessv3.py --input data/dnameth.tsv --output ${ppdir}/dnameth.clean.tsv --type dnameth --num-features -1 --fill-missing-method knnimpute --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

# No missing values
python3 preprocessv3.py --input data/mirna.tsv --output ${ppdir}/mirna.clean.tsv --type mirna --num-features -1 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

# Run CAE model
rm -rf models && mkdir -p models

python3 cae_model_v2.py -i ${ppdir}/mrna.clean.tsv -o models/mrna.out

python3 cae_model_v2.py -i ${ppdir}/cnv.clean.tsv -o models/cnv.out

python3 cae_model_v2.py -i ${ppdir}/dnameth.clean.tsv -o models/dnameth.out

python3 cae_model_v2.py -i ${ppdir}/mirna.clean.tsv -o models/mirna.out

# Final
python3 main.py
