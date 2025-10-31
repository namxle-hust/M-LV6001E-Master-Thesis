# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis project (LV6001E) implementing a Contractive AutoEncoder (CtAE) based approach for cancer subtype classification using multi-omics data from TCGA-LUAD (Lung Adenocarcinoma). The pipeline performs feature extraction, dimensionality reduction, and survival-based clustering.

## Development Environment

### Docker Setup
```bash
# Build Docker image
docker build -t namxle/python:3.12.10-slim .

# Run container with workspace mounted
docker run -itv .:/workspace namxle/python:3.12.10-slim bash
```

### Dependencies
Python 3.12.10 with packages defined in `requirements.txt`:
- PyTorch (deep learning)
- lifelines (survival analysis)
- scikit-learn, scikit-image (ML utilities)
- pandas, numpy (data processing)
- seaborn (visualization)

## Pipeline Architecture

The complete pipeline follows this sequence:

### 1. Data Acquisition & Preparation
Multi-omics data from TCGA-LUAD:
- **mRNA**: Gene expression (star_tpm.tsv, log-transformed)
- **CNV**: Copy number variation (gene-level_absolute.tsv)
- **DNA Methylation**: Methylation450 data
- **miRNA**: microRNA expression (log-transformed)
- **Survival**: Clinical survival data

Sample filtering script `scripts/getsamples.py` identifies common samples across all omics types and generates `data/sample_ids.txt` for downstream filtering.

### 2. Preprocessing (`scripts/preprocessv3.py`)
Current version (v3) performs:
- **Missing value handling**: KNN imputation for CNV and DNA methylation data
- **Feature selection**: Gene-wise scoring (mean × std) as per CtAE paper
- **Sample filtering**: Only keeps samples in `sample_ids.txt`
- **No normalization** by default (data already log-transformed or normalized)

Key parameters by omics type (from README):
- mRNA: 2000 features, no missing values
- CNV: 1500 features, KNN imputation
- DNA methylation: 1000 features, KNN imputation
- miRNA: 300 features, no missing values

Use `--num-features -1` to keep all features without selection.

### 3. CAE Training (`cae_model_v2.py`)
Implements `ContractiveAutoEncoder` with:
- Configurable hidden dimensions (default: [5000, 128, 64])
- Contractive penalty term in loss function
- Multiple initialization methods (Xavier, He)
- Optional batch normalization and dropout
- Saves: model weights, scalers, extracted features (.npy files)

Run for each omics type separately:
```bash
python3 cae_model_v2.py -i <preprocessed_data> -o models/<omics>.out
```

Outputs:
- `<omics>.out_model.pth`: Trained model weights
- `<omics>.out_scaler.pkl`: MinMax scaler
- `<omics>.out_ef.npy`: Extracted features (bottleneck layer)

### 4. Subtype Classification (`main.py` → `subtype_ctae.py`)
`SubtypeCtAE` class orchestrates:
1. **Load CAE-extracted features** from `.npy` files for all omics
2. **Cox regression feature selection**: Filters features by survival correlation (p-value threshold)
3. **Feature integration**: Concatenates selected features from all omics types
4. **Clustering**: K-means on integrated features (optimizes K by silhouette score)
5. **Survival analysis**: Log-rank test and Cox PH model for subtype validation

Key outputs saved by `TrainingInfoSaver` (in `save_training_info.py`):
- `feature_selection_info.pkl`: Selected feature indices per omics
- `cluster_info.pkl`: Cluster centroids and labels
- `training_summary.pkl`: Performance metrics (C-index, p-value, silhouette score)

### 5. Prediction (`predict.py`)
`SubtypeCtAEPredictor` class for classifying new patients:
- Loads trained CAE models and metadata
- Applies same preprocessing and feature selection
- Extracts features using trained CAE models
- Assigns subtypes based on distance to cluster centroids

## Common Commands

### Full Pipeline Execution
```bash
# Complete workflow using run.sh
./run.sh
```

This script:
1. Runs preprocessing for all 4 omics types → `pp/` directory
2. Trains CAE models for each omics → `models/` directory
3. Runs subtype classification → final clustering results

### Individual Steps

**Preprocessing** (specific feature counts):
```bash
outdir=ppv7
rm -rf ${outdir} && mkdir -p ${outdir}

python3 preprocessv3.py --input data/mrna.tsv --output ${outdir}/mrna.clean.tsv --type mrna --num-features 2000 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

python3 preprocessv3.py --input data/cnv.tsv --output ${outdir}/cnv.clean.tsv --type cnv --num-features 1500 --sample-ids data/sample_ids.txt --fill-missing-method knnimpute --feature-selection genewise --no-normalization

python3 preprocessv3.py --input data/dnameth.tsv --output ${outdir}/dnameth.clean.tsv --type dnameth --num-features 1000 --fill-missing-method knnimpute --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization

python3 preprocessv3.py --input data/mirna.tsv --output ${outdir}/mirna.clean.tsv --type mirna --num-features 300 --sample-ids data/sample_ids.txt --feature-selection genewise --no-normalization
```

**CAE Training**:
```bash
rm -rf models && mkdir -p models

python3 cae_model_v2.py -i ppv7/mrna.clean.tsv -o models/mrna.out
python3 cae_model_v2.py -i ppv7/cnv.clean.tsv -o models/cnv.out
python3 cae_model_v2.py -i ppv7/dnameth.clean.tsv -o models/dnameth.out
python3 cae_model_v2.py -i ppv7/mirna.clean.tsv -o models/mirna.out
```

**Subtype Classification**:
```bash
python3 main.py
```

**Cox Regression Analysis** (standalone):
```bash
python3 scripts/cox.py
```

## Key Architecture Details

### Data Flow
```
Raw TCGA data → Preprocessing (feature selection, imputation) →
CAE (dimensionality reduction) → Cox feature selection →
Feature integration → K-means clustering → Survival validation
```

### Critical Files
- `scripts/preprocessv3.py`: Latest preprocessing with gene-wise feature selection
- `cae_model_v2.py`: Improved CAE with contractive loss
- `subtype_ctae.py`: Core clustering and survival analysis logic
- `main.py`: Entry point for complete training workflow
- `predict.py`: Inference pipeline for new samples
- `save_training_info.py`: Serialization utilities for trained models

### Model Versioning
The codebase shows evolution through versions (v1, v2, v3 noted in git history and file names):
- `preprocess.py` → `preprocessv2.py` → `preprocessv3.py`
- `cae_model.py` → `cae_model_v2.py`
- Check git commits and version notes when modifying pipeline components

### Data Directories
- `data/`: Raw and filtered omics data, sample IDs, survival data
- `pp*/` or `ppv*/`: Preprocessed data output directories (versioned)
- `models/`: Trained CAE models, scalers, extracted features, clustering metadata
- `examples/`: Reference usage examples (if available)

## Testing & Validation

No formal test suite exists. Validation is performed through:
- Silhouette scores for clustering quality
- Log-rank test p-values for subtype separation
- Concordance index (C-index) for survival prediction

When modifying the pipeline, verify:
1. Feature dimensions match at each stage
2. Sample IDs align across omics types
3. Survival data merges correctly with features
4. Model outputs (`.npy`, `.pkl`, `.pth` files) are generated
