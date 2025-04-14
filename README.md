# WSI-MIL-Feature-Extraction

This repository provides a reproducible pipeline for:

- Splitting Whole Slide Images (WSIs) into patches using the Yottixel method.
- Extracting features from patches using the ViT-based Phikon model.
- Organizing extracted features into fixed-size bags for Multiple Instance Learning (MIL) tasks.

## Installation

```bash
git clone https://github.com/hkussaibi/WSI2bags.git
cd WSI2bags
pip install -r requirements.txt
```

## Usage
Run scripts sequentially:

Extract features directly from WSIs (patches processed on-the-fly):

```bash
python scripts/extract_features.py
```
Prepare fixed-size bags for MIL:

```bash
python scripts/prepare_bags.py.
```

Example Notebook

Check notebooks/example_usage.ipynb for an interactive example.