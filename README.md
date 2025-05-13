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

### Citation

If you use this code or methodology in your research, please cite the following:

```bibtex
@article {Kussaibi2025.05.11.25327389,
	author = {Kussaibi, Haitham},
	title = {LiteMIL: A Computationally Efficient Transformer-Based MIL for Cancer Subtyping on Whole Slide Images.},
	elocation-id = {2025.05.11.25327389},
	year = {2025},
	doi = {10.1101/2025.05.11.25327389},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/05/12/2025.05.11.25327389},
	journal = {medRxiv}
}```
