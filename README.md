# RNA 3D Folding Prediction

This project implements a deep learning model for RNA 3D folding prediction using TensorFlow/Keras.

## Project Structure

```
rna-folding/
├── src/
│   └── notebook_conversion.py  # Main implementation file
├── data/                      # Data directory (not included in repo)
├── pyproject.toml            # Poetry dependencies
└── README.md                 # This file
```

## Setup

1. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Place your data files in the `data/` directory:
- train_sequences.csv
- train_labels.csv
- validation_sequences.csv
- validation_labels.csv
- test_sequences.csv
- sample_submission.csv

## Usage

Run the main script:
```bash
poetry run python src/notebook_conversion.py
```

## Dependencies

- Python >= 3.8
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
