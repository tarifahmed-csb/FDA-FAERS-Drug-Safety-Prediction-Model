# FDA FAERS Drug Safety Prediction Model

A machine learning project that predicts adverse drug events from patient and drug characteristics using the FDA FAERS (FDA Adverse Event Reporting System) database. This model helps drug safety teams prioritize reviews of high-risk reports.

## Overview

This project analyzes FDA adverse event reports from Q2 2025 to:
- Identify patterns in serious vs. non-serious adverse events
- Build predictive models using patient demographics and drug information
- Determine which factors most strongly predict serious outcomes

### Key Findings

| Finding | Detail |
|---------|--------|
| **Age Pattern** | Elderly (65+) have 59.5% serious outcome rate vs. 47.0% for young adults |
| **Sex Difference** | Males have 7.7% higher serious outcome rate than females |
| **Most Common Serious Outcome** | Hospitalization (~27%), followed by Death (~9.5%) |
| **Top Predictive Features** | Number of drugs, dose amount, route of administration |

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:tarifahmed-csb/FDA-FAERS-Drug-Safety-Prediction-Model.git
```

### 2. Download the FAERS data

Download the Q2 2025 ASCII data files from the [FDA FAERS website](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html) and extract them into an `ASCII/` folder in the project directory.

Required files:
- `DEMO25Q2.txt` - Patient demographics
- `DRUG25Q2.txt` - Drug information
- `REAC25Q2.txt` - Adverse reactions
- `OUTC25Q2.txt` - Patient outcomes
- `INDI25Q2.txt` - Drug indications
- `THER25Q2.txt` - Therapy dates
- `RPSR25Q2.txt` - Report sources

### 3. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the full analysis

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python fda_faers_analysis.py
```

The script will:
1. Load and clean all 7 FAERS data tables
2. Perform exploratory data analysis
3. Engineer features from demographics and drug data
4. Train and evaluate 4 machine learning models
5. Output results and save confusion matrix plots

### Output Files

| File | Description |
|------|-------------|
| `FAERS25Q2_CLEANED.csv` | Cleaned and merged dataset |
| `model_comparison_results.csv` | Performance metrics for all models |
| `*_confusion_matrix.png` | Confusion matrix visualizations |

## Project Structure

```
Drug Safety/
├── ASCII/                          # FAERS data files (download separately)
│   ├── DEMO25Q2.txt
│   ├── DRUG25Q2.txt
│   ├── REAC25Q2.txt
│   ├── OUTC25Q2.txt
│   ├── INDI25Q2.txt
│   ├── THER25Q2.txt
│   └── RPSR25Q2.txt
├── course documents/               # Project report and presentation
├── fda_faers_analysis.py          # Main analysis script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Models

The project trains and compares 4 classification models:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear model with balanced class weights |
| **Decision Tree** | Interpretable tree-based classifier (max depth=5) |
| **Random Forest** | Ensemble of 100 trees for robust predictions |
| **K-Nearest Neighbors** | Instance-based learning (k=5) |

### Expected Performance

| Model | ROC AUC | Accuracy |
|-------|---------|----------|
| Logistic Regression | ~0.80 | ~71% |
| Random Forest | ~0.78 | ~73% |
| KNN | ~0.83 | ~77% |
| Decision Tree | ~0.70 | ~68% |

## Data Source

**FDA FAERS Database**  
The FDA Adverse Event Reporting System (FAERS) is a database containing adverse event reports, medication error reports, and product quality complaints submitted to the FDA.

- Website: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- Data used: Q2 2025 (April - June 2025)
- Format: Dollar-sign ($) delimited text files

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## License

This project is for educational purposes.

## Acknowledgments

- FDA for providing the FAERS public dataset
- DSCI 310 course project
