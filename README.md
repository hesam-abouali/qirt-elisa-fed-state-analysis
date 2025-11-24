# Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series - QIRT-ELISA Fed-State Analysis

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and the experimental _in vivo_ data accompanying the accepted paper, Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series at **NeurIPS 2025 Workshop: Learning from Time Series for Health (TS4H)**.

---

## Overview

This repository contains the **complete feature extraction** and **classification pipeline** for classifying metabolic phenotypes (healthy, pre-diabetic, and type 2 diabetic) of animal models using high-resolution QIRT-ELISA hormone measurements.

**QIRT-ELISA** (Quantum dot–Integrated Real-Time ELISA) enables continuous, multiplexed monitoring of:
- **Insulin**
- **Glucagon**  
- **C-peptide**

at **1-minute resolution** (15-fold improvement over conventional ELISA's 15-minute sampling intervals).

### Key Features

-  **184 physiologically-informed features** extracted from hormone time-series
-  **Three-stage feature selection and filteration** pipeline (variance → correlation → F-test)
-  **Four classical machine learning models** (Logistic Regression, SVM, Random Forest, KNN)
-  **89% classification accuracy** with Leave-One-Out Cross-Validation

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hesam-abouali/qirt-elisa-fed-state-analysis.git
cd qirt-elisa-fed-state-analysis

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Feature Extraction

Extract 184 physiologically-informed features from raw time-series data:

```python
from src.feature_extraction import ComprehensiveFedStateAnalyzer

# Initialize analyzer
analyzer = ComprehensiveFedStateAnalyzer()

# Run feature extraction
results = analyzer.run_complete_analysis(
    csv_path='data/fed_state_raw.csv',
    save_plots_dir='results/features',
    save_report_path='results/features/report.txt'
)

# Export features
analyzer.export_metrics_to_csv('results/comprehensive_metrics.csv')
```

### Step 2: Machine Learning Classification

Train and evaluate 4 ML models on the extracted features:

```python
from src.classification import EnhancedMultiModelClassifier

# Initialize classifier
classifier = EnhancedMultiModelClassifier()

# Run classification with all 4 models
results = classifier.run_complete_analysis(
    csv_path='results/comprehensive_metrics.csv',
    save_plot_path='results/model_comparison.png',
    export_csv=True,
    hyperparameter_tuning=True
)

# Results include:
# - Training & CV accuracy for all 4 models
# - Decision boundary visualizations
# - Individual animal predictions
# - Comprehensive CSV exports
```

### Quick Classification (Using Pre-extracted Features)

If you just want to run classification on the included features:

```python
from src.classification import EnhancedMultiModelClassifier

classifier = EnhancedMultiModelClassifier()

results = classifier.run_complete_analysis(
    csv_path='data/comprehensive_metrics.csv',
    save_plot_path='results/classification.png',
    export_csv=True
)
```

---

## Data Format

### Input Data (fed_state_raw.csv)

**Long format** CSV with 54 rows (9 animals × 6 timepoints):

| Column | Type | Description |
|--------|------|-------------|
| `animal_id` | str | Unique animal identifier |
| `rat_type` | str | Group: 'healthy', 'pre-diabetic', 'diabetic' |
| `timepoint` | int | Time point (0, 2.5, 5, 7.5, 10, 12.5 minutes) |
| `insulin_au` | float | Normalized insulin (Fi/F0) |
| `c_peptide_au` | float | Normalized C-peptide (Fi/F0) |
| `glucagon_au` | float | Normalized glucagon (Fi/F0) |
| `glucose_mM` | float | Glucose concentration (mM) |
| Additional ELISA columns... | | Cross-validation measurements |

### Output Data (comprehensive_metrics.csv)

**Wide format** CSV with 9 rows (one per animal) × 184 features:

- **Basic statistics**: Mean, CV, range, min, max for each hormone
- **Kinetics**: Rise/decline rates, peak characteristics, frequency
- **Oscillations**: FFT-based frequency, rhythmicity, entropy
- **AUC metrics**: Total, above-baseline, glucose-corrected
- **Correlations**: Hormone-hormone and hormone-glucose relationships
- **ELISA calibration**: Concentration estimates, calibration slopes

---

## Feature Categories

The 184 features are organized into **10 categories** per hormone (insulin, C-peptide, glucagon):

| Category | # Features | Description |
|----------|------------|-------------|
| **Basic Statistics** | 5 × 3 = 15 | Mean, CV, min, max, range |
| **AUC Metrics** | 5 × 3 = 15 | Total, baseline, glucose-corrected AUC |
| **Kinetics** | 7 × 3 = 21 | Rise/decline rates, peak counts, frequencies |
| **Peak Analysis** | 8 × 3 = 24 | Peak prominence, width, frequency, excursion |
| **Oscillations** | 6 × 3 = 18 | FFT frequency, rhythmicity, entropy, stability |
| **Rate of Change** | 7 × 3 = 21 | Maximum acceleration, variability, trends |
| **Glucose Relations** | 7 × 3 = 21 | Correlations, efficiency, sensitivity |
| **ELISA Calibration** | 7 × 3 = 21 | Concentration, slopes, secretion rates |
| **Hormone Correlations** | 6 × 3 = 18 | Cross-correlations, time lags |
| **Functional Scores** | 10 | Overall dysregulation scores |
| **Total** | **184** | |

---

## Machine Learning Models

### Feature Selection (Three-Stage Pipeline)

From 184 features → **15 top discriminative features**:

1. **Stage 1**: Variance filtering (remove low-variance features)
2. **Stage 2**: Correlation filtering (remove redundant features, r > 0.9)
3. **Stage 3**: F-test selection (top 15 by discriminative power)

### Top 15 Features Selected

Ranked by Logistic Regression coefficients and F-scores:

| Rank | Feature | LR Coef | F-Score | Interpretation |
|------|---------|---------|---------|----------------|
| 1 | Glucagon Mean Rise Rate | 0.301 | 14.82 | α-cell counter-regulation |
| 2 | Insulin Mean Rise Time | 0.262 | 7.80 | β-cell response kinetics |
| 3 | Insulin Peak Count | 0.254 | inf | Pulsatility patterns |
| 4 | Glucagon Concentration Range | 0.234 | 7.94 | Dynamic range |
| 5 | Glucagon Calibration Slope | 0.197 | 5.12 | Secretion efficiency |
| ... | ... | ... | ... | ... |
| 15 | Insulin Oscillation Frequency | 0.131 | 7.00 | Ultradian rhythm |

### Classification Results

| Model | Training Accuracy | CV Accuracy (LOO) | Overfitting Gap |
|-------|-------------------|-------------------|-----------------|
| **Logistic Regression** | 100% | **89%** | 0.11 |
| **SVM (RBF)** | 100% | **89%** | 0.11 |
| **Random Forest** | 89% | 56% | 0.33 |
| **KNN (k=3)** | 89% | 78% | 0.11 |

**Key Finding**: Logistic Regression and SVM achieve best performance.

---

## Key Scientific Findings

1. **α-cell dysfunction precedes β-cell failure** in T2D progression
   - Glucagon kinetics are the primary discriminator (6 of top 15 features)
   - Healthy rats: proper glucagon suppression after feeding
   - Diabetic rats: impaired α-cell glucose sensing

2. **Temporal resolution matters**
   - QIRT-ELISA (2.5-min) captures dynamics missed by conventional ELISA (15-min)
   - 6-fold improvement in sampling frequency
   - Enables detection of oscillatory patterns and rapid kinetics

3. **Simple models outperform complex ones**
   - Logistic Regression: 89% accuracy, biologically interpretable
   - Random Forest: 56% accuracy, overfitting due to small sample size (n=9)
   - Linear decision boundaries sufficient for metabolic phenotype separation

---

## Citation

If you use this code or data, please cite our paper:

### BibTeX

```bibtex
@inproceedings{abouali2025decoding,
  title={Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series},
  author={Abouali, Hesam and [Co-authors]},
  booktitle={NeurIPS 2025 Workshop on Learning from Time Series for Health},
  year={2025},
  organization={NeurIPS}
}
```

### QIRT-ELISA Technology Paper

```bibtex
@article{abouali2025qirtelisa,
  title={A Bead-Based Quantum Dot Immunoassay Integrated with Multi-Module Microfluidics Enables Real-Time Multiplexed Detection of Blood Insulin and Glucagon},
  author={Abouali, Hesam and Srikant, Sanjana and Al Fattah, Md Fahim and Barra, Nicole G and Chan, Darryl and Ban, Dayan and Schertzer, Jonathan D and Poudineh, Mahla},
  journal={Advanced Science},
  pages={2412185},
  month={April},
  year={2025},
  publisher={Wiley}
}
```

---

## Experimental Details

**Animal Models:**
- **Healthy**: Lean Zucker rats (n=3)
- **Pre-diabetic**: Obese normoglycemic Zucker rats (n=3)
- **Diabetic**: Dysglycemic ZDF rats (n=3)

**Protocol:**
- **Experiment**: Fed-state (no fasting, no glucose intervention)
- **Duration**: 15 minutes
- **Sampling**: Every 2.5 minutes (6 timepoints)
- **Measurements**: Insulin, C-peptide, glucagon (QIRT-ELISA) + glucose (glucometer)
- **Cross-validation**: Conventional ELISA at t=0 and t=15 minutes

---

## Contact

**Hesam Abouali**  
[hesam.abouali@uwaterloo.ca]

Department of Electrical & Computer Engineering  
University of Waterloo, Waterloo, ON, Canada


---

## Acknowledgments

This work was supported by the Canadian Institutes of Health Research (CIHR) and the Natural Sciences and Engineering Research Council of Canada (NSERC).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



