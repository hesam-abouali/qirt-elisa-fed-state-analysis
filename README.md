# QIRT-ELISA Fed-State Analysis

**Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series**

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data accompanying our paper accepted at **NeurIPS 2025 Workshop: Learning from Time Series for Health (TS4H)**.

---

## Overview

This repository contains the complete analysis pipeline for classifying metabolic phenotypes (healthy, pre-diabetic, and type 2 diabetic) using high-resolution QIRT-ELISA hormone measurements.

**QIRT-ELISA** (Quantum dot–Integrated Real-Time ELISA) enables continuous, multiplexed monitoring of:
- Insulin
- Glucagon  
- C-peptide

at **1-minute resolution** (15-fold improvement over conventional ELISA's 15-minute intervals).

**ELISA**

The conventional ELISA was used to cross-validate QIRT-ELISA at two time points of the experiment:

- t = 0 min
- t = 15 min

### Key Features

- **184 physiologically-informed features** extracted from hormone time-series data
- **Three-stage feature selection** pipeline (variance → correlation → F-test)
- **Machine learning classification** (Logistic Regression, SVM, Random Forest, KNN)
- **89% classification accuracy** with Leave-One-Out Cross-Validation
- **Comprehensive documentation** with all metric definitions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qirt-elisa-fed-state-analysis.git
cd qirt-elisa-fed-state-analysis

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.feature_extraction import ComprehensiveFedStateAnalyzer

# Initialize analyzer
analyzer = ComprehensiveFedStateAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis(
    csv_path='data/fed_state_raw.csv',
    save_plots_dir='results',
    save_report_path='results/report.txt'
)

# Export metrics
analyzer.export_metrics_to_csv('results/metrics.csv')
```

---

## Data Format

### Input Data Format (Long Format CSV)

Required columns:
```
animal_id, rat_type, timepoint, 
insulin_au, c_peptide_au, glucagon_au,
glucose_start_mM, glucose_end_mM,
insulin_start_ELISA_pM, insulin_end_ELISA_pM,
c_peptide_start_ELISA_pM, c_peptide_end_ELISA_pM,
glucagon_start_ELISA_pM, glucagon_end_ELISA_pM
```

- **animal_id**: Unique identifier for each animal
- **rat_type**: Metabolic phenotype (`healthy`, `pre-diabetic`, `diabetic`)
- **timepoint**: Time in minutes (e.g., 0, 2.5, 5, 7.5, 10, 12.5, 15)
- **\*_au**: QIRT-ELISA normalized measurements (F/F0)
- **glucose_\*_mM**: Glucometer measurements in mM
- **\*_ELISA_pM**: Conventional ELISA cross-validation in pM

### Output: 184 Features

The pipeline extracts 184 comprehensive features across 10 categories:

1. **Basic Device Measurements** (15 metrics)
2. **Glucose-Adjusted Parameters** (21 metrics)
3. **Peak Dynamics** (21 metrics)
4. **Rate Analysis** (21 metrics)
5. **Area Under Curve (AUC)** (15 metrics)
6. **Oscillation Patterns** (18 metrics)
7. **Inter-Biomarker Correlations** (18 metrics)
8. **ELISA Quantitative** (21 metrics)
9. **Pancreatic Function** (11 metrics)
10. **Peak Kinetics** (21 metrics)

See [`data/feature_dictionary.md`](data/feature_dictionary.md) for complete descriptions.

---

## Methodology

### Three-Stage Feature Selection

1. **Variance Filtering**: Remove low-variance features (threshold: 0.01)
   - 181 → 177 features

2. **Correlation Filtering**: Remove redundant features (correlation > 0.95)
   - 177 → 120 features

3. **Univariate F-Test**: Select top 15 most discriminative features
   - 120 → 15 features

### Classification Models

- **Logistic Regression** (Best: 89% accuracy)
- **Support Vector Machine** (89% accuracy)
- **Random Forest** (56% cross-validation accuracy)
- **K-Nearest Neighbors** (78% accuracy)

### Cross-Validation

- **Leave-One-Out Cross-Validation (LOO-CV)** for small sample size (n=9)
- **Permutation testing** (1000 iterations) for statistical significance

---

## Key Results

### Top Discriminative Features

1. **Glucagon Mean Rise Rate** (GMRS)
   - Negative values: Pre-diabetic α-cell hypercompensation
   - Positive values: Diabetic α-cell dysfunction

2. **Insulin Mean Rise Time** (IMRT)
   - Sustained in healthy rats
   - Shortened in pre-diabetic/diabetic rats (β-cell exhaustion)

### Classification Performance

| Model | Training Acc. | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression | 100% | **89%** |
| SVM | 100% | 89% |
| Random Forest | 100% | 56% |
| KNN | 100% | 78% |

**Statistical Significance**: p < 0.05 (permutation testing)

---

## Citation

If you use this code or data, please cite our NeurIPS 2025 workshop paper:

```bibtex
@inproceedings{abouali2025qirt,
  title={Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series},
  author={Abouali, Hesam and Srikant, Sanjana and Barra, Nicole G. and Etemad, Ali and Schertzer, Jonathan D. and Poudineh, Mahla},
  booktitle={39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Learning from Time Series for Health},
  year={2025}
}
```

**Paper**: [Link to paper will be added upon publication]

**Original QIRT-ELISA Technology**:
```bibtex
@article{abouali2025bead,
  title={A Bead-Based Quantum Dot Immunoassay Integrated with Multi-Module Microfluidics Enables Real-Time Multiplexed Detection of Blood Insulin and Glucagon},
  author={Abouali, Hesam and Srikant, Sanjana and Al Fattah, Md Fahim and Barra, Nicole G. and Chan, Darryl and Ban, Dayan and Schertzer, Jonathan D. and Poudineh, Mahla},
  journal={Advanced Science},
  pages={2412185},
  year={2025}
}
```

---

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- Seaborn ≥ 0.11
- Scikit-learn ≥ 1.0

See [`requirements.txt`](requirements.txt) for complete dependencies.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Hesam Abouali** - University of Waterloo - hesam.abouali@uwaterloo.ca
- **Sanjana Srikant** - University of Waterloo - sanjana.srikant@uwaterloo.ca
- **Nicole G. Barra** - McMaster University - barrang@mcmaster.ca
- **Ali Etemad** - Queen's University - ali.etemad@queensu.ca
- **Jonathan D. Schertzer** - McMaster University - schertze@mcmaster.ca
- **Mahla Poudineh** - University of Waterloo - mahla.poudineh@uwaterloo.ca

---

## Acknowledgments

This research was supported by the Canadian Institutes of Health Research (CIHR) and the Natural Sciences and Engineering Research Council of Canada (NSERC).

---

## Contact

For questions or collaboration inquiries:
- **Hesam Abouali**: hesam.abouali@uwaterloo.ca
- **Dr. Mahla Poudineh**: mahla.poudineh@uwaterloo.ca

---

## 🔗 Related Links
[QIRT-ELISA Technology Paper (Advanced Science)](https://doi.org/10.1002/advs.202412185)
[NeurIPS 2025 TS4H Workshop](https://timeseries4health.github.io/)

---

**Last Updated**: November 23, 2025
