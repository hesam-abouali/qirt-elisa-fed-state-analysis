# Data Directory

This directory contains the experimental data from QIRT-ELISA fed-state experiments.

## Files

### 1. `fed_state_raw.csv`

**Description**: Raw QIRT-ELISA measurements from fed-state experiments

**Format**: Long format (one row per timepoint per animal)

**Size**: 54 rows × 14 columns (9 animals × 6 timepoints each)

**Columns**:

| Column Name | Type | Unit | Description |
|------------|------|------|-------------|
| `animal_id` | string | - | Unique identifier for each animal |
| `rat_type` | string | - | Metabolic phenotype (`healthy`, `pre-diabetic`, `diabetic`) |
| `timepoint` | float | minutes | Time of measurement (0, 2.5, 5, 7.5, 10, 12.5, 15) |
| `insulin_au` | float | a.u. | QIRT-ELISA insulin measurement (normalized F/F0) |
| `c_peptide_au` | float | a.u. | QIRT-ELISA C-peptide measurement (normalized F/F0) |
| `glucagon_au` | float | a.u. | QIRT-ELISA glucagon measurement (normalized F/F0) |
| `glucose_start_mM` | float | mM | Glucometer reading at experiment start |
| `glucose_end_mM` | float | mM | Glucometer reading at experiment end |
| `insulin_start_ELISA_pM` | float | pM | Conventional ELISA insulin at start |
| `insulin_end_ELISA_pM` | float | pM | Conventional ELISA insulin at end |
| `c_peptide_start_ELISA_pM` | float | pM | Conventional ELISA C-peptide at start |
| `c_peptide_end_ELISA_pM` | float | pM | Conventional ELISA C-peptide at end |
| `glucagon_start_ELISA_pM` | float | pM | Conventional ELISA glucagon at start |
| `glucagon_end_ELISA_pM` | float | pM | Conventional ELISA glucagon at end |

**Rat Groups**:
- **Healthy** (n=3): Lean Zucker rats
- **Pre-diabetic** (n=3): Obese normoglycemic Zucker rats
- **Diabetic** (n=3): Dysglycemic ZDF (Zucker Diabetic Fatty) rats

---

### 2. `comprehensive_metrics.csv`

**Description**: Extracted features from the analysis pipeline

**Format**: Wide format (one row per animal)

**Size**: 9 rows × 184 columns

**Content**: 184 physiologically-informed features across 10 categories:

1. **Basic Device Measurements** (15 metrics)
   - Mean, range, CV, max, min for each biomarker

2. **Glucose-Adjusted Parameters** (21 metrics)
   - Glucose-corrected values, efficiency ratios, correlations

3. **Peak Dynamics** (21 metrics)
   - Peak count, frequency, prominence, width, dynamic range

4. **Rate Analysis** (21 metrics)
   - Maximum increase/decrease rates, acceleration, trends

5. **Area Under Curve (AUC)** (15 metrics)
   - Total AUC, above/below baseline, glucose-corrected

6. **Oscillation Patterns** (18 metrics)
   - Frequency, zero crossings, rhythmicity, entropy

7. **Inter-Biomarker Correlations** (18 metrics)
   - Pearson correlations, cross-correlations, lags

8. **ELISA Quantitative** (21 metrics)
   - Concentration means, ranges, calibration quality

9. **Pancreatic Function** (11 metrics)
   - Beta-cell and alpha-cell function scores

10. **Peak Kinetics** (21 metrics)
    - Rise/decline times and rates for each biomarker

See [`feature_dictionary.md`](feature_dictionary.md) for complete descriptions of all 184 features.

---

## Data Privacy

All data has been de-identified:
- Animal IDs are anonymized codes
- No traceable information included
- Complies with institutional animal care protocols

---

## Usage Example

```python
import pandas as pd

# Load raw data
raw_data = pd.read_csv('data/fed_state_raw.csv')

# Load extracted features
features = pd.read_csv('data/comprehensive_metrics.csv')

# View structure
print(f"Raw data: {raw_data.shape}")
print(f"Features: {features.shape}")
```

---

## Experimental Protocol

**Experiment Type**: Fed-state (non-fasted)

**Duration**: 15 minutes

**Sampling Frequency**: 
- QIRT-ELISA: 1-minute raw measurements, averaged to 2.5-minute intervals
- Conventional ELISA: Start and end points only (15-minute interval)
- Glucometer: Start and end points only

**Animals**: 
- Conscious rats with jugular catheter
- Blood drawn continuously via peristaltic pump
- Normal physical activity during measurement

---

## Citation

If you use this data, please cite:

```bibtex
@inproceedings{abouali2025qirt,
  title={Decoding Type 2 Diabetes Progression via Metabolic Hormone Time-Series},
  author={Abouali, Hesam and Srikant, Sanjana and Barra, Nicole G. and Etemad, Ali and Schertzer, Jonathan D. and Poudineh, Mahla},
  booktitle={39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Learning from Time Series for Health},
  year={2025}
}
```

---

**Last Updated**: November 2025
