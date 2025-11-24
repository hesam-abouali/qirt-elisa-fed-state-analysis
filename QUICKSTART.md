# QIRT-ELISA Fed-State Analysis - Quick Start Guide

Get up and running in **5 minutes**!

---

## Step-by-Step Workflow

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Extract Features

```python
from src.feature_extraction import ComprehensiveFedStateAnalyzer

analyzer = ComprehensiveFedStateAnalyzer()

# Extract 184 features from raw data
results = analyzer.run_complete_analysis(
    csv_path='data/fed_state_raw.csv',
    save_plots_dir='results/features',
    save_report_path='results/report.txt'
)

# Export to CSV
analyzer.export_metrics_to_csv('results/comprehensive_metrics.csv')
```

### Step 3: Run Classification

```python
from src.classification import EnhancedMultiModelClassifier

classifier = EnhancedMultiModelClassifier()

# Train all 4 models
results = classifier.run_complete_analysis(
    csv_path='results/comprehensive_metrics.csv',
    save_plot_path='results/classification.png',
    export_csv=True
)
```

---

## Quick Classification (Using Pre-extracted Features)

If you want to skip feature extraction and just run classification:

```python
from src.classification import EnhancedMultiModelClassifier

classifier = EnhancedMultiModelClassifier()

# Use the included comprehensive_metrics.csv
results = classifier.run_complete_analysis(
    csv_path='data/comprehensive_metrics.csv',
    save_plot_path='results/classification.png',
    export_csv=True
)
```

---

## View Results

After running, check:

```
results/
├── comprehensive_metrics.csv              # 184 features
├── classification.png                     # Model comparison
├── comprehensive_metrics_multimodel_results.csv
├── comprehensive_metrics_model_summary.csv
└── features/
    ├── plots/                            # Time-series visualizations
    └── report.txt                        # Feature extraction report
```

---

## Expected Output

You should see:

```
✓ Feature extraction complete!
✓ 184 features extracted

PERFORMANCE RANKING:
1. Logistic Regression: CV=0.889
2. SVM                : CV=0.889
3. KNN                : CV=0.778
4. Random Forest      : CV=0.556

✓ SUCCESS: Analysis completed successfully!
```

---

## Common Issues

### Issue: ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: File not found

**Solution:** Make sure you're in the repository root directory:
```bash
cd qirt-elisa-fed-state-analysis
ls  # Should see: data/, src/, README.md, etc.
```

### Issue: Import errors

**Solution:** Add src to Python path:
```python
import sys
sys.path.append('.')
from src.feature_extraction import ComprehensiveFedStateAnalyzer
```

---

## What's Next?

1. **Explore the data**: Check `data/README.md` for detailed data documentation
2. **Modify parameters**: Edit hyperparameters in the scripts
3. **Custom analysis**: Use `src/example_usage.py` as a template
4. **Read the paper**: See full methodology in our NeurIPS 2025 paper

---

## Need Help?

- **Documentation**: See main [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **Questions**: Contact the authors

---

**Happy analyzing! 🚀**
