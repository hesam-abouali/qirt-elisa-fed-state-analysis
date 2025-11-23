# Quick Start Guide

Get up and running with QIRT-ELISA analysis in 5 minutes!

## 📥 Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/qirt-elisa-fed-state-analysis.git
cd qirt-elisa-fed-state-analysis
```

## 🐍 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n qirt python=3.8
conda activate qirt
pip install -r requirements.txt
```

## 🚀 Step 3: Run Example Analysis

```bash
python src/example_usage.py
```

This will:
- Load the data from `data/fed_state_raw.csv`
- Extract 184 features
- Generate analysis reports
- Save results to `results/` directory

## 📊 Step 4: View Results

Check the generated files:

```bash
ls results/basic_analysis/
```

You'll find:
- `metrics.csv` - All 184 extracted features
- `report.txt` - Statistical summary
- Various plots showing hormone dynamics

## 🔬 Step 5: Use in Your Own Analysis

```python
from src.feature_extraction import ComprehensiveFedStateAnalyzer

# Initialize
analyzer = ComprehensiveFedStateAnalyzer()

# Load your data
analyzer.load_csv_data('path/to/your/data.csv')

# Extract features
for animal_id in analyzer.data['animal_id'].unique():
    metrics = analyzer.calculate_comprehensive_metrics(animal_id)
    print(f"Animal {animal_id}: {len(metrics)} metric categories")

# Export for machine learning
df = analyzer.export_metrics_to_csv('my_features.csv')
```

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [data/README.md](data/README.md) for data format details
- Explore [src/example_usage.py](src/example_usage.py) for more examples
- Review the [NeurIPS paper](paper/) for methodology details

## 🆘 Need Help?

**Common Issues:**

1. **Import Error**: Make sure you're in the repository root directory
2. **File Not Found**: Check that data files are in `data/` folder
3. **Module Not Found**: Run `pip install -r requirements.txt`

**Still stuck?** Open an issue on GitHub or contact:
- Hesam Abouali: hesam.abouali@uwaterloo.ca

## 🎯 What's Next?

Once you're comfortable with the basics:

1. **Modify for your data**: Adapt the analysis pipeline
2. **Add new features**: Extend the feature extraction
3. **Train ML models**: Use the extracted features for classification
4. **Visualize results**: Create custom plots
5. **Contribute**: Submit pull requests with improvements!

---

**Happy analyzing! 🚀**
