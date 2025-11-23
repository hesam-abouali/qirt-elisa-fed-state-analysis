"""
Example Usage of QIRT-ELISA Fed-State Analysis Pipeline

This script demonstrates how to use the ComprehensiveFedStateAnalyzer
to extract features and analyze QIRT-ELISA time-series data.
"""

import sys
sys.path.append('.')

from src.feature_extraction import ComprehensiveFedStateAnalyzer


def example_basic_analysis():
    """
    Example 1: Basic Analysis Pipeline
    
    This example shows how to:
    1. Load data from CSV
    2. Run complete analysis
    3. Export metrics to CSV
    """
    print("="*80)
    print("Example 1: Basic Analysis Pipeline")
    print("="*80)
    
    # Initialize the analyzer
    analyzer = ComprehensiveFedStateAnalyzer()
    
    # Load and analyze data
    results = analyzer.run_complete_analysis(
        csv_path='data/fed_state_raw.csv',
        save_plots_dir='results/basic_analysis',
        save_report_path='results/basic_analysis/report.txt'
    )
    
    # Export comprehensive metrics
    analyzer.export_metrics_to_csv('results/basic_analysis/metrics.csv')
    
    print(f"\n✓ Analysis complete for {len(results)} animals")
    print(f"✓ Results saved to: results/basic_analysis/")


def example_custom_analysis():
    """
    Example 2: Custom Analysis with Individual Metric Access
    
    This example shows how to:
    1. Load data
    2. Access specific metric categories
    3. Extract individual features
    """
    print("\n" + "="*80)
    print("Example 2: Custom Analysis - Accessing Individual Metrics")
    print("="*80)
    
    # Initialize and load data
    analyzer = ComprehensiveFedStateAnalyzer()
    data = analyzer.load_csv_data('data/fed_state_raw.csv')
    
    # Calculate metrics for all animals
    all_metrics = {}
    for animal_id in data['animal_id'].unique():
        metrics = analyzer.calculate_comprehensive_metrics(animal_id)
        all_metrics[animal_id] = metrics
    
    # Example: Access specific metrics
    animal_id = data['animal_id'].iloc[0]
    metrics = all_metrics[animal_id]
    
    print(f"\nExample metrics for animal {animal_id}:")
    print(f"  Rat type: {metrics['basic_info']['rat_type']}")
    print(f"  Glucose (start): {metrics['basic_info']['glucose_start_mM']:.2f} mM")
    print(f"  Glucose (end): {metrics['basic_info']['glucose_end_mM']:.2f} mM")
    
    # Access insulin metrics
    insulin_metrics = metrics['basic_device_measurements']['insulin']
    print(f"\nInsulin Basic Metrics:")
    print(f"  Mean: {insulin_metrics['mean_raw_signal']:.3f} a.u.")
    print(f"  CV: {insulin_metrics['coefficient_of_variation']:.2f}%")
    print(f"  Range: {insulin_metrics['range_raw_signal']:.3f} a.u.")
    
    # Access glucagon kinetics
    glucagon_kinetics = metrics['peak_kinetics']['glucagon']
    print(f"\nGlucagon Kinetics:")
    print(f"  Peak count: {glucagon_kinetics['peak_count']}")
    print(f"  Mean rise rate: {glucagon_kinetics['mean_rise_rate']:.4f} a.u./min")
    print(f"  Mean decline rate: {glucagon_kinetics['mean_decline_rate']:.4f} a.u./min")


def example_group_comparison():
    """
    Example 3: Group-Level Statistical Comparison
    
    This example shows how to compare metrics across different
    rat types (healthy, pre-diabetic, diabetic).
    """
    print("\n" + "="*80)
    print("Example 3: Group-Level Comparison")
    print("="*80)
    
    # Initialize and run analysis
    analyzer = ComprehensiveFedStateAnalyzer()
    analyzer.load_csv_data('data/fed_state_raw.csv')
    
    # Calculate metrics for all animals
    for animal_id in analyzer.data['animal_id'].unique():
        analyzer.calculate_comprehensive_metrics(animal_id)
    
    # Generate group-level statistics
    group_stats = analyzer.generate_group_level_statistics()
    
    print("\nGroup-level comparison (glucagon mean rise rate):")
    if 'glucagon_kinetics_mean_rise_rate' in group_stats:
        metric_stats = group_stats['glucagon_kinetics_mean_rise_rate']
        for group in ['healthy', 'pre-diabetic', 'diabetic']:
            if group in metric_stats:
                stats = metric_stats[group]
                print(f"  {group.capitalize()}:")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Std: {stats['std']:.4f}")
                print(f"    N: {stats['n']}")


def example_export_for_ml():
    """
    Example 4: Export Data for Machine Learning
    
    This example shows how to prepare data for machine learning
    classification tasks.
    """
    print("\n" + "="*80)
    print("Example 4: Export for Machine Learning")
    print("="*80)
    
    # Initialize and analyze
    analyzer = ComprehensiveFedStateAnalyzer()
    analyzer.run_complete_analysis(
        csv_path='data/fed_state_raw.csv',
        save_plots_dir='results/ml_ready'
    )
    
    # Export comprehensive metrics for ML
    ml_data = analyzer.export_metrics_to_csv('results/ml_ready/ml_features.csv')
    
    print(f"\n✓ ML-ready dataset created:")
    print(f"  Shape: {ml_data.shape}")
    print(f"  Animals: {len(ml_data)}")
    print(f"  Features: {len(ml_data.columns)}")
    print(f"  Groups: {ml_data['rat_type'].value_counts().to_dict()}")
    print(f"\n✓ Data saved to: results/ml_ready/ml_features.csv")
    print(f"\nColumn categories:")
    print(f"  - Basic info: animal_id, rat_type, glucose metrics")
    print(f"  - Device measurements: *_basic_*")
    print(f"  - Glucose-adjusted: *_glucose_*")
    print(f"  - Peak dynamics: *_peak_*")
    print(f"  - Rate analysis: *_rate_*")
    print(f"  - AUC metrics: *_auc_*")
    print(f"  - Oscillations: *_osc_*")
    print(f"  - Correlations: corr_*")
    print(f"  - ELISA validation: *_elisa_*")
    print(f"  - Pancreatic function: *_cell_*, function_*")
    print(f"  - Peak kinetics: *_kinetics_*")


def main():
    """
    Main function to run all examples.
    
    Uncomment the examples you want to run.
    """
    print("\n" + "="*80)
    print("QIRT-ELISA Fed-State Analysis - Usage Examples")
    print("="*80)
    
    # Run examples (comment out ones you don't need)
    example_basic_analysis()
    # example_custom_analysis()
    # example_group_comparison()
    # example_export_for_ml()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
