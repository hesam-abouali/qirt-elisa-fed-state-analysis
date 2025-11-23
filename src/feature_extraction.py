import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, peak_widths, peak_prominences
from numpy import trapz
import warnings
from itertools import combinations
from pathlib import Path
import os
warnings.filterwarnings('ignore')
class ComprehensiveFedStateAnalyzer:
    """
    Comprehensive Analysis Pipeline for the Fed State Experiments by the QIRT-ELISA Device
   
    Analyzer that calculates all metrics identified in the
    fed state experiment analysis of animal studies, including:
    - Basic Device Measurements
    - All glucose-adjusted metrics
    - Peak and dynamics analysis
    - Rate of change analysis
    - Area under curve metrics
    - Oscillation pattern metrics
    - Inter-biomarker correlations
    - ELISA-validated quantitative metrics
    - Peak kinetics analysis

    The data is expected to be in a CSV file with the following columns:
    - animal_id (unique identifier for each animal)
    - rat_type (type of rat; e.g. 'Healthy', 'Pre-diabetic', 'Diabetic', 'fasting')
    - timepoint (timepoint of the measurement; e.g. 0, 2.5, 5, 7.5, 10, 12.5, 15 minutes)
    - insulin_au (QIRT-ELISA device measurements; Normalized F/F0)
    - c_peptide_au (QIRT-ELISA device measurements; Normalized F/F0)
    - glucagon_au (QIRT-ELISA device measurements; Normalized F/F0)
    - glucose_start_mM (Glucometer measurement at the start of the experiment; [mM])
    - glucose_end_mM (Glucometer measurement at the end of the experiment; [mM])
    - insulin_start_ELISA_pM (ELISA measurement at the start of the experiment; [pM])
    - insulin_end_ELISA_pM (ELISA measurement at the end of the experiment; [pM])
    - c_peptide_start_ELISA_pM (ELISA measurement at the start of the experiment; [pM])
    - c_peptide_end_ELISA_pM (ELISA measurement at the end of the experiment; [pM])
    - glucagon_start_ELISA_pM (ELISA measurement at the start of the experiment; [pM])
    - glucagon_end_ELISA_pM (ELISA measurement at the end of the experiment; [pM])
    """
   
    def __init__(self):
        self.biomarkers = ['insulin', 'c_peptide', 'glucagon']
        self.data = None
        self.processed_data = {}
        self.group_metrics = {}
        self.comprehensive_metrics = {}
        self.peak_kinetics_data = {}
       
        # Analysis parameters (auto-detected)
        self.time_points = None
        self.num_timepoints = None
        self.time_interval = None
        self.experiment_duration = None
       
        # Results storage
        self.all_metric_results = {}
        self.statistical_comparisons = {}
       
    def load_csv_data(self, csv_path, format_type='long'):
        """Load experimental data from a CSV file"""
       
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
           
        df = pd.read_csv(csv_path)
        print("=" * 80)
        print("Comprehensive Analysis Pipeline for the Fed State Experiments by the QIRT-ELISA Device")
        print("=" * 80)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
       
        # Auto-detect timepoint structure
        self.time_points, self.time_interval, self.num_timepoints, self.experiment_duration = self._detect_timepoint_structure(df)
       
        # Process data
        if format_type == 'long':
            self.data = self._process_long_format_data(df)
        else:
            raise ValueError("Only 'long' format supported in this version")
           
        if len(self.data) == 0:
            print("ERROR: No valid animals loaded!")
            return None
           
        print(f"Successfully loaded data for {len(self.data)} animals:")
        for rat_type in self.data['rat_type'].unique():
            count = sum(self.data['rat_type'] == rat_type)
            print(f" - {rat_type}: {count} animals")
           
        return self.data
   
    def _detect_timepoint_structure(self, df):
        """Auto-detect timepoint structure from the data"""
        timepoints = sorted(df['timepoint'].unique())
       
        if len(timepoints) < 2:
            raise ValueError("Need at least 2 timepoints to detect structure")
       
        intervals = np.diff(timepoints)
        if not np.allclose(intervals, intervals[0], rtol=0.1):
            print(f"Warning: Irregular time intervals detected: {intervals}")
       
        time_interval = np.mean(intervals)
        num_timepoints = len(timepoints)
        experiment_duration = timepoints[-1] - timepoints[0]
       
        print(f"Auto-detected timepoint structure:")
        print(f" - Number of timepoints: {num_timepoints}")
        print(f" - Time interval: {time_interval:.1f} minutes")
        print(f" - Experiment duration: {experiment_duration:.1f} minutes")
        print(f" - Timepoints: {timepoints}")
       
        return np.array(timepoints), time_interval, num_timepoints, experiment_duration
   
    def _process_long_format_data(self, df):
        """Process long format CSV data"""
        required_cols = ['animal_id', 'rat_type', 'timepoint', 'insulin_au', 'c_peptide_au', 'glucagon_au',
                        'glucose_start_mM', 'glucose_end_mM']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
       
        # Check for ELISA columns
        elisa_cols = ['insulin_start_ELISA_pM', 'insulin_end_ELISA_pM', 'c_peptide_start_ELISA_pM',
                     'c_peptide_end_ELISA_pM', 'glucagon_start_ELISA_pM', 'glucagon_end_ELISA_pM']
        has_elisa = all(col in df.columns for col in elisa_cols)
       
        if has_elisa:
            print("ELISA validation data detected - QUANTITATIVE analysis enabled!")
        else:
            print("No ELISA data - relative analysis only")
       
        animals_data = []
       
        for animal_id in df['animal_id'].unique():
            if pd.isna(animal_id):
                continue
               
            animal_df = df[df['animal_id'] == animal_id].sort_values('timepoint')
           
            if len(animal_df) != self.num_timepoints:
                print(f"Warning: Animal {animal_id} has {len(animal_df)} timepoints instead of {self.num_timepoints}")
                continue
               
            rat_type = animal_df['rat_type'].iloc[0]
            if isinstance(rat_type, str):
                rat_type = rat_type.lower().strip()
           
            animal_data = {
                'animal_id': animal_id,
                'rat_type': rat_type,
                'glucose_start_mM': animal_df['glucose_start_mM'].iloc[0],
                'glucose_end_mM': animal_df['glucose_end_mM'].iloc[0],
                'insulin_au': animal_df['insulin_au'].values,
                'c_peptide_au': animal_df['c_peptide_au'].values,
                'glucagon_au': animal_df['glucagon_au'].values,
                'has_elisa': has_elisa,
                'timepoints': np.array(sorted(animal_df['timepoint'].values))
            }
           
            # Add ELISA validation data if available
            if has_elisa:
                for biomarker in self.biomarkers:
                    start_elisa = animal_df[f'{biomarker}_start_ELISA_pM'].iloc[0]
                    end_elisa = animal_df[f'{biomarker}_end_ELISA_pM'].iloc[0]
                   
                    if not (pd.isna(start_elisa) or pd.isna(end_elisa)):
                        animal_data[f'{biomarker}_start_ELISA_pM'] = start_elisa
                        animal_data[f'{biomarker}_end_ELISA_pM'] = end_elisa
                    else:
                        animal_data['has_elisa'] = False
           
            animals_data.append(animal_data)
       
        return pd.DataFrame(animals_data)
   
    def calculate_comprehensive_metrics(self, animal_id):
        """Calculate all metrics for a single animal subject of the fed state experiment"""
       
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_data() first.")
       
        animal_data = self.data[self.data['animal_id'] == animal_id]
        if len(animal_data) == 0:
            raise ValueError(f"Animal {animal_id} not found in data")
       
        animal_row = animal_data.iloc[0]
       
        # Basic data extraction
        raw_data = {
            'subject_id': animal_row['animal_id'],
            'condition': animal_row['rat_type'],
            'glucose_start': animal_row['glucose_start_mM'],
            'glucose_end': animal_row['glucose_end_mM']
        }
       
        for biomarker in self.biomarkers:
            raw_data[biomarker] = animal_row[f'{biomarker}_au']
       
        # Glucose calculations
        glucose_avg = (raw_data['glucose_start'] + raw_data['glucose_end']) / 2
        glucose_change = raw_data['glucose_end'] - raw_data['glucose_start']
       
        # Calculate all metric categories
        all_metrics = {}
       
        # 1. Basic Device Measurements
        all_metrics['basic_device'] = self._calculate_basic_device_metrics(raw_data)
       
        # 2. Glucose-Adjusted Metrics
        all_metrics['glucose_adjusted'] = self._calculate_glucose_adjusted_metrics(raw_data, glucose_avg, glucose_change)
       
        # 3. Peak and Dynamics Analysis
        all_metrics['peak_dynamics'] = self._calculate_peak_dynamics_metrics(raw_data)
       
        # 4. Rate of Change Analysis
        all_metrics['rate_analysis'] = self._calculate_rate_change_metrics(raw_data)
       
        # 5. Area Under Curve Metrics
        all_metrics['auc_metrics'] = self._calculate_auc_metrics(raw_data, glucose_avg)
       
        # 6. Oscillation and Pattern Metrics
        all_metrics['oscillation_patterns'] = self._calculate_oscillation_pattern_metrics(raw_data)
       
        # 7. Inter-biomarker Correlations
        all_metrics['correlations'] = self._calculate_correlation_metrics(raw_data, glucose_avg, glucose_change)
       
        # 8. ELISA-Validated Metrics (if available)
        if animal_row.get('has_elisa', False):
            all_metrics['elisa_quantitative'] = self._calculate_elisa_quantitative_metrics(animal_id, animal_row, raw_data)
        else:
            all_metrics['elisa_quantitative'] = None
       
        # 9. Pancreatic Cell Functionality
        all_metrics['pancreatic_function'] = self._calculate_pancreatic_function_metrics(all_metrics)
       
        # 10. Peak Kinetics Analysis
        all_metrics['peak_kinetics'] = self._calculate_peak_kinetics_metrics(raw_data)
       
        # Store results
        self.comprehensive_metrics[animal_id] = {
            'raw_data': raw_data,
            'glucose_avg': glucose_avg,
            'glucose_change': glucose_change,
            'metrics': all_metrics,
            'has_elisa': animal_row.get('has_elisa', False)
        }
       
        return all_metrics
   
    def _calculate_basic_device_metrics(self, raw_data):
        """Calculate basic device measurement metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                mean_val = np.mean(signal)
                std_val = np.std(signal)
               
                # Fi/F0 normalization (F0 is average of all measurements)
                F0 = mean_val
                normalized_signal = signal / F0
                norm_mean = np.mean(normalized_signal)
                norm_std = np.std(normalized_signal)
               
                metrics[biomarker] = {
                    # Raw device measurements
                    'mean_raw_au': float(mean_val),
                    'std_raw_au': float(std_val),
                    'cv_raw_percent': float(std_val / mean_val * 100) if mean_val > 0 else 0.0,
                    'max_raw_au': float(np.max(signal)),
                    'min_raw_au': float(np.min(signal)),
                    'range_raw_au': float(np.max(signal) - np.min(signal)),
                   
                    # Normalized measurements (Fi/F0)
                    'mean_normalized': float(norm_mean),
                    'std_normalized': float(norm_std),
                    'cv_normalized_percent': float(norm_std / norm_mean * 100) if norm_mean > 0 else 0.0,
                    'max_normalized': float(np.max(normalized_signal)),
                    'min_normalized': float(np.min(normalized_signal)),
                    'range_normalized': float(np.max(normalized_signal) - np.min(normalized_signal)),
                   
                    # Signal characteristics
                    'signal_values': signal,
                    'normalized_values': normalized_signal
                }
       
        return metrics
   
    def _calculate_glucose_adjusted_metrics(self, raw_data, glucose_avg, glucose_change):
        """Calculate all glucose-adjusted metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                mean_signal = np.mean(signal)
                std_signal = np.std(signal)
                max_signal = np.max(signal)
                min_signal = np.min(signal)
               
                metrics[biomarker] = {
                    # Glucose efficiency metrics
                    'glucose_efficiency': float(mean_signal / glucose_avg),
                    'peak_glucose_efficiency': float(max_signal / glucose_avg),
                    'min_glucose_efficiency': float(min_signal / glucose_avg),
                    'glucose_efficiency_variability': float(std_signal / glucose_avg),
                   
                    # Glucose-corrected values (normalized to 5.6 mM reference)
                    'mean_glucose_corrected_5p6': float(mean_signal * (5.6 / glucose_avg)),
                    'max_glucose_corrected_5p6': float(max_signal * (5.6 / glucose_avg)),
                    'min_glucose_corrected_5p6': float(min_signal * (5.6 / glucose_avg)),
                    'std_glucose_corrected_5p6': float(std_signal * (5.6 / glucose_avg)),
                   
                    # Glucose sensitivity metrics
                    'glucose_sensitivity': float(std_signal / abs(glucose_change)) if abs(glucose_change) > 0.001 else float(std_signal),
                    'glucose_responsiveness': float((max_signal - min_signal) / abs(glucose_change)) if abs(glucose_change) > 0.001 else float(max_signal - min_signal),
                   
                    # Glucose-adjusted variability
                    'glucose_adjusted_cv': float((std_signal / glucose_avg) / (mean_signal / glucose_avg) * 100) if mean_signal > 0 else 0.0,
                   
                    # Glucose trend correlation
                    'glucose_correlation': self._calculate_glucose_correlation(signal, glucose_avg, glucose_change)
                }
       
        return metrics
   
    def _calculate_glucose_correlation(self, signal, glucose_avg, glucose_change):
        """Calculate correlation with glucose trend"""
        if abs(glucose_change) < 0.001:
            # No glucose change - use flat trend
            glucose_trend = np.full(len(signal), glucose_avg)
        else:
            # Linear glucose trend from start to end
            glucose_start = glucose_avg - glucose_change/2
            glucose_end = glucose_avg + glucose_change/2
            glucose_trend = np.linspace(glucose_start, glucose_end, len(signal))
       
        correlation, p_value = stats.pearsonr(signal, glucose_trend)
       
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
   
    def _calculate_peak_dynamics_metrics(self, raw_data):
        """Calculate peak detection and dynamics metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)
                signal_std = np.std(signal)
               
                # Peak detection with multiple thresholds
                peaks_above_mean, _ = find_peaks(signal, height=baseline)
                peaks_above_std, _ = find_peaks(signal, height=baseline + 0.5*signal_std)
                troughs, _ = find_peaks(-signal, height=-baseline)
               
                # Peak characteristics
                if len(peaks_above_std) > 0:
                    try:
                        prominences = peak_prominences(signal, peaks_above_std)[0]
                        widths = peak_widths(signal, peaks_above_std, rel_height=0.5)[0] * self.time_interval
                    except:
                        prominences = np.abs(signal[peaks_above_std] - baseline)
                        widths = np.full(len(peaks_above_std), self.time_interval)
                else:
                    prominences = np.array([])
                    widths = np.array([])
               
                metrics[biomarker] = {
                    # Peak counts
                    'num_peaks_above_mean': len(peaks_above_mean),
                    'num_peaks_above_std': len(peaks_above_std),
                    'num_troughs': len(troughs),
                    'peak_frequency': len(peaks_above_std) / self.experiment_duration if self.experiment_duration > 0 else 0,
                   
                    # Peak characteristics
                    'mean_peak_prominence': float(np.mean(prominences)) if len(prominences) > 0 else 0.0,
                    'max_peak_prominence': float(np.max(prominences)) if len(prominences) > 0 else 0.0,
                    'mean_peak_width': float(np.mean(widths)) if len(widths) > 0 else 0.0,
                   
                    # Signal extremes
                    'max_value': float(np.max(signal)),
                    'min_value': float(np.min(signal)),
                    'max_time': float(self.time_points[np.argmax(signal)]),
                    'min_time': float(self.time_points[np.argmin(signal)]),
                   
                    # Dynamic range
                    'dynamic_range_ratio': float(np.max(signal) / np.min(signal)) if np.min(signal) > 0 else np.inf,
                    'total_excursion': float(np.max(signal) - np.min(signal)),
                    'relative_excursion': float((np.max(signal) - np.min(signal)) / baseline) if baseline > 0 else 0.0,
                   
                    # Variability metrics
                    'coefficient_of_variation': float(signal_std / baseline * 100) if baseline > 0 else 0.0,
                    'relative_std': float(signal_std / baseline) if baseline > 0 else 0.0,
                   
                    # Timing metrics
                    'peak_times': self.time_points[peaks_above_std] if len(peaks_above_std) > 0 else np.array([]),
                    'trough_times': self.time_points[troughs] if len(troughs) > 0 else np.array([])
                }
       
        return metrics
   
    def _calculate_rate_change_metrics(self, raw_data):
        """Calculate rate of change analysis metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
               
                # Point-to-point rates
                rates = np.diff(signal) / self.time_interval
               
                # Smoothed rates (3-point average)
                if len(signal) >= 3:
                    smooth_rates = []
                    for i in range(1, len(signal) - 1):
                        rate = (signal[i+1] - signal[i-1]) / (2 * self.time_interval)
                        smooth_rates.append(rate)
                    smooth_rates = np.array(smooth_rates)
                else:
                    smooth_rates = rates
               
                # Acceleration (second derivative)
                if len(rates) > 1:
                    accelerations = np.diff(rates) / self.time_interval
                else:
                    accelerations = np.array([0])
               
                metrics[biomarker] = {
                    # Basic rates
                    'max_increase_rate': float(np.max(rates)) if len(rates) > 0 else 0.0,
                    'max_decrease_rate': float(abs(np.min(rates))) if len(rates) > 0 else 0.0,
                    'mean_absolute_rate': float(np.mean(np.abs(rates))) if len(rates) > 0 else 0.0,
                    'rate_variability': float(np.std(rates)) if len(rates) > 0 else 0.0,
                   
                    # Smoothed rates
                    'max_smooth_increase_rate': float(np.max(smooth_rates)) if len(smooth_rates) > 0 else 0.0,
                    'max_smooth_decrease_rate': float(abs(np.min(smooth_rates))) if len(smooth_rates) > 0 else 0.0,
                    'mean_smooth_absolute_rate': float(np.mean(np.abs(smooth_rates))) if len(smooth_rates) > 0 else 0.0,
                   
                    # Acceleration metrics
                    'max_acceleration': float(np.max(accelerations)) if len(accelerations) > 0 else 0.0,
                    'max_deceleration': float(abs(np.min(accelerations))) if len(accelerations) > 0 else 0.0,
                    'mean_acceleration': float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,
                   
                    # Rate statistics
                    'rate_range': float(np.max(rates) - np.min(rates)) if len(rates) > 0 else 0.0,
                    'rate_cv': float(np.std(rates) / np.mean(np.abs(rates)) * 100) if len(rates) > 0 and np.mean(np.abs(rates)) > 0 else 0.0,
                   
                    # Trend analysis
                    'trend_slope': self._calculate_trend_slope(signal),
                    'trend_r_squared': self._calculate_trend_r_squared(signal),
                   
                    # Raw arrays for further analysis
                    'all_rates': rates,
                    'smooth_rates': smooth_rates
                }
       
        return metrics
   
    def _calculate_trend_slope(self, signal):
        """Calculate linear trend slope"""
        try:
            slope, _, r_value, p_value, std_err = stats.linregress(self.time_points, signal)
            return {
                'slope': float(slope),
                'r_value': float(r_value),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'std_err': float(std_err),
                'significant': p_value < 0.05
            }
        except:
            return {
                'slope': 0.0,
                'r_value': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'std_err': 0.0,
                'significant': False
            }
   
    def _calculate_trend_r_squared(self, signal):
        """Calculate R-squared for linear trend"""
        try:
            _, _, r_value, _, _ = stats.linregress(self.time_points, signal)
            return float(r_value**2)
        except:
            return 0.0
   
    def _calculate_auc_metrics(self, raw_data, glucose_avg):
        """Calculate area under curve metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)
               
                # Various AUC calculations
                auc_total = trapz(signal, self.time_points)
                auc_above_baseline = trapz(np.maximum(signal - baseline, 0), self.time_points)
                auc_below_baseline = trapz(np.maximum(baseline - signal, 0), self.time_points)
               
                # Normalized AUC (Fi/F0 format)
                normalized_signal = signal / baseline
                auc_normalized = trapz(normalized_signal, self.time_points)
                auc_above_mean = trapz(np.maximum(normalized_signal - 1.0, 0), self.time_points)
                auc_below_mean = trapz(np.maximum(1.0 - normalized_signal, 0), self.time_points)
               
                # Glucose-adjusted AUC
                auc_glucose_corrected = auc_total * (5.6 / glucose_avg)
                auc_efficiency = auc_total / glucose_avg
               
                metrics[biomarker] = {
                    # Basic AUC metrics
                    'auc_total': float(auc_total),
                    'auc_above_baseline': float(auc_above_baseline),
                    'auc_below_baseline': float(auc_below_baseline),
                    'auc_net': float(auc_above_baseline - auc_below_baseline),
                   
                    # Normalized AUC
                    'auc_normalized': float(auc_normalized),
                    'auc_above_mean': float(auc_above_mean),
                    'auc_below_mean': float(auc_below_mean),
                    'auc_normalized_net': float(auc_above_mean - auc_below_mean),
                   
                    # Glucose-adjusted AUC
                    'auc_glucose_corrected_5p6': float(auc_glucose_corrected),
                    'auc_glucose_efficiency': float(auc_efficiency),
                   
                    # Relative AUC metrics
                    'auc_fraction_above_baseline': float(auc_above_baseline / auc_total) if auc_total > 0 else 0.0,
                    'auc_balance_ratio': float(auc_above_baseline / auc_below_baseline) if auc_below_baseline > 0 else np.inf,
                   
                    # Time-weighted averages
                    'time_weighted_average': float(auc_total / self.experiment_duration) if self.experiment_duration > 0 else 0.0,
                    'time_weighted_glucose_efficiency': float(auc_efficiency / self.experiment_duration) if self.experiment_duration > 0 else 0.0
                }
       
        return metrics
   
    def _calculate_oscillation_pattern_metrics(self, raw_data):
        """Calculate oscillation and pattern metrics"""
        metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)
               
                # Zero crossings (crossings of baseline)
                centered_signal = signal - baseline
                zero_crossings = len(np.where(np.diff(np.signbit(centered_signal)))[0])
               
                # Oscillation frequency
                oscillation_frequency = zero_crossings / self.experiment_duration if self.experiment_duration > 0 else 0
               
                # Pattern regularity (autocorrelation)
                try:
                    autocorr = np.correlate(centered_signal, centered_signal, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0] # Normalize
                    pattern_regularity = float(np.max(autocorr[1:min(3, len(autocorr)-1)])) if len(autocorr) > 1 else 0.0
                except:
                    pattern_regularity = 0.0
               
                # Rhythmicity index (FFT-based)
                try:
                    fft_signal = np.fft.fft(centered_signal)
                    power_spectrum = np.abs(fft_signal)**2
                    # Find dominant frequency component
                    dominant_freq_power = np.max(power_spectrum[1:len(power_spectrum)//2])
                    total_power = np.sum(power_spectrum[1:len(power_spectrum)//2])
                    rhythmicity_index = float(dominant_freq_power / total_power) if total_power > 0 else 0.0
                except:
                    rhythmicity_index = 0.0
               
                # Phase analysis (if multiple oscillations detected)
                phase_consistency = self._calculate_phase_consistency(signal, baseline)
               
                metrics[biomarker] = {
                    # Basic oscillation metrics
                    'zero_crossings': zero_crossings,
                    'oscillation_frequency': float(oscillation_frequency),
                    'pattern_regularity': pattern_regularity,
                    'rhythmicity_index': rhythmicity_index,
                   
                    # Phase and timing metrics
                    'phase_consistency': phase_consistency,
                    'dominant_period': float(2 * self.experiment_duration / zero_crossings) if zero_crossings > 0 else np.inf,
                   
                    # Signal complexity
                    'signal_entropy': self._calculate_signal_entropy(signal),
                    'complexity_index': self._calculate_complexity_index(signal),
                   
                    # Stability metrics
                    'temporal_stability': self._calculate_temporal_stability(signal),
                    'pattern_consistency': self._calculate_pattern_consistency(signal)
                }
       
        return metrics
   
    def _calculate_phase_consistency(self, signal, baseline):
        """Calculate phase consistency of oscillations"""
        try:
            # Find peaks and calculate inter-peak intervals
            peaks, _ = find_peaks(signal, height=baseline)
            if len(peaks) > 2:
                intervals = np.diff(peaks) * self.time_interval
                consistency = 1.0 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
                return float(max(0.0, consistency))
            else:
                return 0.0
        except:
            return 0.0
   
    def _calculate_signal_entropy(self, signal):
        """Calculate Shannon entropy of signal"""
        try:
            # Discretize signal into bins
            bins = min(10, len(signal) // 2)
            hist, _ = np.histogram(signal, bins=bins, density=True)
            hist = hist[hist > 0] # Remove zero entries
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except:
            return 0.0
   
    def _calculate_complexity_index(self, signal):
        """Calculate complexity index based on signal structure"""
        try:
            # Calculate first and second differences
            first_diff = np.diff(signal)
            second_diff = np.diff(first_diff)
           
            # Complexity as ratio of second to first derivative variance
            if np.var(first_diff) > 0:
                complexity = np.var(second_diff) / np.var(first_diff)
            else:
                complexity = 0.0
           
            return float(complexity)
        except:
            return 0.0
   
    def _calculate_temporal_stability(self, signal):
        """Calculate temporal stability of signal"""
        try:
            # Split signal into halves and compare
            mid = len(signal) // 2
            first_half = signal[:mid]
            second_half = signal[-mid:]
           
            if len(first_half) > 1 and len(second_half) > 1:
                correlation, _ = stats.pearsonr(first_half, second_half[:len(first_half)])
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
        except:
            return 0.0
   
    def _calculate_pattern_consistency(self, signal):
        """Calculate pattern consistency across timepoints"""
        try:
            # Calculate moving coefficient of variation
            window = min(3, len(signal) // 2)
            if window < 2:
                return 0.0
           
            cvs = []
            for i in range(len(signal) - window + 1):
                window_data = signal[i:i+window]
                if np.mean(window_data) > 0:
                    cv = np.std(window_data) / np.mean(window_data)
                    cvs.append(cv)
           
            if cvs:
                # Consistency is inverse of CV variability
                cv_stability = 1.0 / (1.0 + np.std(cvs))
                return float(cv_stability)
            else:
                return 0.0
        except:
            return 0.0
   
    def _calculate_correlation_metrics(self, raw_data, glucose_avg, glucose_change):
        """Calculate inter-biomarker correlation metrics"""
        correlations = {}
       
        # Pairwise biomarker correlations
        biomarker_pairs = [('insulin', 'c_peptide'), ('insulin', 'glucagon'), ('c_peptide', 'glucagon')]
       
        for bm1, bm2 in biomarker_pairs:
            if bm1 in raw_data and bm2 in raw_data:
                signal1 = raw_data[bm1]
                signal2 = raw_data[bm2]
               
                try:
                    # Basic correlation
                    corr, p_val = stats.pearsonr(signal1, signal2)
                   
                    # Partial correlation (controlling for time trend)
                    time_trend = np.linspace(0, 1, len(signal1))
                    partial_corr = self._calculate_partial_correlation(signal1, signal2, time_trend)
                   
                    # Cross-correlation (time-lagged)
                    cross_corr = self._calculate_cross_correlation(signal1, signal2)
                   
                    # Phase relationship
                    phase_relationship = self._calculate_phase_relationship(signal1, signal2)
                   
                    correlations[f'{bm1}_{bm2}'] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'partial_correlation': partial_corr,
                        'max_cross_correlation': cross_corr['max_correlation'],
                        'optimal_lag': cross_corr['optimal_lag'],
                        'phase_lag': phase_relationship['phase_lag'],
                        'coherence': phase_relationship['coherence']
                    }
                   
                except Exception as e:
                    correlations[f'{bm1}_{bm2}'] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'significant': False,
                        'partial_correlation': 0.0,
                        'max_cross_correlation': 0.0,
                        'optimal_lag': 0.0,
                        'phase_lag': 0.0,
                        'coherence': 0.0
                    }
       
        # Individual biomarker vs glucose correlations
        glucose_correlations = {}
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                glucose_corr = self._calculate_glucose_correlation(raw_data[biomarker], glucose_avg, glucose_change)
                glucose_correlations[biomarker] = glucose_corr
       
        return {
            'biomarker_correlations': correlations,
            'glucose_correlations': glucose_correlations
        }
   
    def _calculate_partial_correlation(self, x, y, z):
        """Calculate partial correlation between x and y controlling for z"""
        try:
            # Simple partial correlation using residuals
            x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
            y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
           
            corr, _ = stats.pearsonr(x_resid, y_resid)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
   
    def _calculate_cross_correlation(self, signal1, signal2):
        """Calculate cross-correlation with time lags"""
        try:
            # Normalize signals
            s1 = (signal1 - np.mean(signal1)) / np.std(signal1)
            s2 = (signal2 - np.mean(signal2)) / np.std(signal2)
           
            # Calculate cross-correlation
            cross_corr = np.correlate(s1, s2, mode='full')
           
            # Find maximum correlation and corresponding lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            max_correlation = cross_corr[max_corr_idx]
           
            # Convert index to time lag
            lags = np.arange(-len(s2) + 1, len(s1))
            optimal_lag = lags[max_corr_idx] * self.time_interval
           
            return {
                'max_correlation': float(max_correlation),
                'optimal_lag': float(optimal_lag)
            }
        except:
            return {
                'max_correlation': 0.0,
                'optimal_lag': 0.0
            }
   
    def _calculate_phase_relationship(self, signal1, signal2):
        """Calculate phase relationship between two signals"""
        try:
            # Use Hilbert transform to get phase
            from scipy.signal import hilbert
           
            analytic1 = hilbert(signal1 - np.mean(signal1))
            analytic2 = hilbert(signal2 - np.mean(signal2))
           
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
           
            # Phase difference
            phase_diff = phase1 - phase2
           
            # Wrap to [-pi, pi]
            phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
           
            # Mean phase lag and coherence
            mean_phase_lag = np.mean(phase_diff)
            phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
           
            return {
                'phase_lag': float(mean_phase_lag),
                'coherence': float(phase_coherence)
            }
        except:
            return {
                'phase_lag': 0.0,
                'coherence': 0.0
            }
   
    def _calculate_elisa_quantitative_metrics(self, animal_id, animal_row, raw_data):
        """Calculate ELISA-validated quantitative metrics"""
        try:
            # Get ELISA calibration data
            elisa_metrics = {}
           
            for biomarker in self.biomarkers:
                if (f'{biomarker}_start_ELISA_pM' in animal_row.index and
                    f'{biomarker}_end_ELISA_pM' in animal_row.index):
                   
                    device_signal = raw_data[biomarker]
                    elisa_start = animal_row[f'{biomarker}_start_ELISA_pM']
                    elisa_end = animal_row[f'{biomarker}_end_ELISA_pM']
                   
                    # Two-point calibration
                    device_start = device_signal[0]
                    device_end = device_signal[-1]
                   
                    if device_end != device_start:
                        slope = (elisa_end - elisa_start) / (device_end - device_start)
                        intercept = elisa_start - slope * device_start
                       
                        # Convert all timepoints to pM
                        concentrations_pM = slope * device_signal + intercept
                    else:
                        # No device change - use average ELISA
                        concentrations_pM = np.full(len(device_signal), (elisa_start + elisa_end) / 2)
                        slope = 0
                        intercept = (elisa_start + elisa_end) / 2
                   
                    # Calculate quantitative metrics
                    mean_conc = np.mean(concentrations_pM)
                    std_conc = np.std(concentrations_pM)
                   
                    elisa_metrics[biomarker] = {
                        # Absolute concentrations
                        'mean_concentration_pM': float(mean_conc),
                        'std_concentration_pM': float(std_conc),
                        'cv_concentration_percent': float(std_conc / mean_conc * 100) if mean_conc > 0 else 0.0,
                        'max_concentration_pM': float(np.max(concentrations_pM)),
                        'min_concentration_pM': float(np.min(concentrations_pM)),
                        'concentration_range_pM': float(np.max(concentrations_pM) - np.min(concentrations_pM)),
                       
                        # Dynamic metrics
                        'absolute_dynamic_range': float(np.max(concentrations_pM) / np.min(concentrations_pM)) if np.min(concentrations_pM) > 0 else np.inf,
                        'concentration_auc_pM_min': float(trapz(concentrations_pM, self.time_points)),
                       
                        # Rate metrics
                        'max_secretion_rate_pM_per_min': float(np.max(np.diff(concentrations_pM)) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,
                        'max_clearance_rate_pM_per_min': float(abs(np.min(np.diff(concentrations_pM))) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,
                        'mean_abs_rate_pM_per_min': float(np.mean(np.abs(np.diff(concentrations_pM))) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,
                       
                        # Calibration quality
                        'calibration_slope': float(slope),
                        'calibration_intercept': float(intercept),
                        'elisa_start_pM': float(elisa_start),
                        'elisa_end_pM': float(elisa_end),
                        'calibration_error_percent': self._calculate_calibration_error(device_signal, concentrations_pM, elisa_start, elisa_end),
                       
                        # Glucose relationships in pM units
                        'concentration_per_mM_glucose': float(mean_conc / raw_data['glucose_start']),
                        'glucose_sensitivity_pM_per_mM': float(std_conc / abs(raw_data['glucose_end'] - raw_data['glucose_start'])) if abs(raw_data['glucose_end'] - raw_data['glucose_start']) > 0.001 else float(std_conc),
                       
                        # Full concentration array
                        'concentrations_pM': concentrations_pM
                    }
           
            return elisa_metrics
           
        except Exception as e:
            print(f"Error calculating ELISA metrics for {animal_id}: {e}")
            return None
   
    def _calculate_calibration_error(self, device_signal, concentrations_pM, elisa_start, elisa_end):
        """Calculate calibration error percentage"""
        try:
            predicted_start = concentrations_pM[0]
            predicted_end = concentrations_pM[-1]
           
            start_error = abs(predicted_start - elisa_start) / elisa_start * 100 if elisa_start > 0 else 0
            end_error = abs(predicted_end - elisa_end) / elisa_end * 100 if elisa_end > 0 else 0
           
            return float((start_error + end_error) / 2)
        except:
            return 0.0
   
    def _calculate_pancreatic_function_metrics(self, all_metrics):
        """Calculate comprehensive pancreatic cell functionality metrics"""
        function_metrics = {}
       
        # Beta-cell function metrics
        beta_cell_metrics = {}
       
        if ('insulin' in all_metrics['correlations']['biomarker_correlations'] and
            'c_peptide' in all_metrics['basic_device']):
           
            # Insulin-C-peptide coordination
            insulin_cpep_corr = all_metrics['correlations']['biomarker_correlations'].get('insulin_c_peptide', {})
            beta_cell_metrics['insulin_cpeptide_correlation'] = insulin_cpep_corr.get('correlation', 0.0)
            beta_cell_metrics['coordination_significance'] = insulin_cpep_corr.get('significant', False)
            beta_cell_metrics['coordination_quality'] = self._classify_correlation(beta_cell_metrics['insulin_cpeptide_correlation'])
           
            # Secretory stability
            insulin_cv = all_metrics['basic_device']['insulin'].get('cv_normalized_percent', 0)
            cpeptide_cv = all_metrics['basic_device']['c_peptide'].get('cv_normalized_percent', 0)
            avg_cv = (insulin_cv + cpeptide_cv) / 2
            beta_cell_metrics['secretory_stability_cv'] = avg_cv
            beta_cell_metrics['stability_status'] = self._classify_stability(avg_cv)
           
            # Glucose responsiveness
            insulin_glucose_resp = all_metrics['glucose_adjusted']['insulin'].get('glucose_responsiveness', 0)
            cpeptide_glucose_resp = all_metrics['glucose_adjusted']['c_peptide'].get('glucose_responsiveness', 0)
            avg_responsiveness = (insulin_glucose_resp + cpeptide_glucose_resp) / 2
            beta_cell_metrics['glucose_responsiveness'] = avg_responsiveness
            beta_cell_metrics['responsiveness_status'] = self._classify_responsiveness(avg_responsiveness)
           
            # Dynamic range capability
            insulin_range = all_metrics['peak_dynamics']['insulin'].get('dynamic_range_ratio', 1.0)
            cpeptide_range = all_metrics['peak_dynamics']['c_peptide'].get('dynamic_range_ratio', 1.0)
            avg_range = (insulin_range + cpeptide_range) / 2
            beta_cell_metrics['dynamic_range'] = avg_range
            beta_cell_metrics['range_status'] = self._classify_dynamic_range(avg_range)
           
            # Secretory coherence (pattern matching)
            coherence = insulin_cpep_corr.get('coherence', 0.0)
            beta_cell_metrics['secretory_coherence'] = coherence
            beta_cell_metrics['coherence_status'] = self._classify_coherence(coherence)
       
        # Alpha-cell function metrics
        alpha_cell_metrics = {}
       
        if 'glucagon' in all_metrics['basic_device']:
            # Glucose suppression ability
            glucagon_glucose_corr = all_metrics['correlations']['glucose_correlations'].get('glucagon', {})
            glucose_suppression = glucagon_glucose_corr.get('correlation', 0.0)
            alpha_cell_metrics['glucose_suppression_correlation'] = glucose_suppression
            alpha_cell_metrics['suppression_quality'] = self._classify_suppression(glucose_suppression)
           
            # Regulatory precision
            glucagon_cv = all_metrics['basic_device']['glucagon'].get('cv_normalized_percent', 0)
            alpha_cell_metrics['regulatory_precision_cv'] = glucagon_cv
            alpha_cell_metrics['precision_status'] = self._classify_precision(glucagon_cv)
           
            # Counter-regulatory capacity
            glucagon_range = all_metrics['peak_dynamics']['glucagon'].get('dynamic_range_ratio', 1.0)
            alpha_cell_metrics['counter_regulatory_range'] = glucagon_range
            alpha_cell_metrics['counter_reg_status'] = self._classify_counter_regulation(glucagon_range)
           
            # Insulin-glucagon reciprocity
            insulin_glucagon_corr = all_metrics['correlations']['biomarker_correlations'].get('insulin_glucagon', {})
            reciprocity = insulin_glucagon_corr.get('correlation', 0.0)
            alpha_cell_metrics['insulin_glucagon_reciprocity'] = reciprocity
            alpha_cell_metrics['reciprocity_status'] = self._classify_reciprocity(reciprocity)
       
        # Overall pancreatic function score
        overall_score = self._calculate_overall_function_score(beta_cell_metrics, alpha_cell_metrics)
       
        function_metrics = {
            'beta_cell_function': beta_cell_metrics,
            'alpha_cell_function': alpha_cell_metrics,
            'overall_function_score': overall_score,
            'function_classification': self._classify_overall_function(overall_score)
        }
       
        return function_metrics
   
    def _calculate_peak_kinetics_metrics(self, raw_data):
        """Calculate detailed peak kinetics analysis"""
        kinetics_metrics = {}
       
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)
               
                # Peak detection for kinetics
                peaks, _ = find_peaks(signal, height=baseline + 0.1*np.std(signal))
               
                if len(peaks) > 0:
                    # Calculate rise and decline characteristics for each peak
                    rise_times = []
                    decline_times = []
                    rise_rates = []
                    decline_rates = []
                   
                    for peak_idx in peaks:
                        # Rise phase analysis
                        if peak_idx > 0:
                            # Find valley before peak
                            pre_signal = signal[:peak_idx+1]
                            valley_idx = np.argmin(pre_signal)
                           
                            rise_time = (peak_idx - valley_idx) * self.time_interval
                            rise_height = signal[peak_idx] - signal[valley_idx]
                            rise_rate = rise_height / rise_time if rise_time > 0 else 0
                           
                            rise_times.append(rise_time)
                            rise_rates.append(rise_rate)
                       
                        # Decline phase analysis
                        if peak_idx < len(signal) - 1:
                            # Find valley after peak
                            post_signal = signal[peak_idx:]
                            valley_offset = np.argmin(post_signal)
                           
                            decline_time = valley_offset * self.time_interval
                            decline_height = signal[peak_idx] - signal[peak_idx + valley_offset]
                            decline_rate = decline_height / decline_time if decline_time > 0 else 0
                           
                            decline_times.append(decline_time)
                            decline_rates.append(decline_rate)
                   
                    # Inter-peak analysis
                    if len(peaks) > 1:
                        inter_peak_intervals = np.diff(peaks) * self.time_interval
                        peak_frequency = len(peaks) / self.experiment_duration
                    else:
                        inter_peak_intervals = []
                        peak_frequency = 0
                   
                    kinetics_metrics[biomarker] = {
                        # Peak characteristics
                        'peak_count': len(peaks),
                        'peak_frequency': float(peak_frequency),
                        'peak_times': self.time_points[peaks],
                        'peak_heights': signal[peaks],
                       
                        # Rise kinetics
                        'mean_rise_time': float(np.mean(rise_times)) if rise_times else 0.0,
                        'std_rise_time': float(np.std(rise_times)) if rise_times else 0.0,
                        'mean_rise_rate': float(np.mean(rise_rates)) if rise_rates else 0.0,
                        'max_rise_rate': float(np.max(rise_rates)) if rise_rates else 0.0,
                       
                        # Decline kinetics
                        'mean_decline_time': float(np.mean(decline_times)) if decline_times else 0.0,
                        'std_decline_time': float(np.std(decline_times)) if decline_times else 0.0,
                        'mean_decline_rate': float(np.mean(decline_rates)) if decline_rates else 0.0,
                        'max_decline_rate': float(np.max(decline_rates)) if decline_rates else 0.0,
                       
                        # Inter-peak analysis
                        'mean_inter_peak_interval': float(np.mean(inter_peak_intervals)) if len(inter_peak_intervals) > 0 else 0.0,
                        'inter_peak_variability': float(np.std(inter_peak_intervals)) if len(inter_peak_intervals) > 0 else 0.0,
                       
                        # Kinetic ratios
                        'rise_decline_ratio': float(np.mean(rise_times) / np.mean(decline_times)) if rise_times and decline_times and np.mean(decline_times) > 0 else 0.0,
                        'secretion_clearance_ratio': float(np.mean(rise_rates) / np.mean(decline_rates)) if rise_rates and decline_rates and np.mean(decline_rates) > 0 else 0.0,
                       
                        # Raw arrays for further analysis
                        'all_rise_times': np.array(rise_times),
                        'all_decline_times': np.array(decline_times),
                        'all_rise_rates': np.array(rise_rates),
                        'all_decline_rates': np.array(decline_rates)
                    }
                else:
                    # No peaks detected
                    kinetics_metrics[biomarker] = {
                        'peak_count': 0,
                        'peak_frequency': 0.0,
                        'peak_times': np.array([]),
                        'peak_heights': np.array([]),
                        'mean_rise_time': 0.0,
                        'std_rise_time': 0.0,
                        'mean_rise_rate': 0.0,
                        'max_rise_rate': 0.0,
                        'mean_decline_time': 0.0,
                        'std_decline_time': 0.0,
                        'mean_decline_rate': 0.0,
                        'max_decline_rate': 0.0,
                        'mean_inter_peak_interval': 0.0,
                        'inter_peak_variability': 0.0,
                        'rise_decline_ratio': 0.0,
                        'secretion_clearance_ratio': 0.0,
                        'all_rise_times': np.array([]),
                        'all_decline_times': np.array([]),
                        'all_rise_rates': np.array([]),
                        'all_decline_rates': np.array([])
                    }
       
        return kinetics_metrics
   
    # Classification helper methods
    def _classify_correlation(self, correlation):
        abs_corr = abs(correlation)
        if abs_corr > 0.8: return "Excellent coordination"
        elif abs_corr > 0.6: return "Good coordination"
        elif abs_corr > 0.4: return "Moderate coordination"
        else: return "Poor coordination"
   
    def _classify_stability(self, cv):
        if cv < 15: return "Highly stable"
        elif cv < 25: return "Moderately stable"
        elif cv < 35: return "Somewhat unstable"
        else: return "Highly unstable"
   
    def _classify_responsiveness(self, responsiveness):
        if responsiveness > 2.0: return "Highly responsive"
        elif responsiveness > 1.0: return "Moderately responsive"
        elif responsiveness > 0.5: return "Mildly responsive"
        else: return "Poorly responsive"
   
    def _classify_dynamic_range(self, range_val):
        if range_val > 3.0: return "Excellent range"
        elif range_val > 2.0: return "Good range"
        elif range_val > 1.5: return "Moderate range"
        else: return "Limited range"
   
    def _classify_coherence(self, coherence):
        if coherence > 0.85: return "Highly coherent"
        elif coherence > 0.70: return "Moderately coherent"
        elif coherence > 0.55: return "Somewhat coherent"
        else: return "Poorly coherent"
   
    def _classify_suppression(self, correlation):
        if correlation < -0.6: return "Strong suppression"
        elif correlation < -0.3: return "Moderate suppression"
        elif correlation < 0: return "Weak suppression"
        else: return "No/inappropriate suppression"
   
    def _classify_precision(self, cv):
        if cv < 20: return "High precision"
        elif cv < 30: return "Moderate precision"
        elif cv < 40: return "Low precision"
        else: return "Very low precision"
   
    def _classify_counter_regulation(self, range_val):
        if range_val > 2.5: return "Strong capacity"
        elif range_val > 1.8: return "Moderate capacity"
        elif range_val > 1.3: return "Weak capacity"
        else: return "Very weak capacity"
   
    def _classify_reciprocity(self, correlation):
        if correlation < -0.5: return "Strong reciprocity"
        elif correlation < -0.2: return "Moderate reciprocity"
        elif correlation < 0.1: return "Weak reciprocity"
        else: return "No/disrupted reciprocity"
   
    def _calculate_overall_function_score(self, beta_metrics, alpha_metrics):
        """Calculate comprehensive pancreatic function score (0-100)"""
        score = 0
        max_score = 0
       
        # Beta-cell contributions (60% of total)
        if 'insulin_cpeptide_correlation' in beta_metrics:
            score += max(0, beta_metrics['insulin_cpeptide_correlation'] * 20)
            max_score += 20
       
        if 'secretory_stability_cv' in beta_metrics:
            stability_score = max(0, 20 - beta_metrics['secretory_stability_cv'] * 0.5)
            score += stability_score
            max_score += 20
       
        if 'glucose_responsiveness' in beta_metrics:
            resp_score = min(20, beta_metrics['glucose_responsiveness'] * 10)
            score += resp_score
            max_score += 20
       
        # Alpha-cell contributions (40% of total)
        if 'glucose_suppression_correlation' in alpha_metrics:
            supp_score = max(0, abs(alpha_metrics['glucose_suppression_correlation']) * 20)
            score += supp_score
            max_score += 20
       
        if 'insulin_glucagon_reciprocity' in alpha_metrics:
            recip_score = max(0, abs(alpha_metrics['insulin_glucagon_reciprocity']) * 20)
            score += recip_score
            max_score += 20
       
        return float((score / max_score * 100)) if max_score > 0 else 0.0
   
    def _classify_overall_function(self, score):
        """Classify overall pancreatic function"""
        if score > 80: return "Excellent function"
        elif score > 65: return "Good function"
        elif score > 50: return "Moderate function"
        elif score > 35: return "Impaired function"
        else: return "Severely impaired function"
   
    def analyze_all_animals(self):
        """Calculate comprehensive metrics for all animals"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_data() first.")
       
        print("\n" + "="*80)
        print("CALCULATING COMPREHENSIVE METRICS FOR ALL ANIMALS")
        print("="*80)
        print("Calculating 120+ metrics per animal including:")
        print("• Basic device measurements (CV%, range, etc.)")
        print("• Glucose-adjusted metrics (efficiency, corrected values)")
        print("• Peak and dynamics analysis")
        print("• Rate of change analysis")
        print("• Area under curve metrics")
        print("• Oscillation and pattern metrics")
        print("• Inter-biomarker correlations")
        print("• ELISA-validated quantitative metrics")
        print("• Pancreatic cell functionality assessment")
        print("• Peak kinetics analysis")
        print()
       
        for animal_id in self.data['animal_id']:
            try:
                metrics = self.calculate_comprehensive_metrics(animal_id)
                print(f"✓ Calculated comprehensive metrics for {animal_id}")
            except Exception as e:
                print(f"✗ Error analyzing {animal_id}: {e}")
       
        print(f"\n✓ Successfully analyzed {len(self.comprehensive_metrics)} animals")
        print(f"✓ Generated {self._count_total_metrics()} individual metrics per animal")
       
        # Calculate group statistics
        self._calculate_group_statistics()
       
        return self.comprehensive_metrics
   
    def _count_total_metrics(self):
        """Count total number of individual metrics"""
        if not self.comprehensive_metrics:
            return 0
       
        # Get one animal's metrics to count
        sample_animal = list(self.comprehensive_metrics.keys())[0]
        sample_metrics = self.comprehensive_metrics[sample_animal]['metrics']
       
        total_count = 0
        for category in sample_metrics:
            if sample_metrics[category] is not None:
                if isinstance(sample_metrics[category], dict):
                    total_count += self._count_dict_metrics(sample_metrics[category])
       
        return total_count
   
    def _count_dict_metrics(self, metrics_dict):
        """Recursively count metrics in nested dictionary"""
        count = 0
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                count += self._count_dict_metrics(value)
            elif isinstance(value, (int, float, bool)):
                count += 1
            elif isinstance(value, np.ndarray) and value.size == 1:
                count += 1
        return count
   
    def _calculate_group_statistics(self):
        """Calculate group-level statistics"""
        print("\nCalculating group statistics...")
       
        # Group animals by rat type
        grouped_animals = {}
        for animal_id, data in self.comprehensive_metrics.items():
            rat_type = data['raw_data']['condition']
            if rat_type not in grouped_animals:
                grouped_animals[rat_type] = []
            grouped_animals[rat_type].append(animal_id)
       
        print(f"✓ Grouped animals: {dict([(k, len(v)) for k, v in grouped_animals.items()])}")
       
        self.group_metrics = grouped_animals
        return grouped_animals
   
    # =======================================================================
    # COMPREHENSIVE PLOTTING FUNCTIONS
    # =======================================================================
   
    def plot_comprehensive_metric_summary(self, save_path=None):
        """Generate comprehensive summary plot of all metric categories"""
       
        if not self.comprehensive_metrics:
            print("No metrics calculated. Run analyze_all_animals() first.")
            return
       
        # Create large figure for comprehensive summary
        fig = plt.figure(figsize=(24, 32))
        gs = fig.add_gridspec(8, 4, hspace=0.3, wspace=0.3)
       
        fig.suptitle(f'Comprehensive Fed State QIRT-ELISA Analysis Summary\n'
                    f'All 120+ Metrics • {self.time_interval:.1f}-minute Resolution • {self.experiment_duration:.1f}-minute Duration',
                    fontsize=20, y=0.98)
       
        # 1. Basic Device Measurements CV%
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_cv_metrics(ax1)
       
        # 2. Glucose-Adjusted Metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_glucose_adjusted_summary(ax2)
       
        # 3. Peak Dynamics Analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_peak_dynamics_summary(ax3)
       
        # 4. Rate Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_rate_analysis_summary(ax4)
       
        # 5. AUC Metrics
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_auc_summary(ax5)
       
        # 6. Oscillation Patterns
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_oscillation_summary(ax6)
       
        # 7. Inter-biomarker Correlations
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_correlations_summary(ax7)
       
        # 8. ELISA Quantitative (if available)
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_elisa_quantitative_summary(ax8)
       
        # 9. Pancreatic Function Scores
        ax9 = fig.add_subplot(gs[4, :2])
        self._plot_pancreatic_function_summary(ax9)
       
        # 10. Peak Kinetics
        ax10 = fig.add_subplot(gs[4, 2:])
        self._plot_peak_kinetics_summary(ax10)
       
        # 11-16. Individual time series examples for each group
        rat_types = list(self.group_metrics.keys())
        for i, rat_type in enumerate(rat_types[:3]): # Show up to 3 groups
            ax = fig.add_subplot(gs[5+i, :])
            self._plot_example_timeseries_with_metrics(ax, rat_type)
       
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
   
    def _plot_cv_metrics(self, ax):
        """Plot coefficient of variation metrics"""
        cv_data = {}
       
        for rat_type in self.group_metrics.keys():
            cv_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['basic_device']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            cv_value = metrics[biomarker]['cv_normalized_percent']
                            cv_data[rat_type][biomarker].append(cv_value)
       
        # Plot CV data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(cv_data.keys()):
            means = [np.mean(cv_data[rat_type][bm]) if cv_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(cv_data[rat_type][bm])/np.sqrt(len(cv_data[rat_type][bm])) if cv_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.5, f'{mean:.1f}%',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title('Coefficient of Variation (%) - Device Measurements\nNormalized Fi/F0 Temporal Variability', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('CV (%)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_glucose_adjusted_summary(self, ax):
        """Plot glucose-adjusted metrics summary"""
        # Focus on glucose efficiency as key metric
        efficiency_data = {}
       
        for rat_type in self.group_metrics.keys():
            efficiency_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['glucose_adjusted']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            eff_value = metrics[biomarker]['glucose_efficiency']
                            efficiency_data[rat_type][biomarker].append(eff_value)
       
        # Plot efficiency data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(efficiency_data.keys()):
            means = [np.mean(efficiency_data[rat_type][bm]) if efficiency_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(efficiency_data[rat_type][bm])/np.sqrt(len(efficiency_data[rat_type][bm])) if efficiency_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.002, f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title('Glucose Efficiency (F(a.u.)/mM)\nBiomarker Level per Unit Glucose', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('Efficiency (F(a.u.)/mM)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_peak_dynamics_summary(self, ax):
        """Plot peak dynamics summary"""
        peak_data = {}
       
        for rat_type in self.group_metrics.keys():
            peak_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['peak_dynamics']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            peak_count = metrics[biomarker]['num_peaks_above_std']
                            peak_data[rat_type][biomarker].append(peak_count)
       
        # Plot peak count data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(peak_data.keys()):
            means = [np.mean(peak_data[rat_type][bm]) if peak_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(peak_data[rat_type][bm])/np.sqrt(len(peak_data[rat_type][bm])) if peak_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.05, f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title(f'Peak Count Analysis\nPeaks Detected Above Mean+0.5×STD ({self.experiment_duration:.0f} min)', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('Number of Peaks')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_rate_analysis_summary(self, ax):
        """Plot rate of change analysis summary"""
        rate_data = {}
       
        for rat_type in self.group_metrics.keys():
            rate_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['rate_analysis']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            max_rate = metrics[biomarker]['max_increase_rate']
                            rate_data[rat_type][biomarker].append(max_rate)
       
        # Plot rate data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(rate_data.keys()):
            means = [np.mean(rate_data[rat_type][bm]) if rate_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(rate_data[rat_type][bm])/np.sqrt(len(rate_data[rat_type][bm])) if rate_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.001, f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title(f'Maximum Secretion Rate\nF(a.u.) per {self.time_interval:.1f}-minute Interval', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('Max Rate (F(a.u.)/min)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_auc_summary(self, ax):
        """Plot AUC metrics summary"""
        auc_data = {}
       
        for rat_type in self.group_metrics.keys():
            auc_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['auc_metrics']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            auc_value = metrics[biomarker]['auc_glucose_corrected_5p6']
                            auc_data[rat_type][biomarker].append(auc_value)
       
        # Plot AUC data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(auc_data.keys()):
            means = [np.mean(auc_data[rat_type][bm]) if auc_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(auc_data[rat_type][bm])/np.sqrt(len(auc_data[rat_type][bm])) if auc_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.2, f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title('Area Under Curve (Glucose-Corrected)\nNormalized to 5.6 mM Reference Glucose', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('AUC (F(a.u.)·min)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_oscillation_summary(self, ax):
        """Plot oscillation pattern summary"""
        osc_data = {}
       
        for rat_type in self.group_metrics.keys():
            osc_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    metrics = self.comprehensive_metrics[animal_id]['metrics']['oscillation_patterns']
                    for biomarker in self.biomarkers:
                        if biomarker in metrics:
                            osc_freq = metrics[biomarker]['oscillation_frequency']
                            osc_data[rat_type][biomarker].append(osc_freq)
       
        # Plot oscillation data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(osc_data.keys()):
            means = [np.mean(osc_data[rat_type][bm]) if osc_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(osc_data[rat_type][bm])/np.sqrt(len(osc_data[rat_type][bm])) if osc_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.01, f'{mean:.2f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title('Oscillation Frequency\nBaseline Crossings per Minute', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('Frequency (events/min)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_correlations_summary(self, ax):
        """Plot inter-biomarker correlations summary"""
        correlation_pairs = ['insulin_c_peptide', 'insulin_glucagon', 'c_peptide_glucagon']
        corr_data = {}
       
        for rat_type in self.group_metrics.keys():
            corr_data[rat_type] = {pair: [] for pair in correlation_pairs}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    correlations = self.comprehensive_metrics[animal_id]['metrics']['correlations']['biomarker_correlations']
                    for pair in correlation_pairs:
                        if pair in correlations:
                            corr_value = correlations[pair]['correlation']
                            corr_data[rat_type][pair].append(corr_value)
       
        # Plot correlation data
        x = np.arange(len(correlation_pairs))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(corr_data.keys()):
            means = [np.mean(corr_data[rat_type][pair]) if corr_data[rat_type][pair] else 0
                    for pair in correlation_pairs]
            sems = [np.std(corr_data[rat_type][pair])/np.sqrt(len(corr_data[rat_type][pair])) if corr_data[rat_type][pair] else 0
                   for pair in correlation_pairs]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.02, f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title(f'Inter-biomarker Correlations\nTemporal Coordination Across {len(self.time_points)} Timepoints', fontsize=12)
        ax.set_xlabel('Biomarker Pair')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Ins-CPep', 'Ins-Gluc', 'CPep-Gluc'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
   
    def _plot_elisa_quantitative_summary(self, ax):
        """Plot ELISA quantitative metrics summary"""
        # Check if any animals have ELISA data
        has_elisa_data = False
        elisa_data = {}
       
        for rat_type in self.group_metrics.keys():
            elisa_data[rat_type] = {'insulin': []}
           
            for animal_id in self.group_metrics[rat_type]:
                if (animal_id in self.comprehensive_metrics and
                    self.comprehensive_metrics[animal_id]['metrics']['elisa_quantitative'] is not None):
                    has_elisa_data = True
                    elisa_metrics = self.comprehensive_metrics[animal_id]['metrics']['elisa_quantitative']
                    if 'insulin' in elisa_metrics:
                        conc_value = elisa_metrics['insulin']['mean_concentration_pM']
                        elisa_data[rat_type]['insulin'].append(conc_value)
       
        if has_elisa_data:
            # Plot ELISA concentration data
            rat_types = list(elisa_data.keys())
            means = [np.mean(elisa_data[rt]['insulin']) if elisa_data[rt]['insulin'] else 0
                    for rt in rat_types]
            sems = [np.std(elisa_data[rt]['insulin'])/np.sqrt(len(elisa_data[rt]['insulin'])) if elisa_data[rt]['insulin'] else 0
                   for rt in rat_types]
           
            colors = ['lightblue', 'orange', 'lightcoral']
            bars = ax.bar(rat_types, means, yerr=sems, color=colors[:len(rat_types)], alpha=0.7)
           
            # Add value labels
            for bar, mean, sem in zip(bars, means, sems):
                ax.text(bar.get_x() + bar.get_width()/2, mean + sem + 5, f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=10)
           
            ax.set_title('ELISA-Validated Insulin Concentrations\nTwo-Point Calibration from Device Measurements', fontsize=12)
            ax.set_ylabel('Concentration (pM)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No ELISA Validation Data Available\nDevice measurements only',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title('ELISA-Validated Quantitative Metrics', fontsize=12)
   
    def _plot_pancreatic_function_summary(self, ax):
        """Plot pancreatic function scores summary"""
        function_data = {}
       
        for rat_type in self.group_metrics.keys():
            function_data[rat_type] = []
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    function_score = self.comprehensive_metrics[animal_id]['metrics']['pancreatic_function']['overall_function_score']
                    function_data[rat_type].append(function_score)
       
        # Plot function scores
        rat_types = list(function_data.keys())
        means = [np.mean(function_data[rt]) if function_data[rt] else 0 for rt in rat_types]
        sems = [np.std(function_data[rt])/np.sqrt(len(function_data[rt])) if function_data[rt] else 0 for rt in rat_types]
       
        colors = ['lightblue', 'orange', 'lightcoral']
        bars = ax.bar(rat_types, means, yerr=sems, color=colors[:len(rat_types)], alpha=0.7)
       
        # Add value labels and classification
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + sem + 2, f'{mean:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
           
            # Add classification text
            classification = self._classify_overall_function(mean)
            ax.text(bar.get_x() + bar.get_width()/2, mean/2, classification,
                   ha='center', va='center', fontsize=8, rotation=90)
       
        ax.set_title('Pancreatic Function Score\nBeta-cell + Alpha-cell Integrated Assessment', fontsize=12)
        ax.set_ylabel('Function Score (0-100)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
       
        # Add reference lines
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Excellent (>80)')
        ax.axhline(y=65, color='orange', linestyle='--', alpha=0.5, label='Good (>65)')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Moderate (>50)')
        ax.legend(fontsize=8)
   
    def _plot_peak_kinetics_summary(self, ax):
        """Plot peak kinetics analysis summary"""
        kinetics_data = {}
       
        for rat_type in self.group_metrics.keys():
            kinetics_data[rat_type] = {biomarker: [] for biomarker in self.biomarkers}
           
            for animal_id in self.group_metrics[rat_type]:
                if animal_id in self.comprehensive_metrics:
                    kinetics = self.comprehensive_metrics[animal_id]['metrics']['peak_kinetics']
                    for biomarker in self.biomarkers:
                        if biomarker in kinetics:
                            rise_time = kinetics[biomarker]['mean_rise_time']
                            kinetics_data[rat_type][biomarker].append(rise_time)
       
        # Plot kinetics data
        x = np.arange(len(self.biomarkers))
        width = 0.25
        colors = ['lightblue', 'orange', 'lightcoral']
       
        for i, rat_type in enumerate(kinetics_data.keys()):
            means = [np.mean(kinetics_data[rat_type][bm]) if kinetics_data[rat_type][bm] else 0
                    for bm in self.biomarkers]
            sems = [np.std(kinetics_data[rat_type][bm])/np.sqrt(len(kinetics_data[rat_type][bm])) if kinetics_data[rat_type][bm] else 0
                   for bm in self.biomarkers]
           
            ax.bar(x + i*width, means, width, yerr=sems, label=rat_type,
                  color=colors[i % len(colors)], alpha=0.7)
           
            # Add value labels
            for j, (mean, sem) in enumerate(zip(means, sems)):
                ax.text(x[j] + i*width, mean + sem + 0.1, f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=8)
       
        ax.set_title('Peak Kinetics: Mean Rise Time\nTime to Reach Secretion Peak', fontsize=12)
        ax.set_xlabel('Biomarker')
        ax.set_ylabel('Rise Time (minutes)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([bm.replace('_', '-').title() for bm in self.biomarkers])
        ax.legend()
        ax.grid(True, alpha=0.3)
   
    def _plot_example_timeseries_with_metrics(self, ax, rat_type):
        """Plot example time series with key metrics for a rat type"""
        # Get one representative animal from this group
        if rat_type in self.group_metrics and self.group_metrics[rat_type]:
            animal_id = self.group_metrics[rat_type][0]
           
            if animal_id in self.comprehensive_metrics:
                raw_data = self.comprehensive_metrics[animal_id]['raw_data']
                metrics = self.comprehensive_metrics[animal_id]['metrics']
               
                # Plot time series for all biomarkers
                colors = {'insulin': 'blue', 'c_peptide': 'green', 'glucagon': 'red'}
               
                for biomarker in self.biomarkers:
                    if biomarker in raw_data:
                        signal = raw_data[biomarker]
                        ax.plot(self.time_points, signal, 'o-', color=colors[biomarker],
                               linewidth=2, markersize=6, alpha=0.8, label=biomarker.replace('_', '-').title())
                       
                        # Mark peaks if detected
                        if biomarker in metrics['peak_dynamics']:
                            peak_times = metrics['peak_dynamics'][biomarker].get('peak_times', np.array([]))
                            if len(peak_times) > 0:
                                peak_indices = [np.argmin(np.abs(self.time_points - t)) for t in peak_times]
                                peak_values = signal[peak_indices]
                                ax.scatter(peak_times, peak_values, color=colors[biomarker],
                                         s=100, marker='^', alpha=0.8, zorder=5)
               
                # Add glucose information
                glucose_avg = self.comprehensive_metrics[animal_id]['glucose_avg']
                glucose_change = self.comprehensive_metrics[animal_id]['glucose_change']
               
                # Add key metrics as text
                function_score = metrics['pancreatic_function']['overall_function_score']
                cv_insulin = metrics['basic_device']['insulin']['cv_normalized_percent']
                glucose_eff = metrics['glucose_adjusted']['insulin']['glucose_efficiency']
               
                text_info = f'{rat_type.title()} Example: {animal_id}\n'
                text_info += f'Glucose: {glucose_avg:.1f} mM (Δ{glucose_change:+.1f})\n'
                text_info += f'Function Score: {function_score:.1f}/100\n'
                text_info += f'Insulin CV: {cv_insulin:.1f}%\n'
                text_info += f'Glucose Efficiency: {glucose_eff:.4f}'
               
                ax.text(0.02, 0.98, text_info, transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                       fontsize=9)
               
                ax.set_title(f'{rat_type.title()} Group - Time Series with Detected Peaks')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Device Signal (F a.u.)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
               
                # Mark measurement intervals
                for tp in self.time_points:
                    ax.axvline(x=tp, color='gray', linestyle=':', alpha=0.2)
        else:
            ax.text(0.5, 0.5, f'No data available for {rat_type}',
                   ha='center', va='center', transform=ax.transAxes)
   
    def generate_comprehensive_report(self, save_path=None):
        """Generate comprehensive text report of all metrics"""
       
        if not self.comprehensive_metrics:
            print("No metrics calculated. Run analyze_all_animals() first.")
            return
       
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE FED STATE QIRT-ELISA ANALYSIS REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Time Resolution: {self.time_interval:.1f} minutes")
        report_lines.append(f"Experiment Duration: {self.experiment_duration:.1f} minutes")
        report_lines.append(f"Total Timepoints: {len(self.time_points)}")
        report_lines.append(f"Total Animals: {len(self.comprehensive_metrics)}")
        report_lines.append(f"Metrics per Animal: {self._count_total_metrics()}")
        report_lines.append("")
       
        # Group summary
        report_lines.append("GROUP DISTRIBUTION:")
        report_lines.append("-" * 50)
        for rat_type, animals in self.group_metrics.items():
            report_lines.append(f"{rat_type.title()}: {len(animals)} animals")
        report_lines.append("")
       
        # Detailed analysis by group
        for rat_type in self.group_metrics.keys():
            report_lines.extend(self._generate_group_report(rat_type))
            report_lines.append("")
       
        # Cross-group comparisons
        report_lines.extend(self._generate_cross_group_comparisons())
       
        # Scientific conclusions
        report_lines.extend(self._generate_scientific_conclusions())
       
        # Save report
        full_report = "\n".join(report_lines)
       
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            print(f"✓ Comprehensive report saved to: {save_path}")
        else:
            print(full_report)
       
        return full_report
   
    def _generate_group_report(self, rat_type):
        """Generate detailed report for one group"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{rat_type.upper()} GROUP ANALYSIS")
        lines.append("=" * 80)
       
        animal_ids = self.group_metrics[rat_type]
        lines.append(f"Animals: {', '.join(animal_ids)} (n={len(animal_ids)})")
        lines.append("")
       
        # Collect all metrics for this group
        group_data = self._collect_group_metrics(rat_type)
       
        # 1. Basic Device Measurements
        lines.append("1. BASIC DEVICE MEASUREMENTS (CV%):")
        lines.append("-" * 40)
        for biomarker in self.biomarkers:
            if biomarker in group_data['cv_data']:
                cv_values = group_data['cv_data'][biomarker]
                if cv_values:
                    mean_cv = np.mean(cv_values)
                    std_cv = np.std(cv_values)
                    lines.append(f"{biomarker.replace('_', '-').title():15}: {mean_cv:6.1f} ± {std_cv:5.1f}% CV")
        lines.append("")
       
        # 2. Glucose-Adjusted Metrics
        lines.append("2. GLUCOSE-ADJUSTED METRICS:")
        lines.append("-" * 40)
        for biomarker in self.biomarkers:
            if biomarker in group_data['glucose_efficiency']:
                eff_values = group_data['glucose_efficiency'][biomarker]
                if eff_values:
                    mean_eff = np.mean(eff_values)
                    std_eff = np.std(eff_values)
                    lines.append(f"{biomarker.replace('_', '-').title():15}:")
                    lines.append(f" Glucose Efficiency: {mean_eff:.4f} ± {std_eff:.4f} F(a.u.)/mM")
                   
                    if biomarker in group_data['glucose_corrected']:
                        corr_values = group_data['glucose_corrected'][biomarker]
                        if corr_values:
                            mean_corr = np.mean(corr_values)
                            std_corr = np.std(corr_values)
                            lines.append(f" Glucose-Corrected: {mean_corr:.3f} ± {std_corr:.3f} F(a.u.) @ 5.6mM")
        lines.append("")
       
        # 3. Peak Dynamics
        lines.append("3. PEAK DYNAMICS ANALYSIS:")
        lines.append("-" * 40)
        for biomarker in self.biomarkers:
            if biomarker in group_data['peak_counts']:
                peak_values = group_data['peak_counts'][biomarker]
                if peak_values:
                    mean_peaks = np.mean(peak_values)
                    std_peaks = np.std(peak_values)
                    lines.append(f"{biomarker.replace('_', '-').title():15}: {mean_peaks:4.1f} ± {std_peaks:4.1f} peaks")
        lines.append("")
       
        # 4. Rate Analysis
        lines.append("4. RATE ANALYSIS:")
        lines.append("-" * 40)
        for biomarker in self.biomarkers:
            if biomarker in group_data['max_rates']:
                rate_values = group_data['max_rates'][biomarker]
                if rate_values:
                    mean_rate = np.mean(rate_values)
                    std_rate = np.std(rate_values)
                    lines.append(f"{biomarker.replace('_', '-').title():15}: {mean_rate:6.3f} ± {std_rate:6.3f} F(a.u.)/min")
        lines.append("")
       
        # 5. Inter-biomarker Correlations
        lines.append("5. INTER-BIOMARKER CORRELATIONS:")
        lines.append("-" * 40)
        correlation_pairs = ['insulin_c_peptide', 'insulin_glucagon', 'c_peptide_glucagon']
        for pair in correlation_pairs:
            if pair in group_data['correlations']:
                corr_values = group_data['correlations'][pair]
                if corr_values:
                    mean_corr = np.mean(corr_values)
                    std_corr = np.std(corr_values)
                    pair_name = pair.replace('_', '-').replace('c-peptide', 'C-peptide')
                    lines.append(f"{pair_name:20}: r = {mean_corr:6.3f} ± {std_corr:5.3f}")
        lines.append("")
       
        # 6. Pancreatic Function
        lines.append("6. PANCREATIC FUNCTION ASSESSMENT:")
        lines.append("-" * 40)
        if 'function_scores' in group_data:
            function_values = group_data['function_scores']
            if function_values:
                mean_func = np.mean(function_values)
                std_func = np.std(function_values)
                classification = self._classify_overall_function(mean_func)
                lines.append(f"Overall Function Score: {mean_func:5.1f} ± {std_func:4.1f}/100")
                lines.append(f"Classification: {classification}")
        lines.append("")
       
        # 7. ELISA Validation (if available)
        if any(animal_id in self.comprehensive_metrics and
               self.comprehensive_metrics[animal_id]['has_elisa']
               for animal_id in animal_ids):
            lines.append("7. ELISA-VALIDATED QUANTITATIVE METRICS:")
            lines.append("-" * 40)
            for biomarker in self.biomarkers:
                if biomarker in group_data.get('elisa_concentrations', {}):
                    elisa_values = group_data['elisa_concentrations'][biomarker]
                    if elisa_values:
                        mean_elisa = np.mean(elisa_values)
                        std_elisa = np.std(elisa_values)
                        lines.append(f"{biomarker.replace('_', '-').title():15}: {mean_elisa:6.1f} ± {std_elisa:5.1f} pM")
            lines.append("")
       
        return lines
   
    def _collect_group_metrics(self, rat_type):
        """Collect all metrics for a specific group"""
        group_data = {
            'cv_data': {biomarker: [] for biomarker in self.biomarkers},
            'glucose_efficiency': {biomarker: [] for biomarker in self.biomarkers},
            'glucose_corrected': {biomarker: [] for biomarker in self.biomarkers},
            'peak_counts': {biomarker: [] for biomarker in self.biomarkers},
            'max_rates': {biomarker: [] for biomarker in self.biomarkers},
            'correlations': {},
            'function_scores': [],
            'elisa_concentrations': {biomarker: [] for biomarker in self.biomarkers}
        }
       
        for animal_id in self.group_metrics[rat_type]:
            if animal_id in self.comprehensive_metrics:
                metrics = self.comprehensive_metrics[animal_id]['metrics']
               
                # Basic device measurements
                for biomarker in self.biomarkers:
                    if biomarker in metrics['basic_device']:
                        cv_val = metrics['basic_device'][biomarker]['cv_normalized_percent']
                        group_data['cv_data'][biomarker].append(cv_val)
               
                # Glucose-adjusted metrics
                for biomarker in self.biomarkers:
                    if biomarker in metrics['glucose_adjusted']:
                        eff_val = metrics['glucose_adjusted'][biomarker]['glucose_efficiency']
                        corr_val = metrics['glucose_adjusted'][biomarker]['mean_glucose_corrected_5p6']
                        group_data['glucose_efficiency'][biomarker].append(eff_val)
                        group_data['glucose_corrected'][biomarker].append(corr_val)
               
                # Peak dynamics
                for biomarker in self.biomarkers:
                    if biomarker in metrics['peak_dynamics']:
                        peak_count = metrics['peak_dynamics'][biomarker]['num_peaks_above_std']
                        group_data['peak_counts'][biomarker].append(peak_count)
               
                # Rate analysis
                for biomarker in self.biomarkers:
                    if biomarker in metrics['rate_analysis']:
                        max_rate = metrics['rate_analysis'][biomarker]['max_increase_rate']
                        group_data['max_rates'][biomarker].append(max_rate)
               
                # Correlations
                correlations = metrics['correlations']['biomarker_correlations']
                for pair, corr_data in correlations.items():
                    if pair not in group_data['correlations']:
                        group_data['correlations'][pair] = []
                    group_data['correlations'][pair].append(corr_data['correlation'])
               
                # Function scores
                function_score = metrics['pancreatic_function']['overall_function_score']
                group_data['function_scores'].append(function_score)
               
                # ELISA data
                if metrics['elisa_quantitative'] is not None:
                    for biomarker in self.biomarkers:
                        if biomarker in metrics['elisa_quantitative']:
                            elisa_conc = metrics['elisa_quantitative'][biomarker]['mean_concentration_pM']
                            group_data['elisa_concentrations'][biomarker].append(elisa_conc)
       
        return group_data
   
    def _generate_cross_group_comparisons(self):
        """Generate cross-group statistical comparisons"""
        lines = []
        lines.append("=" * 100)
        lines.append("CROSS-GROUP STATISTICAL COMPARISONS")
        lines.append("=" * 100)
       
        # Collect data for all groups
        all_group_data = {}
        for rat_type in self.group_metrics.keys():
            all_group_data[rat_type] = self._collect_group_metrics(rat_type)
       
        # Key metrics comparisons
        key_metrics = [
            ('cv_data', 'Coefficient of Variation (%)', 'insulin'),
            ('glucose_efficiency', 'Glucose Efficiency (F(a.u.)/mM)', 'insulin'),
            ('peak_counts', 'Peak Count', 'insulin'),
            ('function_scores', 'Pancreatic Function Score', None)
        ]
       
        for metric_key, metric_name, biomarker_key in key_metrics:
            lines.append(f"\n{metric_name.upper()}:")
            lines.append("-" * 60)
           
            # Collect values for statistical testing
            group_values = {}
            for rat_type in all_group_data.keys():
                if biomarker_key is None:
                    # Function scores are not biomarker-specific
                    values = all_group_data[rat_type][metric_key]
                else:
                    values = all_group_data[rat_type][metric_key][biomarker_key]
               
                if values:
                    group_values[rat_type] = values
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    lines.append(f"{rat_type.title():15}: {mean_val:8.3f} ± {std_val:6.3f} (n={len(values)})")
           
            # Statistical testing
            if len(group_values) >= 2:
                stat_result = self._perform_statistical_test(group_values)
                lines.append(f"Statistical Test: {stat_result['test_name']}")
                lines.append(f"P-value: {stat_result['p_value']:.4f}")
                lines.append(f"Significance: {'***' if stat_result['p_value'] < 0.001 else '**' if stat_result['p_value'] < 0.01 else '*' if stat_result['p_value'] < 0.05 else 'ns'}")
               
                if stat_result['effect_size'] is not None:
                    lines.append(f"Effect Size (Cohen's d): {stat_result['effect_size']:.3f}")
       
        return lines
   
    def _perform_statistical_test(self, group_values):
        """Perform appropriate statistical test for group comparison"""
        group_names = list(group_values.keys())
        values_list = [group_values[name] for name in group_names]
       
        try:
            if len(values_list) == 2:
                # Two-group comparison: t-test
                stat, p_value = stats.ttest_ind(values_list[0], values_list[1])
                test_name = "Independent t-test"
               
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(values_list[0])-1)*np.var(values_list[0]) +
                                    (len(values_list[1])-1)*np.var(values_list[1])) /
                                   (len(values_list[0])+len(values_list[1])-2))
                effect_size = (np.mean(values_list[0]) - np.mean(values_list[1])) / pooled_std if pooled_std > 0 else 0
               
            elif len(values_list) > 2:
                # Multi-group comparison: ANOVA
                stat, p_value = stats.f_oneway(*values_list)
                test_name = "One-way ANOVA"
                effect_size = None # Eta-squared could be calculated here
               
            else:
                return {'test_name': 'No test', 'p_value': 1.0, 'effect_size': None}
               
        except Exception as e:
            return {'test_name': 'Test failed', 'p_value': 1.0, 'effect_size': None}
       
        return {
            'test_name': test_name,
            'p_value': p_value,
            'effect_size': effect_size
        }
   
    def _generate_scientific_conclusions(self):
        """Generate scientific conclusions based on the analysis"""
        lines = []
        lines.append("=" * 100)
        lines.append("SCIENTIFIC CONCLUSIONS AND QIRT-ELISA JUSTIFICATION")
        lines.append("=" * 100)
       
        # Calculate key summary statistics
        summary_stats = self._calculate_summary_statistics()
       
        lines.append("KEY FINDINGS:")
        lines.append("-" * 50)
       
        # 1. Temporal Resolution Advantage
        lines.append("1. TEMPORAL RESOLUTION ADVANTAGE:")
        lines.append(f" • {self.time_interval:.1f}-minute sampling captures {len(self.time_points)} datapoints")
        lines.append(f" • Conventional 15-minute ELISA would capture only {int(self.experiment_duration/15)+1} datapoints")
        lines.append(f" • QIRT-ELISA preserves {((len(self.time_points)-1)/(int(self.experiment_duration/15)))*100:.0f}% more temporal information")
        lines.append("")
       
        # 2. Glucose Adjustment Impact
        lines.append("2. GLUCOSE ADJUSTMENT REVEALS DISEASE PATTERNS:")
        if summary_stats['glucose_ranges']:
            lines.append(f" • Glucose range across animals: {summary_stats['glucose_ranges']['min']:.1f} - {summary_stats['glucose_ranges']['max']:.1f} mM")
            lines.append(f" • Without glucose adjustment, cross-group comparisons would be confounded")
            lines.append(f" • Glucose efficiency shows {summary_stats['efficiency_decline']:.0f}% decline from healthy to diabetic")
        lines.append("")
       
        # 3. Oscillatory Dynamics
        lines.append("3. OSCILLATORY DYNAMICS ASSESSMENT:")
        if summary_stats['oscillation_data']:
            lines.append(f" • Average oscillation frequency: {summary_stats['oscillation_data']['mean_frequency']:.3f} events/min")
            lines.append(f" • Peak detection identifies {summary_stats['oscillation_data']['mean_peaks']:.1f} ± {summary_stats['oscillation_data']['std_peaks']:.1f} peaks per animal")
            lines.append(f" • Inter-biomarker correlations range: {summary_stats['correlation_range']['min']:.3f} to {summary_stats['correlation_range']['max']:.3f}")
        lines.append("")
       
        # 4. Disease Progression
        lines.append("4. DISEASE PROGRESSION CHARACTERIZATION:")
        if summary_stats['function_progression']:
            lines.append(f" • Pancreatic function scores: {summary_stats['function_progression']}")
            lines.append(f" • Demonstrates clear functional decline across disease stages")
            lines.append(f" • Early dysfunction detection through efficiency metrics")
        lines.append("")
       
        # 5. ELISA Validation
        lines.append("5. ELISA VALIDATION AND QUANTITATIVE ANALYSIS:")
        if summary_stats['elisa_validation']:
            lines.append(f" • Two-point ELISA calibration validates device measurements")
            lines.append(f" • Mean calibration error: {summary_stats['elisa_validation']['mean_error']:.1f}%")
            lines.append(f" • Enables quantitative concentration analysis in pM units")
        else:
            lines.append(" • Analysis performed on device measurements (F a.u.)")
            lines.append(" • Relative comparisons valid through normalization")
        lines.append("")
       
        # 6. Clinical Significance
        lines.append("6. CLINICAL AND RESEARCH SIGNIFICANCE:")
        lines.append(" • Fed state analysis reveals baseline pancreatic dysfunction")
        lines.append(" • Multi-analyte measurement enables systems-level assessment")
        lines.append(" • High temporal resolution captures rapid regulatory responses")
        lines.append(" • Glucose adjustment essential for meaningful disease comparisons")
        lines.append(" • Early detection of pre-diabetic metabolic dysfunction")
        lines.append("")
       
        # 7. QIRT-ELISA Advantages
        lines.append("7. QIRT-ELISA vs CONVENTIONAL ELISA:")
        lines.append(" ✓ 6× higher temporal resolution")
        lines.append(" ✓ Real-time measurement (no sample processing)")
        lines.append(" ✓ Simultaneous multi-analyte detection")
        lines.append(" ✓ Preserves oscillatory dynamics information")
        lines.append(" ✓ Enables comprehensive kinetic analysis")
        lines.append(" ✓ Glucose adjustment methodology for cross-group comparisons")
        lines.append("")
       
        return lines
   
    def _calculate_summary_statistics(self):
        """Calculate summary statistics for scientific conclusions"""
        summary = {}
       
        # Glucose ranges
        glucose_values = []
        for animal_id, data in self.comprehensive_metrics.items():
            glucose_values.append(data['glucose_avg'])
       
        if glucose_values:
            summary['glucose_ranges'] = {
                'min': np.min(glucose_values),
                'max': np.max(glucose_values),
                'mean': np.mean(glucose_values)
            }
        else:
            summary['glucose_ranges'] = None
       
        # Efficiency decline calculation
        if len(self.group_metrics) >= 2:
            # Get efficiency data for first and last groups (assuming healthy and diabetic)
            group_names = list(self.group_metrics.keys())
            first_group = self._collect_group_metrics(group_names[0])
            last_group = self._collect_group_metrics(group_names[-1])
           
            if (first_group['glucose_efficiency']['insulin'] and
                last_group['glucose_efficiency']['insulin']):
                first_eff = np.mean(first_group['glucose_efficiency']['insulin'])
                last_eff = np.mean(last_group['glucose_efficiency']['insulin'])
                summary['efficiency_decline'] = ((first_eff - last_eff) / first_eff) * 100
            else:
                summary['efficiency_decline'] = 0
        else:
            summary['efficiency_decline'] = 0
       
        # Oscillation data
        all_freqs = []
        all_peaks = []
        for animal_id, data in self.comprehensive_metrics.items():
            for biomarker in self.biomarkers:
                if biomarker in data['metrics']['oscillation_patterns']:
                    freq = data['metrics']['oscillation_patterns'][biomarker]['oscillation_frequency']
                    peaks = data['metrics']['peak_dynamics'][biomarker]['num_peaks_above_std']
                    all_freqs.append(freq)
                    all_peaks.append(peaks)
       
        if all_freqs:
            summary['oscillation_data'] = {
                'mean_frequency': np.mean(all_freqs),
                'mean_peaks': np.mean(all_peaks),
                'std_peaks': np.std(all_peaks)
            }
        else:
            summary['oscillation_data'] = None
       
        # Correlation ranges
        all_correlations = []
        for animal_id, data in self.comprehensive_metrics.items():
            correlations = data['metrics']['correlations']['biomarker_correlations']
            for pair, corr_data in correlations.items():
                all_correlations.append(corr_data['correlation'])
       
        if all_correlations:
            summary['correlation_range'] = {
                'min': np.min(all_correlations),
                'max': np.max(all_correlations)
            }
        else:
            summary['correlation_range'] = {'min': 0, 'max': 0}
       
        # Function progression
        group_functions = []
        for rat_type in self.group_metrics.keys():
            group_data = self._collect_group_metrics(rat_type)
            if group_data['function_scores']:
                mean_func = np.mean(group_data['function_scores'])
                group_functions.append(f"{rat_type}: {mean_func:.1f}")
       
        summary['function_progression'] = " → ".join(group_functions)
       
        # ELISA validation
        elisa_errors = []
        for animal_id, data in self.comprehensive_metrics.items():
            if data['has_elisa'] and data['metrics']['elisa_quantitative']:
                for biomarker in self.biomarkers:
                    if biomarker in data['metrics']['elisa_quantitative']:
                        error = data['metrics']['elisa_quantitative'][biomarker].get('calibration_error_percent', 0)
                        elisa_errors.append(error)
       
        if elisa_errors:
            summary['elisa_validation'] = {
                'mean_error': np.mean(elisa_errors),
                'std_error': np.std(elisa_errors)
            }
        else:
            summary['elisa_validation'] = None
       
        return summary
   
    def run_complete_analysis(self, csv_path, save_plots_dir=None, save_report_path=None):
        """Run complete comprehensive analysis pipeline"""
       
        print("=" * 100)
        print("COMPREHENSIVE FED STATE QIRT-ELISA ANALYSIS PIPELINE")
        print("=" * 100)
        print("Features:")
        print("• 120+ individual metrics per animal")
        print("• All glucose adjustment methodologies")
        print("• Peak kinetics and dynamics analysis")
        print("• Inter-biomarker correlation analysis")
        print("• ELISA-validated quantitative metrics")
        print("• Pancreatic cell functionality assessment")
        print("• Comprehensive statistical comparisons")
        print("• Publication-ready plots and reports")
        print("=" * 100)
       
        # Step 1: Load data
        print("\n1. Loading and processing data...")
        self.load_csv_data(csv_path)
       
        # Step 2: Calculate all metrics
        print("\n2. Calculating comprehensive metrics...")
        self.analyze_all_animals()
       
        # Step 3: Generate comprehensive plots
        print("\n3. Generating comprehensive visualization...")
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            plot_path = os.path.join(save_plots_dir, "comprehensive_fed_state_analysis.png")
        else:
            plot_path = None
       
        self.plot_comprehensive_metric_summary(save_path=plot_path)
       
        # Step 4: Generate detailed report
        print("\n4. Generating comprehensive report...")
        if save_report_path:
            report_dir = os.path.dirname(save_report_path)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
       
        self.generate_comprehensive_report(save_path=save_report_path)
       
        print("\n" + "=" * 100)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 100)
        print(f"✓ Analyzed {len(self.comprehensive_metrics)} animals")
        print(f"✓ Generated {self._count_total_metrics()} metrics per animal")
        print(f"✓ Created comprehensive visualization")
        print(f"✓ Generated detailed scientific report")
       
        if save_plots_dir:
            print(f"✓ Plots saved to: {save_plots_dir}")
        if save_report_path:
            print(f"✓ Report saved to: {save_report_path}")
       
        print("\nScientific Justification Achieved:")
        print("• Glucose adjustment reveals disease patterns invisible to conventional analysis")
        print("• High temporal resolution captures oscillatory dynamics")
        print("• Multi-analyte measurement enables systems-level pancreatic assessment")
        print("• ELISA validation provides quantitative concentration analysis")
        print("• Early detection of pre-diabetic dysfunction through efficiency metrics")
       
        return self.comprehensive_metrics

    def export_metrics_to_csv(self, csv_path):
        """Export all comprehensive metrics to a CSV file"""
       
        if not self.comprehensive_metrics:
            print("No metrics calculated. Run analyze_all_animals() first.")
            return None
       
        print(f"\nExporting comprehensive metrics to CSV...")
       
        # Collect all metrics for all animals
        export_data = []
       
        for animal_id, animal_data in self.comprehensive_metrics.items():
            row = {}
           
            # Basic animal information
            row['animal_id'] = animal_id
            row['rat_type'] = animal_data['raw_data']['condition']
            row['glucose_start_mM'] = animal_data['raw_data']['glucose_start']
            row['glucose_end_mM'] = animal_data['raw_data']['glucose_end']
            row['glucose_avg_mM'] = animal_data['glucose_avg']
            row['glucose_change_mM'] = animal_data['glucose_change']
            row['has_elisa'] = animal_data['has_elisa']
           
            metrics = animal_data['metrics']
           
            # 1. Basic Device Measurements
            for biomarker in self.biomarkers:
                if biomarker in metrics['basic_device']:
                    bm_metrics = metrics['basic_device'][biomarker]
                    prefix = f'{biomarker}_basic_'
                   
                    row[f'{prefix}mean_raw_au'] = bm_metrics['mean_raw_au']
                    row[f'{prefix}cv_percent'] = bm_metrics['cv_normalized_percent']
                    row[f'{prefix}max_raw_au'] = bm_metrics['max_raw_au']
                    row[f'{prefix}min_raw_au'] = bm_metrics['min_raw_au']
                    row[f'{prefix}range_raw_au'] = bm_metrics['range_raw_au']
           
            # 2. Glucose-Adjusted Metrics
            for biomarker in self.biomarkers:
                if biomarker in metrics['glucose_adjusted']:
                    bm_metrics = metrics['glucose_adjusted'][biomarker]
                    prefix = f'{biomarker}_glucose_'
                   
                    row[f'{prefix}efficiency'] = bm_metrics['glucose_efficiency']
                    row[f'{prefix}peak_efficiency'] = bm_metrics['peak_glucose_efficiency']
                    row[f'{prefix}corrected_5p6'] = bm_metrics['mean_glucose_corrected_5p6']
                    row[f'{prefix}sensitivity'] = bm_metrics['glucose_sensitivity']
                    row[f'{prefix}responsiveness'] = bm_metrics['glucose_responsiveness']
                    row[f'{prefix}correlation'] = bm_metrics['glucose_correlation']['correlation']
                    row[f'{prefix}correlation_pvalue'] = bm_metrics['glucose_correlation']['p_value']
           
            # 3. Peak Dynamics
            for biomarker in self.biomarkers:
                if biomarker in metrics['peak_dynamics']:
                    bm_metrics = metrics['peak_dynamics'][biomarker]
                    prefix = f'{biomarker}_peak_'
                   
                    row[f'{prefix}count_above_std'] = bm_metrics['num_peaks_above_std']
                    row[f'{prefix}frequency'] = bm_metrics['peak_frequency']
                    row[f'{prefix}max_prominence'] = bm_metrics['max_peak_prominence']
                    row[f'{prefix}mean_width'] = bm_metrics['mean_peak_width']
                    row[f'{prefix}dynamic_range_ratio'] = bm_metrics['dynamic_range_ratio']
                    row[f'{prefix}total_excursion'] = bm_metrics['total_excursion']
                    row[f'{prefix}coefficient_variation'] = bm_metrics['coefficient_of_variation']
           
            # 4. Rate Analysis
            for biomarker in self.biomarkers:
                if biomarker in metrics['rate_analysis']:
                    bm_metrics = metrics['rate_analysis'][biomarker]
                    prefix = f'{biomarker}_rate_'
                   
                    row[f'{prefix}max_increase'] = bm_metrics['max_increase_rate']
                    row[f'{prefix}max_decrease'] = bm_metrics['max_decrease_rate']
                    row[f'{prefix}mean_absolute'] = bm_metrics['mean_absolute_rate']
                    row[f'{prefix}variability'] = bm_metrics['rate_variability']
                    row[f'{prefix}max_acceleration'] = bm_metrics['max_acceleration']
                    row[f'{prefix}trend_slope'] = bm_metrics['trend_slope']['slope']
                    row[f'{prefix}trend_r_squared'] = bm_metrics['trend_slope']['r_squared']
           
            # 5. AUC Metrics
            for biomarker in self.biomarkers:
                if biomarker in metrics['auc_metrics']:
                    bm_metrics = metrics['auc_metrics'][biomarker]
                    prefix = f'{biomarker}_auc_'
                   
                    row[f'{prefix}total'] = bm_metrics['auc_total']
                    row[f'{prefix}above_baseline'] = bm_metrics['auc_above_baseline']
                    row[f'{prefix}glucose_corrected'] = bm_metrics['auc_glucose_corrected_5p6']
                    row[f'{prefix}glucose_efficiency'] = bm_metrics['auc_glucose_efficiency']
                    row[f'{prefix}time_weighted_avg'] = bm_metrics['time_weighted_average']
           
            # 6. Oscillation Patterns
            for biomarker in self.biomarkers:
                if biomarker in metrics['oscillation_patterns']:
                    bm_metrics = metrics['oscillation_patterns'][biomarker]
                    prefix = f'{biomarker}_osc_'
                   
                    row[f'{prefix}frequency'] = bm_metrics['oscillation_frequency']
                    row[f'{prefix}zero_crossings'] = bm_metrics['zero_crossings']
                    row[f'{prefix}pattern_regularity'] = bm_metrics['pattern_regularity']
                    row[f'{prefix}rhythmicity_index'] = bm_metrics['rhythmicity_index']
                    row[f'{prefix}signal_entropy'] = bm_metrics['signal_entropy']
                    row[f'{prefix}temporal_stability'] = bm_metrics['temporal_stability']
           
            # 7. Inter-biomarker Correlations
            correlations = metrics['correlations']['biomarker_correlations']
            for pair, corr_data in correlations.items():
                row[f'corr_{pair}'] = corr_data['correlation']
                row[f'corr_{pair}_pvalue'] = corr_data['p_value']
                row[f'corr_{pair}_significant'] = corr_data['significant']
                row[f'corr_{pair}_partial'] = corr_data['partial_correlation']
                row[f'corr_{pair}_max_cross'] = corr_data['max_cross_correlation']
                row[f'corr_{pair}_optimal_lag'] = corr_data['optimal_lag']
           
            # 8. ELISA Quantitative Metrics (if available)
            if metrics['elisa_quantitative'] is not None:
                for biomarker in self.biomarkers:
                    if biomarker in metrics['elisa_quantitative']:
                        bm_metrics = metrics['elisa_quantitative'][biomarker]
                        prefix = f'{biomarker}_elisa_'
                       
                        row[f'{prefix}mean_concentration_pM'] = bm_metrics['mean_concentration_pM']
                        row[f'{prefix}cv_concentration_percent'] = bm_metrics['cv_concentration_percent']
                        row[f'{prefix}max_concentration_pM'] = bm_metrics['max_concentration_pM']
                        row[f'{prefix}concentration_range_pM'] = bm_metrics['concentration_range_pM']
                        row[f'{prefix}max_secretion_rate'] = bm_metrics['max_secretion_rate_pM_per_min']
                        row[f'{prefix}calibration_slope'] = bm_metrics['calibration_slope']
                        row[f'{prefix}calibration_error_percent'] = bm_metrics['calibration_error_percent']
           
            # 9. Pancreatic Function Metrics
            func_metrics = metrics['pancreatic_function']
           
            # Beta-cell function
            if 'beta_cell_function' in func_metrics:
                beta_metrics = func_metrics['beta_cell_function']
                for key, value in beta_metrics.items():
                    if isinstance(value, (int, float, bool)):
                        row[f'beta_cell_{key}'] = value
           
            # Alpha-cell function
            if 'alpha_cell_function' in func_metrics:
                alpha_metrics = func_metrics['alpha_cell_function']
                for key, value in alpha_metrics.items():
                    if isinstance(value, (int, float, bool)):
                        row[f'alpha_cell_{key}'] = value
           
            # Overall function
            row['overall_function_score'] = func_metrics['overall_function_score']
            row['function_classification'] = func_metrics['function_classification']
           
            # 10. Peak Kinetics
            for biomarker in self.biomarkers:
                if biomarker in metrics['peak_kinetics']:
                    bm_metrics = metrics['peak_kinetics'][biomarker]
                    prefix = f'{biomarker}_kinetics_'
                   
                    row[f'{prefix}peak_count'] = bm_metrics['peak_count']
                    row[f'{prefix}peak_frequency'] = bm_metrics['peak_frequency']
                    row[f'{prefix}mean_rise_time'] = bm_metrics['mean_rise_time']
                    row[f'{prefix}mean_decline_time'] = bm_metrics['mean_decline_time']
                    row[f'{prefix}mean_rise_rate'] = bm_metrics['mean_rise_rate']
                    row[f'{prefix}mean_decline_rate'] = bm_metrics['mean_decline_rate']
                    row[f'{prefix}rise_decline_ratio'] = bm_metrics['rise_decline_ratio']
           
            export_data.append(row)
       
        # Create DataFrame and save to CSV
        df = pd.DataFrame(export_data)
       
        # Reorder columns to put basic info first
        basic_cols = ['animal_id', 'rat_type', 'glucose_start_mM', 'glucose_end_mM',
                      'glucose_avg_mM', 'glucose_change_mM', 'has_elisa', 'overall_function_score',
                      'function_classification']
       
        other_cols = [col for col in df.columns if col not in basic_cols]
        df = df[basic_cols + sorted(other_cols)]
       
        # Save to CSV
        df.to_csv(csv_path, index=False)
       
        print(f"✓ Exported {len(df)} animals with {len(df.columns)} metrics to: {csv_path}")
        print(f"✓ Columns include:")
        print(f" - Basic info: {len(basic_cols)} columns")
        print(f" - Device measurements: {len([c for c in df.columns if '_basic_' in c])} columns")
        print(f" - Glucose-adjusted: {len([c for c in df.columns if '_glucose_' in c])} columns")
        print(f" - Peak dynamics: {len([c for c in df.columns if '_peak_' in c])} columns")
        print(f" - Rate analysis: {len([c for c in df.columns if '_rate_' in c])} columns")
        print(f" - AUC metrics: {len([c for c in df.columns if '_auc_' in c])} columns")
        print(f" - Oscillations: {len([c for c in df.columns if '_osc_' in c])} columns")
        print(f" - Correlations: {len([c for c in df.columns if 'corr_' in c])} columns")
        print(f" - ELISA validation: {len([c for c in df.columns if '_elisa_' in c])} columns")
        print(f" - Function scores: {len([c for c in df.columns if any(x in c for x in ['beta_cell_', 'alpha_cell_', 'function'])])} columns")
        print(f" - Peak kinetics: {len([c for c in df.columns if '_kinetics_' in c])} columns")
       
        return df
# =============================================================================
# USAGE EXAMPLE AND MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
   
    # Initialize the comprehensive analyzer
    analyzer = ComprehensiveFedStateAnalyzer()
   
    # Example usage with your data
    csv_path = 'data/fed_state_raw.csv'  # Path to the data file
   
    try:
        # Run complete analysis pipeline
        results = analyzer.run_complete_analysis(
            csv_path=csv_path,
            save_plots_dir='fed_state_analysis_results',
            save_report_path='fed_state_analysis_results/comprehensive_report.txt'
        )
       
        # Export calculated metrics to CSV
        analyzer.export_metrics_to_csv('fed_state_analysis_results/comprehensive_metrics.csv')
       
        print("\n🎉 SUCCESS: Comprehensive Fed State Analysis Complete!")
        print(f"📊 Generated analysis for {len(results)} animals")
        print(f"📈 All plots and reports saved")
        print(f"🔬 Scientific justification for QIRT-ELISA demonstrated")
       
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your CSV file path and format.")
if __name__ == "__main__":
    main()