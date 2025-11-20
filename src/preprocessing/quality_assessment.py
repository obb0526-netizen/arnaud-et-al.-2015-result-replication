"""
Quality Assessment Module for EEG Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class EEGQualityAssessment:
    """Comprehensive quality assessment for EEG data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def assess_raw_data_quality(self, raw: mne.io.Raw, subject: str, events: np.ndarray = None) -> Dict:
        """
        Comprehensive quality assessment of raw EEG data
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data
        subject : str
            Subject identifier
        events : np.ndarray, optional
            Event array for epoching (required for ERP-based SNR computation)
            
        Returns:
        --------
        quality_metrics : Dict
            Dictionary containing various quality metrics
        """
        metrics = {
            'subject': subject,
            'n_channels': raw.info['nchan'],
            'duration_sec': raw.times[-1],
            'sampling_rate': raw.info['sfreq']
        }
        
        # Get EEG data
        data = raw.get_data(picks='eeg')
        
        # Channel-wise quality metrics
        metrics.update(self._assess_channel_quality(data, raw.ch_names))
        
        # Temporal quality metrics
        metrics.update(self._assess_temporal_quality(data))
        
        # Frequency domain analysis
        metrics.update(self._assess_frequency_quality(raw))
        
        # Compute standardized SNR metrics if events are provided
        if events is not None and len(events) > 0:
            # Get ROI channels from config
            frontal_roi = self.config.get('erp_analysis', {}).get('roi', {}).get('frontal', ['F3', 'FZ', 'F4'])
            parietal_roi = self.config.get('erp_analysis', {}).get('roi', {}).get('parieto_occipital', ['P3"', 'PZ"', 'P4"'])
            
            # Compute comprehensive SNR for both ROIs
            frontal_comprehensive = self.compute_comprehensive_snr(raw, events, frontal_roi, subject)
            parietal_comprehensive = self.compute_comprehensive_snr(raw, events, parietal_roi, subject)
            
            # Add SNR metrics to quality report
            metrics.update({
                'frontal_comprehensive': frontal_comprehensive,
                'parietal_comprehensive': parietal_comprehensive,
                'frontal_snr': frontal_comprehensive.get('erp_baseline', 0.0),  # Primary SNR metric
                'parietal_snr': parietal_comprehensive.get('erp_baseline', 0.0)  # Primary SNR metric
            })
        
        # Overall quality score
        metrics['overall_quality_score'] = self._compute_overall_score(metrics)
        
        return metrics
    
    def _assess_channel_quality(self, data: np.ndarray, ch_names: List[str]) -> Dict:
        """Assess quality metrics for each channel"""
        n_channels = data.shape[0]
        
        # Variance across channels
        channel_vars = np.var(data, axis=1)
        
        # Identify flat channels (very low variance)
        flat_threshold = 1e-12  # Very low threshold
        flat_channels = np.where(channel_vars < flat_threshold)[0]
        
        # Identify high-variance channels (potential artifacts)
        high_var_threshold = np.percentile(channel_vars, 95)
        high_var_channels = np.where(channel_vars > high_var_threshold)[0]
        
        # Correlation with neighboring channels
        correlation_matrix = np.corrcoef(data)
        
        # Average correlation per channel (excluding self-correlation)
        avg_correlations = []
        for i in range(n_channels):
            corr_values = correlation_matrix[i, :]
            # Remove self-correlation and NaN values
            corr_values = corr_values[np.arange(len(corr_values)) != i]
            corr_values = corr_values[~np.isnan(corr_values)]
            avg_correlations.append(np.mean(corr_values) if len(corr_values) > 0 else 0)
        
        return {
            'channel_variances': channel_vars.tolist(),
            'flat_channels': [ch_names[i] for i in flat_channels],
            'high_variance_channels': [ch_names[i] for i in high_var_channels],
            'avg_channel_correlations': avg_correlations,
            'n_flat_channels': len(flat_channels),
            'n_high_var_channels': len(high_var_channels)
        }
    
    def _assess_temporal_quality(self, data: np.ndarray) -> Dict:
        """Assess temporal characteristics of the data"""
        
        # Peak-to-peak amplitude
        ptp_amplitudes = np.ptp(data, axis=1)
        
        # Zero crossings (indicator of potential artifacts)
        zero_crossings = []
        for ch_data in data:
            crossings = np.where(np.diff(np.signbit(ch_data)))[0]
            zero_crossings.append(len(crossings))
        
        # Gradient analysis (sudden jumps)
        gradients = np.abs(np.diff(data, axis=1))
        max_gradients = np.max(gradients, axis=1)
        
        return {
            'ptp_amplitudes': ptp_amplitudes.tolist(),
            'zero_crossings_per_channel': zero_crossings,
            'max_gradients': max_gradients.tolist(),
            'mean_ptp_amplitude': np.mean(ptp_amplitudes),
            'mean_zero_crossings': np.mean(zero_crossings)
        }
    
    def _assess_frequency_quality(self, raw: mne.io.Raw) -> Dict:
        """Assess frequency domain characteristics"""
        
        # Compute power spectral density
        psds, freqs = mne.time_frequency.psd_array_welch(
            raw.get_data(picks='eeg'), raw.info['sfreq'], fmin=0.5, fmax=100, n_fft=2048
        )
        
        # Line noise detection (50 Hz and harmonics)
        line_freqs = [50, 100]  # 50 Hz and first harmonic
        line_noise_power = []
        
        for freq in line_freqs:
            freq_idx = np.argmin(np.abs(freqs - freq))
            # Average power in ±1 Hz around line frequency
            freq_range = slice(max(0, freq_idx-2), min(len(freqs), freq_idx+3))
            power_at_freq = np.mean(psds[:, freq_range], axis=1)
            line_noise_power.append(power_at_freq.tolist())
        
        # Alpha power (8-12 Hz)
        alpha_range = (freqs >= 8) & (freqs <= 12)
        alpha_power = np.mean(psds[:, alpha_range], axis=1)
        
        # Beta power (13-30 Hz)
        beta_range = (freqs >= 13) & (freqs <= 30)
        beta_power = np.mean(psds[:, beta_range], axis=1)
        
        # Gamma power (30-100 Hz)
        gamma_range = (freqs >= 30) & (freqs <= 100)
        gamma_power = np.mean(psds[:, gamma_range], axis=1)
        
        return {
            'line_noise_50hz': line_noise_power[0] if line_noise_power else [],
            'line_noise_100hz': line_noise_power[1] if len(line_noise_power) > 1 else [],
            'alpha_power': alpha_power.tolist(),
            'beta_power': beta_power.tolist(),
            'gamma_power': gamma_power.tolist(),
            'mean_line_noise_50hz': np.mean(line_noise_power[0]) if line_noise_power else 0,
            'mean_alpha_power': np.mean(alpha_power)
        }
    
    def _compute_overall_score(self, metrics: Dict) -> float:
        """Compute overall quality score (0-100, higher is better)"""
        
        score = 100.0
        
        # Penalize for flat channels
        if metrics['n_flat_channels'] > 0:
            score -= metrics['n_flat_channels'] * 10
        
        # Penalize for high variance channels
        if metrics['n_high_var_channels'] > 0:
            score -= metrics['n_high_var_channels'] * 5
        
        # Penalize for excessive line noise
        if metrics['mean_line_noise_50hz'] > 1e-10:  # Threshold for significant line noise
            score -= 15
        
        # Penalize for low channel correlations (disconnected electrodes)
        if len(metrics['avg_channel_correlations']) > 0:
            mean_correlation = np.mean(metrics['avg_channel_correlations'])
            if mean_correlation < 0.3:
                score -= 10
        
        return max(0, score)
    
    def compute_comprehensive_snr(self, raw: mne.io.Raw, events: np.ndarray, roi_channels: List[str], subject_id: str) -> Dict:
        """
        Compute comprehensive SNR using ERP-based methods (standardized approach)
        
        This is the standardized SNR computation method that uses:
        - Event-Related Signal Power for signal estimation
        - Pre-stimulus baseline for noise estimation
        - Multiple alternative noise estimation methods
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data
        events : np.ndarray
            Event array for epoching
        roi_channels : List[str]
            List of ROI channel names
        subject_id : str
            Subject identifier for error reporting
            
        Returns:
        --------
        results : Dict
            Dictionary containing comprehensive SNR metrics
        """
        results = {}
        
        try:
            # Method 1: Event-Related Signal Power + Pre-stimulus Baseline
            if events is not None and len(events) > 0:
                # Create epochs
                epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.6, 
                                   baseline=(-0.2, 0), preload=True, verbose=False)
                
                if len(epochs) > 0:
                    # Signal power: ERP variance
                    erp = epochs.average()
                    picks = mne.pick_channels(erp.ch_names, roi_channels)
                    if len(picks) > 0:
                        erp_data = erp.get_data(picks=picks)
                        signal_power_erp = np.var(erp_data, axis=1)
                        
                        # Noise power: pre-stimulus baseline variance
                        baseline_data = epochs.get_data(picks=picks)[:, :, :int(0.2 * epochs.info['sfreq'])]
                        noise_power_baseline = np.var(baseline_data, axis=(0, 2))
                        
                        # SNR computation
                        snr_erp_baseline = 10 * np.log10(signal_power_erp / (noise_power_baseline + 1e-12))
                        results['erp_baseline'] = np.mean(snr_erp_baseline)
                        
                        # Method 2: Event-Related Signal Power + Inter-Trial Variability
                        # Noise power: inter-trial variability
                        itv_noise = np.var(epochs.get_data(picks=picks), axis=0)
                        noise_power_itv = np.mean(itv_noise, axis=1)
                        
                        snr_erp_itv = 10 * np.log10(signal_power_erp / (noise_power_itv + 1e-12))
                        results['erp_itv'] = np.mean(snr_erp_itv)
                        
                        # Method 3: Event-Related Signal Power + High-Frequency Noise
                        raw_copy = raw.copy()
                        raw_copy.filter(l_freq=50, h_freq=100, picks=picks, verbose=False)
                        hf_data = raw_copy.get_data(picks=picks)
                        noise_power_hf = np.var(hf_data, axis=1)
                        
                        snr_erp_hf = 10 * np.log10(signal_power_erp / (noise_power_hf + 1e-12))
                        results['erp_hf'] = np.mean(snr_erp_hf)
            
            # Method 4: Total Signal Variance + Pre-stimulus Baseline
            picks = mne.pick_channels(raw.ch_names, roi_channels)
            if len(picks) > 0:
                roi_data = raw.get_data(picks=picks)
                signal_power_total = np.var(roi_data, axis=1)
                
                if events is not None and len(events) > 0:
                    epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.6, 
                                       baseline=(-0.2, 0), preload=True, verbose=False)
                    if len(epochs) > 0:
                        baseline_data = epochs.get_data(picks=picks)[:, :, :int(0.2 * epochs.info['sfreq'])]
                        noise_power_baseline = np.var(baseline_data, axis=(0, 2))
                        
                        snr_total_baseline = 10 * np.log10(signal_power_total / (noise_power_baseline + 1e-12))
                        results['total_baseline'] = np.mean(snr_total_baseline)
                        
                        # Method 5: Total Signal Variance + Inter-Trial Variability
                        itv_noise = np.var(epochs.get_data(picks=picks), axis=0)
                        noise_power_itv = np.mean(itv_noise, axis=1)
                        
                        snr_total_itv = 10 * np.log10(signal_power_total / (noise_power_itv + 1e-12))
                        results['total_itv'] = np.mean(snr_total_itv)
                        
                        # Method 6: Total Signal Variance + High-Frequency Noise
                        raw_copy = raw.copy()
                        raw_copy.filter(l_freq=50, h_freq=100, picks=picks, verbose=False)
                        hf_data = raw_copy.get_data(picks=picks)
                        noise_power_hf = np.var(hf_data, axis=1)
                        
                        snr_total_hf = 10 * np.log10(signal_power_total / (noise_power_hf + 1e-12))
                        results['total_hf'] = np.mean(snr_total_hf)
        
        except Exception as e:
            self.logger.error(f"Error computing comprehensive SNR for {subject_id}: {e}")
        
        return results

    def compute_snr_roi(self, raw: mne.io.Raw, roi_channels: List[str], 
                       events: np.ndarray = None, subject_id: str = "unknown") -> Dict:
        """
        Compute Signal-to-Noise Ratio for ROI channels using standardized ERP-based method
        
        This is the standardized SNR computation method that prioritizes ERP analysis.
        It uses the comprehensive SNR computation with ERP signal power and pre-stimulus baseline.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            EEG data
        roi_channels : List[str]
            List of ROI channel names
        events : np.ndarray, optional
            Event array for epoching (required for ERP-based SNR)
        subject_id : str
            Subject identifier for error reporting
            
        Returns:
        --------
        snr_metrics : Dict
            SNR metrics for ROI using standardized ERP-based method
        """
        
        # Use the comprehensive SNR method as the standard
        comprehensive_results = self.compute_comprehensive_snr(raw, events, roi_channels, subject_id)
        
        # For backward compatibility, extract the primary ERP baseline SNR
        primary_snr = comprehensive_results.get('erp_baseline', 0.0)
        
        found_channels = [ch for ch in roi_channels if ch in raw.ch_names]
        
        return {
            'roi_snr': primary_snr,  # Primary SNR metric (ERP baseline)
            'roi_snr_per_channel': {ch: primary_snr for ch in found_channels},
            'roi_channels_found': found_channels,
            'comprehensive_snr': comprehensive_results,  # Full comprehensive results
            'method': 'erp_baseline'  # Indicate the method used
        }
    
    def select_subjects_for_analysis(self, quality_reports: List[Dict], 
                                   target_n: int = 10) -> Tuple[List[str], str]:
        """
        Enhanced subject selection with ROI SNR tie-breaking for subjects with same quality score
        
        Parameters:
        -----------
        quality_reports : List[Dict]
            Quality assessment reports for all subjects
        target_n : int
            Target number of subjects to select
            
        Returns:
        --------
        selected_subjects : List[str]
            List of selected subject IDs
        manual_ica_subject : str
            Subject selected for manual ICA review
        """
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(quality_reports)
        
        # Check if we have comprehensive SNR metrics available
        has_comprehensive_snr = ('frontal_comprehensive' in df.columns and 
                                'parietal_comprehensive' in df.columns)
        
        if has_comprehensive_snr:
            # Extract ERP signal power / pre-stimulus baseline SNR for tie-breaking
            def extract_erp_baseline_snr(row):
                """Extract ERP baseline SNR from comprehensive metrics"""
                try:
                    # Try frontal ROI first
                    if (row['frontal_comprehensive'] and 
                        'erp_baseline' in row['frontal_comprehensive']):
                        return row['frontal_comprehensive']['erp_baseline']
                    # Fallback to parietal ROI
                    elif (row['parietal_comprehensive'] and 
                          'erp_baseline' in row['parietal_comprehensive']):
                        return row['parietal_comprehensive']['erp_baseline']
                    else:
                        return 0.0
                except (KeyError, TypeError):
                    return 0.0
            
            # Add ERP baseline SNR column for tie-breaking
            df['erp_baseline_snr'] = df.apply(extract_erp_baseline_snr, axis=1)
            
            # Check for tie-breaking scenario: more than 4 subjects with same lowest quality score
            quality_score_counts = df['overall_quality_score'].value_counts()
            lowest_score = df['overall_quality_score'].min()
            lowest_score_count = quality_score_counts.get(lowest_score, 0)
            
            if lowest_score_count > 4:
                self.logger.info(f"Tie-breaking scenario detected: {lowest_score_count} subjects with quality score {lowest_score}")
                self.logger.info("Using ERP Signal Power / Pre-stimulus Baseline SNR for tie-breaking")
                
                # Enhanced sorting with ERP baseline SNR tie-breaking
                df_sorted = df.sort_values([
                    'overall_quality_score',    # Primary: quality score (descending)
                    'erp_baseline_snr',         # Secondary: ERP baseline SNR (descending)
                    'n_flat_channels',          # Tertiary: fewer bad channels (ascending)
                    'n_high_var_channels',      # Quaternary: fewer bad channels (ascending)
                    'subject'                   # Quinary: consistent ordering (ascending)
                ], ascending=[False, False, True, True, True])
            else:
                # Standard sorting without tie-breaking
                df_sorted = df.sort_values([
                    'overall_quality_score',
                    'subject'
                ], ascending=[False, True])
        else:
            # Fallback to basic ROI SNR if comprehensive metrics not available
            has_basic_snr = 'frontal_snr' in df.columns and 'parietal_snr' in df.columns
            
            if has_basic_snr:
                # Calculate overall ROI SNR as average of frontal and parietal
                df['overall_roi_snr'] = (df['frontal_snr'] + df['parietal_snr']) / 2
                
                # Check for tie-breaking scenario
                quality_score_counts = df['overall_quality_score'].value_counts()
                lowest_score = df['overall_quality_score'].min()
                lowest_score_count = quality_score_counts.get(lowest_score, 0)
                
                if lowest_score_count > 4:
                    self.logger.info(f"Tie-breaking scenario detected: {lowest_score_count} subjects with quality score {lowest_score}")
                    self.logger.info("Using basic ROI SNR metrics for tie-breaking (comprehensive metrics not available)")
                    
                    # Basic sorting with ROI SNR tie-breaking
                    df_sorted = df.sort_values([
                        'overall_quality_score',  # Primary: quality score (descending)
                        'overall_roi_snr',        # Secondary: ROI SNR (descending)
                        'frontal_snr',            # Tertiary: frontal SNR (descending)
                        'parietal_snr',           # Quaternary: parietal SNR (descending)
                        'n_flat_channels',        # Quinary: fewer bad channels (ascending)
                        'n_high_var_channels',    # Senary: fewer bad channels (ascending)
                        'subject'                 # Septenary: consistent ordering (ascending)
                    ], ascending=[False, False, False, False, True, True, True])
                else:
                    # Standard sorting without tie-breaking
                    df_sorted = df.sort_values([
                        'overall_quality_score',
                        'subject'
                    ], ascending=[False, True])
            else:
                # Fallback to standard sorting if no SNR metrics available
                df_sorted = df.sort_values([
                    'overall_quality_score',
                    'subject'
                ], ascending=[False, True])
        
        # Select top N subjects
        selected_subjects = df_sorted.head(target_n)['subject'].tolist()
        
        # Select subject for manual ICA (median quality among selected)
        median_idx = len(selected_subjects) // 2
        manual_ica_subject = selected_subjects[median_idx]
        
        # Log selection details
        self.logger.info(f"Selected {len(selected_subjects)} subjects for analysis")
        self.logger.info(f"Subject {manual_ica_subject} selected for manual ICA review")
        
        if has_comprehensive_snr and lowest_score_count > 4:
            selected_df = df_sorted.head(target_n)
            self.logger.info(f"ERP baseline SNR tie-breaking applied: {lowest_score_count} subjects with score {lowest_score}")
            self.logger.info(f"Selected subjects quality scores: {selected_df['overall_quality_score'].tolist()}")
            self.logger.info(f"Selected subjects ERP baseline SNR: {selected_df['erp_baseline_snr'].round(2).tolist()}")
        elif has_basic_snr and lowest_score_count > 4:
            selected_df = df_sorted.head(target_n)
            self.logger.info(f"Basic ROI SNR tie-breaking applied: {lowest_score_count} subjects with score {lowest_score}")
            self.logger.info(f"Selected subjects quality scores: {selected_df['overall_quality_score'].tolist()}")
            self.logger.info(f"Selected subjects ROI SNR: {selected_df['overall_roi_snr'].round(2).tolist()}")
        
        return selected_subjects, manual_ica_subject
    
    def plot_quality_summary(self, quality_reports: List[Dict], 
                           selected_subjects: List[str], 
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive quality assessment visualization
        
        Parameters:
        -----------
        quality_reports : List[Dict]
            Quality assessment reports
        selected_subjects : List[str]
            List of selected subjects
        save_path : Path, optional
            Path to save the figure
            
        Returns:
        --------
        fig : plt.Figure
            Quality summary figure
        """
        
        df = pd.DataFrame(quality_reports)
        df['selected'] = df['subject'].isin(selected_subjects)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EEG Data Quality Assessment Summary', fontsize=16, fontweight='bold')
        
        # Overall quality scores
        ax = axes[0, 0]
        colors = ['green' if selected else 'red' for selected in df['selected']]
        bars = ax.bar(range(len(df)), df['overall_quality_score'], color=colors, alpha=0.7)
        ax.set_xlabel('Subject Index')
        ax.set_ylabel('Overall Quality Score')
        ax.set_title('Overall Quality Scores')
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        ax.legend()
        
        # Channel variance distribution
        ax = axes[0, 1]
        selected_vars = [np.mean(report['channel_variances']) for report in quality_reports 
                        if report['subject'] in selected_subjects]
        excluded_vars = [np.mean(report['channel_variances']) for report in quality_reports 
                        if report['subject'] not in selected_subjects]
        
        if selected_vars:
            ax.hist(selected_vars, alpha=0.7, label='Selected', color='green', bins=10)
        if excluded_vars:
            ax.hist(excluded_vars, alpha=0.7, label='Excluded', color='red', bins=10)
        ax.set_xlabel('Mean Channel Variance')
        ax.set_ylabel('Count')
        ax.set_title('Channel Variance Distribution')
        ax.legend()
        
        # Line noise comparison
        ax = axes[0, 2]
        selected_noise = [report['mean_line_noise_50hz'] for report in quality_reports 
                         if report['subject'] in selected_subjects]
        excluded_noise = [report['mean_line_noise_50hz'] for report in quality_reports 
                         if report['subject'] not in selected_subjects]
        
        ax.boxplot([selected_noise, excluded_noise], labels=['Selected', 'Excluded'])
        ax.set_ylabel('50Hz Line Noise Power')
        ax.set_title('Line Noise Comparison')
        
        # Bad channels per subject
        ax = axes[1, 0]
        bad_channels = [report['n_flat_channels'] + report['n_high_var_channels'] 
                       for report in quality_reports]
        colors = ['green' if selected else 'red' for selected in df['selected']]
        ax.bar(range(len(df)), bad_channels, color=colors, alpha=0.7)
        ax.set_xlabel('Subject Index')
        ax.set_ylabel('Number of Bad Channels')
        ax.set_title('Bad Channels per Subject')
        
        # Alpha power distribution
        ax = axes[1, 1]
        selected_alpha = [np.mean(report['alpha_power']) for report in quality_reports 
                         if report['subject'] in selected_subjects]
        excluded_alpha = [np.mean(report['alpha_power']) for report in quality_reports 
                         if report['subject'] not in selected_subjects]
        
        if selected_alpha:
            ax.hist(selected_alpha, alpha=0.7, label='Selected', color='green', bins=10)
        if excluded_alpha:
            ax.hist(excluded_alpha, alpha=0.7, label='Excluded', color='red', bins=10)
        ax.set_xlabel('Mean Alpha Power')
        ax.set_ylabel('Count')
        ax.set_title('Alpha Power Distribution')
        ax.legend()
        
        # Correlation quality
        ax = axes[1, 2]
        selected_corr = [np.mean(report['avg_channel_correlations']) for report in quality_reports 
                        if report['subject'] in selected_subjects]
        excluded_corr = [np.mean(report['avg_channel_correlations']) for report in quality_reports 
                        if report['subject'] not in selected_subjects]
        
        ax.boxplot([selected_corr, excluded_corr], labels=['Selected', 'Excluded'])
        ax.set_ylabel('Mean Channel Correlation')
        ax.set_title('Channel Correlation Quality')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups"""
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def compute_snr_before_after(raw_before: mne.io.Raw, raw_after: mne.io.Raw, 
                           roi_channels: List[str]) -> Dict:
    """
    Compare SNR before and after preprocessing
    
    Parameters:
    -----------
    raw_before : mne.io.Raw
        Raw data before preprocessing
    raw_after : mne.io.Raw  
        Raw data after preprocessing
    roi_channels : List[str]
        ROI channel names
        
    Returns:
    --------
    snr_comparison : Dict
        SNR comparison metrics
    """
    
    qa = EEGQualityAssessment({})
    
    snr_before = qa.compute_snr_roi(raw_before, roi_channels, method='processed')
    snr_after = qa.compute_snr_roi(raw_after, roi_channels, method='processed')
    
    snr_improvement = snr_after['roi_snr'] - snr_before['roi_snr']
    
    return {
        'snr_before': snr_before['roi_snr'],
        'snr_after': snr_after['roi_snr'],
        'snr_improvement_db': snr_improvement,
        'snr_improvement_ratio': snr_after['roi_snr'] / snr_before['roi_snr'] if snr_before['roi_snr'] > 0 else float('inf'),
        'channels_analyzed': snr_before['roi_channels_found']
    }
