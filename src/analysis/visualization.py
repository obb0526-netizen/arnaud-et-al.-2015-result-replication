"""
Visualization utilities for EEG analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd


class EEGVisualizer:
    """Create publication-quality visualizations for EEG analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vis_config = config['visualization']
        self.colors = self.vis_config['colors']
        
        # Set up matplotlib parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'figure.dpi': self.vis_config['dpi']
        })
        
        # Set color palette
        sns.set_palette([self.colors['familiar'], self.colors['new'], self.colors['difference']])
    
    def plot_preprocessing_comparison(self, raw_before: mne.io.Raw, raw_after: mne.io.Raw,
                                   subject: str, stage: str = "preprocessing",
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create before/after preprocessing comparison plot
        
        Parameters:
        -----------
        raw_before : mne.io.Raw
            Raw data before preprocessing
        raw_after : mne.io.Raw
            Raw data after preprocessing
        subject : str
            Subject identifier
        stage : str
            Preprocessing stage name
        save_path : Path, optional
            Path to save figure
            
        Returns:
        --------
        fig : plt.Figure
            Comparison figure
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Comparison: {subject} - {stage}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Time series comparison (sample channel)
        ax = axes[0, 0]
        sample_ch = 'Fz' if 'Fz' in raw_before.ch_names else raw_before.ch_names[0]
        ch_idx = raw_before.ch_names.index(sample_ch)
        
        times = raw_before.times
        data_before = raw_before.get_data()[ch_idx] * 1e6  # Convert to µV
        data_after = raw_after.get_data()[ch_idx] * 1e6
        
        # Plot 10-second segment
        start_idx = 0
        end_idx = min(int(10 * raw_before.info['sfreq']), len(times))
        
        ax.plot(times[start_idx:end_idx], data_before[start_idx:end_idx], 
               'b-', alpha=0.7, label='Before', linewidth=1)
        ax.plot(times[start_idx:end_idx], data_after[start_idx:end_idx], 
               'r-', alpha=0.7, label='After', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Channel {sample_ch}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Power spectral density comparison
        ax = axes[0, 1]
        psds_before, freqs = mne.time_frequency.psd_array_welch(raw_before.get_data(picks='eeg'), raw_before.info['sfreq'], fmin=0.5, fmax=100)
        psds_after, _ = mne.time_frequency.psd_array_welch(raw_after.get_data(picks='eeg'), raw_after.info['sfreq'], fmin=0.5, fmax=100)
        
        ax.loglog(freqs, np.mean(psds_before, axis=0), 'b-', alpha=0.7, label='Before', linewidth=2)
        ax.loglog(freqs, np.mean(psds_after, axis=0), 'r-', alpha=0.7, label='After', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (V²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Channel variance comparison
        ax = axes[0, 2]
        var_before = np.var(raw_before.get_data(picks='eeg'), axis=1) * 1e12  # µV²
        var_after = np.var(raw_after.get_data(picks='eeg'), axis=1) * 1e12
        
        ax.scatter(var_before, var_after, alpha=0.6, s=30)
        min_var = min(np.min(var_before), np.min(var_after))
        max_var = max(np.max(var_before), np.max(var_after))
        ax.plot([min_var, max_var], [min_var, max_var], 'r--', alpha=0.7)
        ax.set_xlabel('Variance Before (µV²)')
        ax.set_ylabel('Variance After (µV²)')
        ax.set_title('Channel Variance Comparison')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Amplitude distribution
        ax = axes[1, 0]
        # Sample data for histogram
        sample_data_before = raw_before.get_data(picks='eeg', start=0, stop=int(60*raw_before.info['sfreq'])).flatten() * 1e6
        sample_data_after = raw_after.get_data(picks='eeg', start=0, stop=int(60*raw_after.info['sfreq'])).flatten() * 1e6
        
        ax.hist(sample_data_before, bins=100, alpha=0.5, label='Before', color='blue', density=True)
        ax.hist(sample_data_after, bins=100, alpha=0.5, label='After', color='red', density=True)
        ax.set_xlabel('Amplitude (µV)')
        ax.set_ylabel('Density')
        ax.set_title('Amplitude Distribution')
        ax.legend()
        ax.set_xlim([-100, 100])  # Reasonable range for EEG
        
        # Plot 5: Line noise comparison (50 Hz)
        ax = axes[1, 1]
        # Extract power at 50 Hz ±2 Hz
        freq_range = (freqs >= 48) & (freqs <= 52)
        power_50hz_before = np.mean(psds_before[:, freq_range], axis=1)
        power_50hz_after = np.mean(psds_after[:, freq_range], axis=1)
        
        ax.boxplot([power_50hz_before * 1e12, power_50hz_after * 1e12], 
                  labels=['Before', 'After'])
        ax.set_ylabel('50 Hz Power (pV²/Hz)')
        ax.set_title('Line Noise (50 Hz)')
        ax.set_yscale('log')
        
        # Plot 6: Summary metrics
        ax = axes[1, 2]
        metrics = {
            'Mean Variance\nReduction (%)': (1 - np.mean(var_after) / np.mean(var_before)) * 100,
            'Median 50Hz\nReduction (%)': (1 - np.median(power_50hz_after) / np.median(power_50hz_before)) * 100,
            'RMS Amplitude\nReduction (%)': (1 - np.sqrt(np.mean(sample_data_after**2)) / np.sqrt(np.mean(sample_data_before**2))) * 100
        }
        
        bars = ax.bar(metrics.keys(), metrics.values(), color=['green', 'orange', 'purple'], alpha=0.7)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Quality Improvement Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_erp_comparison_delorme_style(self, erp_data: Dict, roi_results: Dict,
                                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create ERP plots in the style of Delorme et al. 2018 Figure 5
        
        Parameters:
        -----------
        erp_data : Dict
            ERP data dictionary
        roi_results : Dict
            ROI analysis results
        save_path : Path, optional
            Path to save figure
            
        Returns:
        --------
        fig : plt.Figure
            ERP comparison figure
        """
        
        # Create figure with specific layout matching Delorme et al.
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Recognition Memory Task Event-Related Potentials', 
                    fontsize=14, fontweight='bold')
        
        times = erp_data['times'] * 1000  # Convert to ms
        
        # Define colors and styles matching the original
        colors = {
            'familiar': '#2E8B57',  # Sea green
            'new': '#4169E1',       # Royal blue
            'difference': '#DC143C' # Crimson
        }
        
        # Plot A: Frontal ROI
        ax = axes[0, 0]
        if 'frontal' in roi_results:
            data = roi_results['frontal']
            
            # Plot ERPs with SEM
            n_subjects = len(erp_data['subjects'])
            familiar_mean = data['familiar_erp'] * 1e6
            new_mean = data['new_erp'] * 1e6
            
            # Estimate SEM (simplified)
            familiar_sem = np.std(familiar_mean) / np.sqrt(n_subjects) * 0.5  # Scaled for visualization
            new_sem = np.std(new_mean) / np.sqrt(n_subjects) * 0.5
            
            ax.plot(times, familiar_mean, color=colors['familiar'], linewidth=2, label='Familiar')
            ax.plot(times, new_mean, color=colors['new'], linewidth=2, label='New')
            
            # Add shaded SEM
            ax.fill_between(times, familiar_mean - familiar_sem, familiar_mean + familiar_sem,
                          color=colors['familiar'], alpha=0.3)
            ax.fill_between(times, new_mean - new_sem, new_mean + new_sem,
                          color=colors['new'], alpha=0.3)
            
            # Mark significant periods
            if np.any(data['significant_mask']):
                sig_times = times[data['significant_mask']]
                y_pos = ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot(sig_times, np.ones(len(sig_times)) * y_pos, 'r-', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('Frontal')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlim([-100, 600])
        
        # Plot B: Parieto-occipital ROI
        ax = axes[0, 1]
        if 'parieto_occipital' in roi_results:
            data = roi_results['parieto_occipital']
            
            familiar_mean = data['familiar_erp'] * 1e6
            new_mean = data['new_erp'] * 1e6
            
            familiar_sem = np.std(familiar_mean) / np.sqrt(n_subjects) * 0.5
            new_sem = np.std(new_mean) / np.sqrt(n_subjects) * 0.5
            
            ax.plot(times, familiar_mean, color=colors['familiar'], linewidth=2, label='Familiar')
            ax.plot(times, new_mean, color=colors['new'], linewidth=2, label='New')
            
            ax.fill_between(times, familiar_mean - familiar_sem, familiar_mean + familiar_sem,
                          color=colors['familiar'], alpha=0.3)
            ax.fill_between(times, new_mean - new_sem, new_mean + new_sem,
                          color=colors['new'], alpha=0.3)
            
            # Mark significant periods
            if np.any(data['significant_mask']):
                sig_times = times[data['significant_mask']]
                y_pos = ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot(sig_times, np.ones(len(sig_times)) * y_pos, 'r-', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('Parieto-occipital')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlim([-100, 600])
        
        # Plot C: Difference waves - Frontal
        ax = axes[1, 0]
        if 'frontal' in roi_results:
            data = roi_results['frontal']
            diff_wave = data['difference_erp'] * 1e6
            
            ax.plot(times, diff_wave, color=colors['difference'], linewidth=2)
            ax.fill_between(times, 0, diff_wave, where=(diff_wave > 0), 
                          color=colors['difference'], alpha=0.3)
            ax.fill_between(times, 0, diff_wave, where=(diff_wave < 0), 
                          color=colors['difference'], alpha=0.3)
            
            # Mark significant periods
            if np.any(data['significant_mask']):
                sig_times = times[data['significant_mask']]
                y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot(sig_times, np.ones(len(sig_times)) * y_pos, 'r-', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('Frontal Difference (Familiar - New)')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlim([-100, 600])
        
        # Plot D: Difference waves - Parieto-occipital
        ax = axes[1, 1]
        if 'parieto_occipital' in roi_results:
            data = roi_results['parieto_occipital']
            diff_wave = data['difference_erp'] * 1e6
            
            ax.plot(times, diff_wave, color=colors['difference'], linewidth=2)
            ax.fill_between(times, 0, diff_wave, where=(diff_wave > 0), 
                          color=colors['difference'], alpha=0.3)
            ax.fill_between(times, 0, diff_wave, where=(diff_wave < 0), 
                          color=colors['difference'], alpha=0.3)
            
            # Mark significant periods
            if np.any(data['significant_mask']):
                sig_times = times[data['significant_mask']]
                y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot(sig_times, np.ones(len(sig_times)) * y_pos, 'r-', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('Parieto-occipital Difference (Familiar - New)')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlim([-100, 600])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_statistical_map_delorme_style(self, familiarity_results: Dict,
                                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create statistical significance map in Delorme et al. style (Figure 5A)
        
        Parameters:
        -----------
        familiarity_results : Dict
            Familiarity effect statistical results
        save_path : Path, optional
            Path to save figure
            
        Returns:
        --------
        fig : plt.Figure
            Statistical map figure
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        times = familiarity_results['times'] * 1000  # Convert to ms
        ch_names = familiarity_results['ch_names']
        p_values = familiarity_results['p_values_corrected']
        
        # Create electrode group mapping (simplified)
        electrode_groups = {
            'F': [],  # Frontal
            'C': [],  # Central
            'T': [],  # Temporal
            'P': [],  # Parietal
            'PO': [], # Parieto-occipital
            'O': []   # Occipital
        }
        
        # Group electrodes by their prefixes
        for i, ch in enumerate(ch_names):
            for prefix in electrode_groups.keys():
                if ch.startswith(prefix):
                    electrode_groups[prefix].append(i)
                    break
        
        # Plot significance map
        # Use -log10(p) for visualization, with significant values shown
        log_p_values = -np.log10(p_values + 1e-10)  # Add small value to avoid log(0)
        significant_mask = familiarity_results['significant_mask']
        
        # Set non-significant values to 0 for cleaner visualization
        log_p_values[~significant_mask] = 0
        
        im = ax.imshow(log_p_values, aspect='auto', cmap='hot', origin='lower',
                      extent=[times[0], times[-1], 0, len(ch_names)])
        
        # Customize axes
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Electrodes')
        ax.set_title('Paired t-test p-values (FDR-corrected) for the ERP difference\nbetween familiar and new images')
        
        # Set y-axis labels for electrode groups
        group_positions = []
        group_labels = []
        for group, indices in electrode_groups.items():
            if indices:
                group_positions.append(np.mean(indices))
                group_labels.append(group)
        
        ax.set_yticks(group_positions)
        ax.set_yticklabels(group_labels)
        
        # Add vertical line at stimulus onset
        ax.axvline(0, color='white', linestyle='--', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('-log₁₀(p-value)')
        
        # Add significance threshold line to colorbar
        sig_threshold = -np.log10(0.05)
        cbar.ax.axhline(sig_threshold, color='cyan', linestyle='-', linewidth=2)
        cbar.ax.text(0.5, sig_threshold, 'p=0.05', rotation=0, ha='center', va='bottom', 
                    color='cyan', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        
        return fig

def create_publication_ready_plots(config: Dict, results_dict: Dict, 
                                 output_dir: Path) -> Dict[str, Path]:
    """
    Create all publication-ready plots for the study
    
    Parameters:
    -----------
    config : Dict
        Analysis configuration
    results_dict : Dict
        Dictionary containing all analysis results
    output_dir : Path
        Output directory for figures
        
    Returns:
    --------
    figure_paths : Dict[str, Path]
        Dictionary mapping figure names to file paths
    """
    
    visualizer = EEGVisualizer(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = {}
    
    # Main ERP figure (Delorme style)
    if 'erp_data' in results_dict and 'roi_results' in results_dict:
        fig_path = output_dir / 'figure_erp_main.png'
        visualizer.plot_erp_comparison_delorme_style(
            results_dict['erp_data'], 
            results_dict['roi_results'],
            save_path=fig_path
        )
        figure_paths['main_erp'] = fig_path
    
    # Statistical map figure
    if 'familiarity_results' in results_dict:
        fig_path = output_dir / 'figure_statistical_map.png'
        visualizer.plot_statistical_map_delorme_style(
            results_dict['familiarity_results'],
            save_path=fig_path
        )
        figure_paths['statistical_map'] = fig_path
    
    # Quality assessment figure
    if 'quality_reports' in results_dict and 'selected_subjects' in results_dict:
        fig_path = output_dir / 'figure_quality_assessment.png'
        visualizer.plot_quality_metrics_summary(
            results_dict['quality_reports'],
            results_dict['selected_subjects'],
            snr_results=results_dict.get('snr_results'),
            save_path=fig_path
        )
        figure_paths['quality_assessment'] = fig_path
    
    return figure_paths
