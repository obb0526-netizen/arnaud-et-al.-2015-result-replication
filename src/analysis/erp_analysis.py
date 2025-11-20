"""
ERP Analysis Module for Memory Recognition Study
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import stats
from mne.stats import permutation_cluster_test, fdr_correction
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


class ERPAnalyzer:
    """Analyze Event-Related Potentials for memory recognition study"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.erp_config = config['erp_analysis']
        self.roi_config = self.erp_config['roi']
        self.stats_config = self.erp_config['statistics']
        self.vis_config = config['visualization']
        self.logger = logging.getLogger(__name__)
        
    def compute_erps(self, epochs_list: List[mne.Epochs], subjects: List[str]) -> Dict:
        """
        Compute ERPs for all subjects and conditions
        
        Parameters:
        -----------
        epochs_list : List[mne.Epochs]
            List of epochs for each subject
        subjects : List[str]
            List of subject identifiers
            
        Returns:
        --------
        erp_data : Dict
            Dictionary containing ERP data for all subjects and conditions
        """
        
        self.logger.info("Computing ERPs for all subjects and conditions")
        
        erp_data = {
            'subjects': subjects,
            'conditions': {},
            'times': None,
            'ch_names': None
        }
        
        # Initialize condition dictionaries
        conditions = ['familiar', 'new', 'familiar_animal', 'familiar_non_animal', 
                     'new_animal', 'new_non_animal']
        
        for condition in conditions:
            erp_data['conditions'][condition] = []
        
        # Process each subject
        for subject, epochs in zip(subjects, epochs_list):
            self.logger.info(f"Computing ERPs for {subject}")
            
            # Set times and channel names from first subject
            if erp_data['times'] is None:
                erp_data['times'] = epochs.times
                erp_data['ch_names'] = epochs.ch_names
            
            # Compute ERPs for each condition
            for condition in conditions:
                try:
                    if condition == 'familiar':
                        # Combine familiar animal and non-animal
                        epochs_cond = epochs['familiar_animal', 'familiar_non_animal']
                    elif condition == 'new':
                        # Combine new animal and non-animal
                        epochs_cond = epochs['new_animal', 'new_non_animal']
                    else:
                        # Individual conditions
                        epochs_cond = epochs[condition]
                    
                    # Compute average ERP
                    erp = epochs_cond.average()
                    erp_data['conditions'][condition].append(erp.data)
                    
                except KeyError:
                    self.logger.warning(f"Condition {condition} not found for {subject}")
                    # Add zeros if condition is missing
                    erp_data['conditions'][condition].append(
                        np.zeros((len(erp_data['ch_names']), len(erp_data['times'])))
                    )
        
        # Convert lists to arrays
        for condition in conditions:
            erp_data['conditions'][condition] = np.array(erp_data['conditions'][condition])
        
        self.logger.info("ERP computation completed")
        return erp_data
    
    def test_familiarity_effect(self, erp_data: Dict) -> Dict:
        """
        Test for familiarity effect (familiar vs new) across all electrodes
        
        Parameters:
        -----------
        erp_data : Dict
            ERP data dictionary
            
        Returns:
        --------
        familiarity_results : Dict
            Statistical test results for familiarity effect
        """
        
        self.logger.info("Testing familiarity effect across all electrodes")
        
        familiar_erps = erp_data['conditions']['familiar']
        new_erps = erp_data['conditions']['new']
        times = erp_data['times']
        ch_names = erp_data['ch_names']
        
        n_subjects, n_channels, n_times = familiar_erps.shape
        
        # Initialize results
        t_values = np.zeros((n_channels, n_times))
        p_values = np.ones((n_channels, n_times))
        
        # Perform t-tests at each channel and time point
        for ch in range(n_channels):
            for tp in range(n_times):
                t_val, p_val = stats.ttest_rel(
                    familiar_erps[:, ch, tp], 
                    new_erps[:, ch, tp]
                )
                t_values[ch, tp] = t_val
                p_values[ch, tp] = p_val
        
        # Apply FDR correction
        if self.stats_config['fdr_correction']:
            p_values_corrected = np.ones_like(p_values)
            for ch in range(n_channels):
                _, p_values_corrected[ch, :] = fdr_correction(p_values[ch, :], 
                                                            alpha=self.stats_config['alpha'])
        else:
            p_values_corrected = p_values
        
        # Find significant effects
        significant_mask = p_values_corrected < self.stats_config['alpha']
        
        familiarity_results = {
            'test_name': 'familiarity_effect_all_electrodes',
            't_values': t_values,
            'p_values': p_values,
            'p_values_corrected': p_values_corrected,
            'significant_mask': significant_mask,
            'times': times,
            'ch_names': ch_names,
            'n_significant_tests': np.sum(significant_mask),
            'earliest_significant_time': None,
            'peak_effect_time': None,
            'peak_effect_channel': None
        }
        
        # Find earliest significant effect
        if np.any(significant_mask):
            sig_times = times[np.any(significant_mask, axis=0)]
            if len(sig_times) > 0:
                familiarity_results['earliest_significant_time'] = sig_times[0]
            
            # Find peak effect
            max_t_idx = np.unravel_index(np.argmax(np.abs(t_values)), t_values.shape)
            familiarity_results['peak_effect_time'] = times[max_t_idx[1]]
            familiarity_results['peak_effect_channel'] = ch_names[max_t_idx[0]]
        
        self.logger.info(f"Familiarity effect analysis completed: "
                        f"{familiarity_results['n_significant_tests']} significant tests")
        
        return familiarity_results
    
    def test_roi_effects(self, erp_data: Dict) -> Dict:
        """
        Test effects in regions of interest (frontal and parieto-occipital)
        
        Parameters:
        -----------
        erp_data : Dict
            ERP data dictionary
            
        Returns:
        --------
        roi_results : Dict
            ROI analysis results
        """
        
        self.logger.info("Testing effects in regions of interest")
        
        times = erp_data['times']
        ch_names = erp_data['ch_names']
        
        roi_results = {}
        
        # Test each ROI
        for roi_name, roi_channels in self.roi_config.items():
            self.logger.info(f"Analyzing {roi_name} ROI")
            
            # Find channel indices for this ROI
            roi_indices = []
            for ch in roi_channels:
                if ch in ch_names:
                    roi_indices.append(ch_names.index(ch))
            
            if not roi_indices:
                self.logger.warning(f"No channels found for {roi_name} ROI")
                continue
            
            # Extract ROI data (average across channels)
            familiar_roi = np.mean(erp_data['conditions']['familiar'][:, roi_indices, :], axis=1)
            new_roi = np.mean(erp_data['conditions']['new'][:, roi_indices, :], axis=1)
            
            # Perform statistical test
            t_values, p_values = [], []
            for tp in range(len(times)):
                t_val, p_val = stats.ttest_rel(familiar_roi[:, tp], new_roi[:, tp])
                t_values.append(t_val)
                p_values.append(p_val)
            
            t_values = np.array(t_values)
            p_values = np.array(p_values)
            
            # Apply FDR correction
            if self.stats_config['fdr_correction']:
                _, p_values_corrected = fdr_correction(p_values, alpha=self.stats_config['alpha'])
            else:
                p_values_corrected = p_values
            
            # Find significant time windows
            significant_mask = p_values_corrected < self.stats_config['alpha']
            
            roi_results[roi_name] = {
                'channels': [ch_names[i] for i in roi_indices],
                'familiar_erp': np.mean(familiar_roi, axis=0),
                'new_erp': np.mean(new_roi, axis=0),
                'difference_erp': np.mean(familiar_roi - new_roi, axis=0),
                't_values': t_values,
                'p_values': p_values,
                'p_values_corrected': p_values_corrected,
                'significant_mask': significant_mask,
                'times': times
            }
            
            # Find significant time windows
            if np.any(significant_mask):
                sig_indices = np.where(significant_mask)[0]
                roi_results[roi_name]['significant_time_windows'] = [
                    (times[sig_indices[0]], times[sig_indices[-1]])
                ]
                roi_results[roi_name]['earliest_significant_time'] = times[sig_indices[0]]
                roi_results[roi_name]['peak_effect_time'] = times[np.argmax(np.abs(t_values))]
            else:
                roi_results[roi_name]['significant_time_windows'] = []
                roi_results[roi_name]['earliest_significant_time'] = None
                roi_results[roi_name]['peak_effect_time'] = None
        
        self.logger.info("ROI effects analysis completed")
        return roi_results
    
    def test_category_effects(self, erp_data: Dict) -> Dict:
        """
        Test for category effects (animal vs non-animal)
        
        Parameters:
        -----------
        erp_data : Dict
            ERP data dictionary
            
        Returns:
        --------
        category_results : Dict
            Category effects analysis results
        """
        
        self.logger.info("Testing category effects")
        
        times = erp_data['times']
        ch_names = erp_data['ch_names']
        
        category_results = {}
        
        # Test each ROI for category effects
        for roi_name, roi_channels in self.roi_config.items():
            self.logger.info(f"Testing category effects in {roi_name} ROI")
            
            # Find channel indices for this ROI
            roi_indices = []
            for ch in roi_channels:
                if ch in ch_names:
                    roi_indices.append(ch_names.index(ch))
            
            if not roi_indices:
                continue
            
            # Compute familiarity differences for each category
            familiar_animal = np.mean(erp_data['conditions']['familiar_animal'][:, roi_indices, :], axis=1)
            new_animal = np.mean(erp_data['conditions']['new_animal'][:, roi_indices, :], axis=1)
            familiar_non_animal = np.mean(erp_data['conditions']['familiar_non_animal'][:, roi_indices, :], axis=1)
            new_non_animal = np.mean(erp_data['conditions']['new_non_animal'][:, roi_indices, :], axis=1)
            
            # Compute familiarity effects for each category
            animal_familiarity_effect = familiar_animal - new_animal
            non_animal_familiarity_effect = familiar_non_animal - new_non_animal
            
            # Compare familiarity effects between categories
            t_values, p_values = [], []
            for tp in range(len(times)):
                t_val, p_val = stats.ttest_rel(
                    animal_familiarity_effect[:, tp], 
                    non_animal_familiarity_effect[:, tp]
                )
                t_values.append(t_val)
                p_values.append(p_val)
            
            t_values = np.array(t_values)
            p_values = np.array(p_values)
            
            # Apply FDR correction
            if self.stats_config['fdr_correction']:
                _, p_values_corrected = fdr_correction(p_values, alpha=self.stats_config['alpha'])
            else:
                p_values_corrected = p_values
            
            significant_mask = p_values_corrected < self.stats_config['alpha']
            
            category_results[roi_name] = {
                'animal_familiarity_effect': np.mean(animal_familiarity_effect, axis=0),
                'non_animal_familiarity_effect': np.mean(non_animal_familiarity_effect, axis=0),
                'category_difference': np.mean(animal_familiarity_effect - non_animal_familiarity_effect, axis=0),
                't_values': t_values,
                'p_values': p_values,
                'p_values_corrected': p_values_corrected,
                'significant_mask': significant_mask,
                'times': times
            }
        
        self.logger.info("Category effects analysis completed")
        return category_results
    
    def plot_erp_results(self, erp_data: Dict, roi_results: Dict, 
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive ERP results visualization matching Delorme et al. 2018
        
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
            ERP results figure
        """
        
        # Create figure matching the original study layout
        fig = plt.figure(figsize=(16, 12))
        
        # Define colors matching the study
        colors = {
            'familiar': self.vis_config['colors']['familiar'],
            'new': self.vis_config['colors']['new'],
            'difference': self.vis_config['colors']['difference']
        }
        
        times = erp_data['times']
        
        # Plot 1: Frontal ROI ERPs
        ax1 = plt.subplot(3, 2, 1)
        if 'frontal' in roi_results:
            frontal_data = roi_results['frontal']
            
            # Plot familiar and new ERPs
            ax1.plot(times * 1000, frontal_data['familiar_erp'] * 1e6, 
                    color=colors['familiar'], linewidth=2, label='Familiar')
            ax1.plot(times * 1000, frontal_data['new_erp'] * 1e6, 
                    color=colors['new'], linewidth=2, label='New')
            
            # Fill between for SEM (simplified)
            ax1.fill_between(times * 1000, 
                           (frontal_data['familiar_erp'] - np.std(frontal_data['familiar_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           (frontal_data['familiar_erp'] + np.std(frontal_data['familiar_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           color=colors['familiar'], alpha=0.3)
            ax1.fill_between(times * 1000, 
                           (frontal_data['new_erp'] - np.std(frontal_data['new_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           (frontal_data['new_erp'] + np.std(frontal_data['new_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           color=colors['new'], alpha=0.3)
            
            # Mark significant time points
            sig_mask = frontal_data['significant_mask']
            if np.any(sig_mask):
                sig_times = times[sig_mask] * 1000
                y_min, y_max = ax1.get_ylim()
                ax1.scatter(sig_times, np.ones(len(sig_times)) * (y_min + 0.1 * (y_max - y_min)), 
                          color='red', s=10, marker='|')
            
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude (µV)')
            ax1.set_title('Frontal ROI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Parieto-occipital ROI ERPs
        ax2 = plt.subplot(3, 2, 2)
        if 'parieto_occipital' in roi_results:
            parietal_data = roi_results['parieto_occipital']
            
            # Plot familiar and new ERPs
            ax2.plot(times * 1000, parietal_data['familiar_erp'] * 1e6, 
                    color=colors['familiar'], linewidth=2, label='Familiar')
            ax2.plot(times * 1000, parietal_data['new_erp'] * 1e6, 
                    color=colors['new'], linewidth=2, label='New')
            
            # Fill between for SEM
            ax2.fill_between(times * 1000, 
                           (parietal_data['familiar_erp'] - np.std(parietal_data['familiar_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           (parietal_data['familiar_erp'] + np.std(parietal_data['familiar_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           color=colors['familiar'], alpha=0.3)
            ax2.fill_between(times * 1000, 
                           (parietal_data['new_erp'] - np.std(parietal_data['new_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           (parietal_data['new_erp'] + np.std(parietal_data['new_erp']) / np.sqrt(len(erp_data['subjects']))) * 1e6,
                           color=colors['new'], alpha=0.3)
            
            # Mark significant time points
            sig_mask = parietal_data['significant_mask']
            if np.any(sig_mask):
                sig_times = times[sig_mask] * 1000
                y_min, y_max = ax2.get_ylim()
                ax2.scatter(sig_times, np.ones(len(sig_times)) * (y_min + 0.1 * (y_max - y_min)), 
                          color='red', s=10, marker='|')
            
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Amplitude (µV)')
            ax2.set_title('Parieto-occipital ROI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Difference waves (Frontal)
        ax3 = plt.subplot(3, 2, 3)
        if 'frontal' in roi_results:
            frontal_data = roi_results['frontal']
            ax3.plot(times * 1000, frontal_data['difference_erp'] * 1e6, 
                    color=colors['difference'], linewidth=2, label='Familiar - New')
            
            # Mark significant time points
            sig_mask = frontal_data['significant_mask']
            if np.any(sig_mask):
                sig_times = times[sig_mask] * 1000
                y_min, y_max = ax3.get_ylim()
                ax3.scatter(sig_times, np.ones(len(sig_times)) * (y_min + 0.1 * (y_max - y_min)), 
                          color='red', s=10, marker='|')
            
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Amplitude (µV)')
            ax3.set_title('Frontal Difference Wave')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Difference waves (Parieto-occipital)
        ax4 = plt.subplot(3, 2, 4)
        if 'parieto_occipital' in roi_results:
            parietal_data = roi_results['parieto_occipital']
            ax4.plot(times * 1000, parietal_data['difference_erp'] * 1e6, 
                    color=colors['difference'], linewidth=2, label='Familiar - New')
            
            # Mark significant time points
            sig_mask = parietal_data['significant_mask']
            if np.any(sig_mask):
                sig_times = times[sig_mask] * 1000
                y_min, y_max = ax4.get_ylim()
                ax4.scatter(sig_times, np.ones(len(sig_times)) * (y_min + 0.1 * (y_max - y_min)), 
                          color='red', s=10, marker='|')
            
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Amplitude (µV)')
            ax4.set_title('Parieto-occipital Difference Wave')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 5: Topographical map at peak effect time
        ax5 = plt.subplot(3, 2, 5)
        # Create a simple topographical representation
        # This is simplified - for full topomap, you'd need electrode positions
        if 'frontal' in roi_results and roi_results['frontal']['peak_effect_time'] is not None:
            peak_time = roi_results['frontal']['peak_effect_time']
            time_idx = np.argmin(np.abs(times - peak_time))
            
            # Get difference data at peak time
            familiar_data = np.mean(erp_data['conditions']['familiar'], axis=0)
            new_data = np.mean(erp_data['conditions']['new'], axis=0)
            diff_data = familiar_data[:, time_idx] - new_data[:, time_idx]
            
            # Simple channel-wise visualization
            ax5.bar(range(len(diff_data)), diff_data * 1e6)
            ax5.set_xlabel('Channel Index')
            ax5.set_ylabel('Amplitude Difference (µV)')
            ax5.set_title(f'Topography at {peak_time*1000:.0f} ms')
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(3, 2, 6)
        
        # Collect key results
        summary_data = []
        
        for roi_name in ['frontal', 'parieto_occipital']:
            if roi_name in roi_results:
                data = roi_results[roi_name]
                if data['earliest_significant_time'] is not None:
                    summary_data.append({
                        'ROI': roi_name.replace('_', '-').title(),
                        'Onset (ms)': data['earliest_significant_time'] * 1000,
                        'Peak (ms)': data['peak_effect_time'] * 1000,
                        'Max t-value': np.max(np.abs(data['t_values']))
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Plot onset times
            ax6.bar(summary_df['ROI'], summary_df['Onset (ms)'], 
                   alpha=0.7, color=['blue', 'green'])
            ax6.set_ylabel('Onset Time (ms)')
            ax6.set_title('Effect Onset Times')
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'No significant effects found', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def save_erp_results(self, erp_data: Dict, familiarity_results: Dict, 
                        roi_results: Dict, save_dir: Path) -> Dict[str, Path]:
        """
        Save ERP analysis results
        
        Parameters:
        -----------
        erp_data : Dict
            ERP data dictionary
        familiarity_results : Dict
            Familiarity effect results
        roi_results : Dict
            ROI analysis results
        save_dir : Path
            Directory to save results
            
        Returns:
        --------
        saved_files : Dict[str, Path]
            Dictionary of saved file paths
        """
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save ERP data (numpy format for efficiency)
        erp_path = save_dir / "erp_data.npz"
        np.savez_compressed(erp_path, **erp_data['conditions'], 
                          times=erp_data['times'], ch_names=erp_data['ch_names'])
        saved_files['erp_data'] = erp_path
        
        # Save statistical results
        stats_results = {
            'familiarity_results': familiarity_results,
            'roi_results': roi_results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        stats_results_json = convert_numpy(stats_results)
        
        stats_path = save_dir / "statistical_results.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_results_json, f, indent=2)
        saved_files['statistics'] = stats_path
        
        self.logger.info("ERP analysis results saved")
        
        return saved_files

def compute_cohens_d_erp(erp1: np.ndarray, erp2: np.ndarray) -> np.ndarray:
    """
    Compute Cohen's d effect size for ERP data
    
    Parameters:
    -----------
    erp1 : np.ndarray
        ERP data for condition 1 (subjects x channels x times)
    erp2 : np.ndarray
        ERP data for condition 2 (subjects x channels x times)
        
    Returns:
    --------
    cohens_d : np.ndarray
        Cohen's d values (channels x times)
    """
    
    diff = erp1 - erp2
    pooled_std = np.sqrt((np.var(erp1, axis=0, ddof=1) + np.var(erp2, axis=0, ddof=1)) / 2)
    cohens_d = np.mean(diff, axis=0) / pooled_std
    
    return cohens_d

def load_erp_results(save_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load saved ERP analysis results
    
    Parameters:
    -----------
    save_dir : Path
        Directory containing saved results
        
    Returns:
    --------
    erp_data : Dict
        ERP data dictionary
    familiarity_results : Dict
        Familiarity effect results
    roi_results : Dict
        ROI analysis results
    """
    
    save_dir = Path(save_dir)
    
    # Load ERP data
    erp_file = save_dir / "erp_data.npz"
    erp_npz = np.load(erp_file)
    
    erp_data = {
        'conditions': {key: erp_npz[key] for key in erp_npz.files if key not in ['times', 'ch_names']},
        'times': erp_npz['times'],
        'ch_names': erp_npz['ch_names'].tolist()
    }
    
    # Load statistical results
    stats_file = save_dir / "statistical_results.json"
    with open(stats_file, 'r') as f:
        stats_results = json.load(f)
    
    familiarity_results = stats_results['familiarity_results']
    roi_results = stats_results['roi_results']
    
    return erp_data, familiarity_results, roi_results
