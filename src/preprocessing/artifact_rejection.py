"""
Artifact Rejection for EEG Epoched Data
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject, get_rejection_threshold
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class ArtifactRejector:
    """Handle epoch-level artifact rejection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reject_config = config['preprocessing']['artist']
        self.epoch_config = config['preprocessing']['epoching']
        self.logger = logging.getLogger(__name__)
        
    def create_epochs(self, raw: mne.io.Raw, events: np.ndarray, 
                     event_id: Dict, subject: str) -> mne.Epochs:
        """
        Create epochs from continuous data
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Continuous EEG data
        events : np.ndarray
            Events array
        event_id : Dict
            Event ID mapping
        subject : str
            Subject identifier
            
        Returns:
        --------
        epochs : mne.Epochs
            Epoched data
        """
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=self.epoch_config['tmin'],
            tmax=self.epoch_config['tmax'],
            baseline=tuple(self.epoch_config['baseline']),
            picks='eeg',
            preload=True,
            reject=None,  # We'll do rejection separately
            flat=None
        )
        
        self.logger.info(f"Created epochs for {subject}: {len(epochs)} epochs")
        
        return epochs
    
    def manual_artifact_rejection(self, epochs: mne.Epochs, subject: str) -> Tuple[mne.Epochs, Dict]:
        """
        Manual artifact rejection using visual inspection
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Epoched data
        subject : str
            Subject identifier
            
        Returns:
        --------
        epochs_clean : mne.Epochs
            Cleaned epochs
        rejection_log : Dict
            Rejection statistics
        """
        
        self.logger.info(f"Starting manual artifact rejection for {subject}")
        
        # Plot epochs for visual inspection
        epochs.plot(n_channels=20, n_epochs=5, scalings='auto', show=False)
        plt.show()
        
        # Interactive epoch rejection
        print(f"\nManual Artifact Rejection for {subject}")
        print(f"Total epochs: {len(epochs)}")
        print("Review the plotted epochs and identify bad epochs")
        print("Enter epoch indices to reject (comma-separated, e.g., 0,5,12):")
        print("Press Enter if no epochs need rejection")
        
        reject_input = input("Epochs to reject: ").strip()
        
        if reject_input:
            try:
                reject_epochs = [int(x.strip()) for x in reject_input.split(',')]
                reject_epochs = [x for x in reject_epochs if 0 <= x < len(epochs)]
            except ValueError:
                self.logger.warning("Invalid input. No epochs will be rejected.")
                reject_epochs = []
        else:
            reject_epochs = []
        
        # Apply manual rejection
        if reject_epochs:
            epochs_clean = epochs.copy()
            epochs_clean.drop(reject_epochs, reason='manual')
        else:
            epochs_clean = epochs.copy()
        
        rejection_log = {
            'subject': subject,
            'total_epochs': len(epochs),
            'rejected_epochs': reject_epochs,
            'epochs_remaining': len(epochs_clean),
            'rejection_rate': len(reject_epochs) / len(epochs),
            'rejection_method': 'manual'
        }
        
        self.logger.info(f"Manual rejection completed for {subject}: "
                        f"{len(reject_epochs)} epochs rejected")
        
        return epochs_clean, rejection_log
    
    def automated_artifact_rejection(self, epochs: mne.Epochs, subject: str) -> Tuple[mne.Epochs, Dict]:
        """
        Automated artifact rejection using AutoReject
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Epoched data
        subject : str
            Subject identifier
            
        Returns:
        --------
        epochs_clean : mne.Epochs
            Cleaned epochs
        rejection_log : Dict
            Rejection statistics
        """
        
        self.logger.info(f"Running automated artifact rejection for {subject}")
        
        try:
            # Initialize AutoReject
            ar = AutoReject(
                n_interpolate=[1, 4, 8],
                consensus_percs=[0.5, 0.7, 0.9],
                thresh_method='random_search',
                n_jobs=1,
                random_state=42,
                cv=3
            )
            
            # Fit and transform epochs
            ar.fit(epochs)
            epochs_clean, reject_log_ar = ar.transform(epochs, return_log=True)
            
            # Get rejection statistics
            n_rejected = np.sum(reject_log_ar.bad_epochs)
            n_interpolated = np.sum(reject_log_ar.labels == 1)
            
            rejection_log = {
                'subject': subject,
                'total_epochs': len(epochs),
                'rejected_epochs': n_rejected,
                'interpolated_epochs': n_interpolated,
                'epochs_remaining': len(epochs_clean),
                'rejection_rate': n_rejected / len(epochs),
                'interpolation_rate': n_interpolated / len(epochs),
                'rejection_method': 'autoreject',
                'rejection_thresholds': ar.threshes_.tolist() if hasattr(ar, 'threshes_') else None
            }
            
        except Exception as e:
            self.logger.error(f"AutoReject failed for {subject}: {e}")
            # Fallback to simple threshold rejection
            epochs_clean, rejection_log = self._fallback_rejection(epochs, subject)
        
        self.logger.info(f"Automated rejection completed for {subject}: "
                        f"{rejection_log['rejected_epochs']} epochs rejected")
        
        return epochs_clean, rejection_log
    
    def _fallback_rejection(self, epochs: mne.Epochs, subject: str) -> Tuple[mne.Epochs, Dict]:
        """Fallback rejection using simple amplitude thresholds"""
        
        self.logger.info(f"Using fallback rejection for {subject}")
        
        # Define rejection criteria
        reject_criteria = self.reject_config.get('reject_criteria', {'eeg': 100e-6})
        flat_criteria = self.reject_config.get('flat_criteria', {'eeg': 1e-6})
        
        # Apply rejection
        epochs_clean = epochs.copy()
        epochs_clean.drop_bad(reject=reject_criteria, flat=flat_criteria)
        
        n_rejected = len(epochs) - len(epochs_clean)
        
        rejection_log = {
            'subject': subject,
            'total_epochs': len(epochs),
            'rejected_epochs': n_rejected,
            'epochs_remaining': len(epochs_clean),
            'rejection_rate': n_rejected / len(epochs),
            'rejection_method': 'threshold_fallback',
            'rejection_criteria': reject_criteria,
            'flat_criteria': flat_criteria
        }
        
        return epochs_clean, rejection_log
    
    def plot_artifact_rejection_results(self, epochs_before: mne.Epochs, 
                                      epochs_after: mne.Epochs,
                                      rejection_log: Dict, subject: str,
                                      save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create artifact rejection results visualization
        
        Parameters:
        -----------
        epochs_before : mne.Epochs
            Epochs before rejection
        epochs_after : mne.Epochs
            Epochs after rejection
        rejection_log : Dict
            Rejection statistics
        subject : str
            Subject identifier
        save_path : Path, optional
            Path to save figure
            
        Returns:
        --------
        fig : plt.Figure
            Artifact rejection results figure
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Artifact Rejection Results for {subject}', fontsize=16, fontweight='bold')
        
        # Plot 1: Epoch count comparison
        ax = axes[0, 0]
        counts = [len(epochs_before), len(epochs_after)]
        labels = ['Before Rejection', 'After Rejection']
        colors = ['blue', 'green']
        
        bars = ax.bar(labels, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Epochs')
        ax.set_title('Epoch Count Comparison')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom')
        
        # Plot 2: Rejection rate by condition
        ax = axes[0, 1]
        if hasattr(epochs_before, 'event_id'):
            conditions = list(epochs_before.event_id.keys())
            rejection_rates = []
            
            for condition in conditions:
                epochs_cond_before = epochs_before[condition]
                epochs_cond_after = epochs_after[condition] if condition in epochs_after.event_id else []
                
                rate = 1 - (len(epochs_cond_after) / len(epochs_cond_before)) if len(epochs_cond_before) > 0 else 0
                rejection_rates.append(rate * 100)
            
            ax.bar(conditions, rejection_rates, alpha=0.7)
            ax.set_ylabel('Rejection Rate (%)')
            ax.set_title('Rejection Rate by Condition')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No condition info available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rejection Rate by Condition')
        
        # Plot 3: Amplitude distribution comparison
        ax = axes[0, 2]
        data_before = epochs_before.get_data()
        data_after = epochs_after.get_data()
        
        # Calculate peak-to-peak amplitudes
        ptp_before = np.ptp(data_before, axis=2).flatten() * 1e6  # Convert to µV
        ptp_after = np.ptp(data_after, axis=2).flatten() * 1e6
        
        ax.hist(ptp_before, bins=50, alpha=0.7, label='Before', color='blue', density=True)
        ax.hist(ptp_after, bins=50, alpha=0.7, label='After', color='green', density=True)
        ax.set_xlabel('Peak-to-Peak Amplitude (µV)')
        ax.set_ylabel('Density')
        ax.set_title('Amplitude Distribution')
        ax.legend()
        
        # Plot 4: Power spectral density comparison
        ax = axes[1, 0]
        psds_before, freqs = mne.time_frequency.psd_array_welch(epochs_before.get_data().reshape(-1, epochs_before.get_data().shape[-1]), epochs_before.info['sfreq'], fmin=1, fmax=50)
        psds_after, _ = mne.time_frequency.psd_array_welch(epochs_after.get_data().reshape(-1, epochs_after.get_data().shape[-1]), epochs_after.info['sfreq'], fmin=1, fmax=50)
        
        ax.loglog(freqs, np.mean(psds_before, axis=(0, 1)), 'b-', alpha=0.7, label='Before')
        ax.loglog(freqs, np.mean(psds_after, axis=(0, 1)), 'g-', alpha=0.7, label='After')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (V²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True)
        
        # Plot 5: Summary statistics
        ax = axes[1, 1]
        metrics = {
            'Original\nEpochs': len(epochs_before),
            'Rejected\nEpochs': rejection_log.get('rejected_epochs', 0),
            'Remaining\nEpochs': len(epochs_after),
            'Rejection\nRate (%)': rejection_log.get('rejection_rate', 0) * 100
        }
        
        bars = ax.bar(metrics.keys(), metrics.values())
        ax.set_title('Rejection Summary')
        ax.set_ylabel('Count / Percentage')
        
        # Color code the bars
        colors = ['blue', 'red', 'green', 'orange']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Plot 6: Variance comparison
        ax = axes[1, 2]
        var_before = np.var(data_before, axis=2).mean(axis=0) * 1e12  # Convert to µV²
        var_after = np.var(data_after, axis=2).mean(axis=0) * 1e12
        
        # Show variance for a subset of channels
        n_show = min(10, len(var_before))
        ch_indices = np.linspace(0, len(var_before)-1, n_show, dtype=int)
        
        x_pos = np.arange(n_show)
        width = 0.35
        
        ax.bar(x_pos - width/2, var_before[ch_indices], width, label='Before', alpha=0.7)
        ax.bar(x_pos + width/2, var_after[ch_indices], width, label='After', alpha=0.7)
        
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Variance (µV²)')
        ax.set_title('Channel Variance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in ch_indices])
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save_rejection_results(self, epochs: mne.Epochs, rejection_log: Dict, 
                             subject: str, save_dir: Path) -> Dict[str, Path]:
        """
        Save artifact rejection results
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Cleaned epochs
        rejection_log : Dict
            Rejection statistics
        subject : str
            Subject identifier
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
        
        # Save cleaned epochs
        epochs_path = save_dir / f"{subject}_clean_epo.fif"
        epochs.save(epochs_path, overwrite=True)
        saved_files['epochs'] = epochs_path
        
        # Save rejection log
        log_path = save_dir / f"{subject}_rejection_log.json"
        with open(log_path, 'w') as f:
            json.dump(rejection_log, f, indent=2)
        saved_files['log'] = log_path
        
        self.logger.info(f"Artifact rejection results saved for {subject}")
        
        return saved_files

def compute_rejection_statistics(epochs_before: mne.Epochs, epochs_after: mne.Epochs) -> Dict:
    """
    Compute comprehensive rejection statistics
    
    Parameters:
    -----------
    epochs_before : mne.Epochs
        Epochs before rejection
    epochs_after : mne.Epochs
        Epochs after rejection
        
    Returns:
    --------
    stats : Dict
        Rejection statistics
    """
    
    stats = {
        'n_epochs_before': len(epochs_before),
        'n_epochs_after': len(epochs_after),
        'n_epochs_rejected': len(epochs_before) - len(epochs_after),
        'rejection_rate': 1 - (len(epochs_after) / len(epochs_before)),
        'data_retention_rate': len(epochs_after) / len(epochs_before)
    }
    
    # Condition-specific statistics
    if hasattr(epochs_before, 'event_id'):
        condition_stats = {}
        for condition in epochs_before.event_id.keys():
            epochs_cond_before = epochs_before[condition]
            epochs_cond_after = epochs_after[condition] if condition in epochs_after.event_id else []
            
            condition_stats[condition] = {
                'n_before': len(epochs_cond_before),
                'n_after': len(epochs_cond_after),
                'rejection_rate': 1 - (len(epochs_cond_after) / len(epochs_cond_before)) if len(epochs_cond_before) > 0 else 0
            }
        
        stats['condition_stats'] = condition_stats
    
    # Signal quality improvement
    data_before = epochs_before.get_data()
    data_after = epochs_after.get_data()
    
    # Variance reduction
    var_before = np.var(data_before, axis=2).mean()
    var_after = np.var(data_after, axis=2).mean()
    stats['variance_reduction'] = 1 - (var_after / var_before)
    
    # SNR improvement (simple estimate)
    signal_power_before = np.var(data_before.mean(axis=2), axis=0).mean()
    noise_power_before = (np.var(data_before, axis=2).mean(axis=0) - signal_power_before).mean()
    
    signal_power_after = np.var(data_after.mean(axis=2), axis=0).mean()
    noise_power_after = (np.var(data_after, axis=2).mean(axis=0) - signal_power_after).mean()
    
    snr_before = signal_power_before / (noise_power_before + 1e-12)
    snr_after = signal_power_after / (noise_power_after + 1e-12)
    
    stats['snr_improvement_db'] = 10 * np.log10(snr_after / snr_before) if snr_before > 0 else 0
    
    return stats

def load_rejection_results(subject: str, save_dir: Path) -> Tuple[mne.Epochs, Dict]:
    """
    Load saved artifact rejection results
    
    Parameters:
    -----------
    subject : str
        Subject identifier
    save_dir : Path
        Directory containing saved results
        
    Returns:
    --------
    epochs : mne.Epochs
        Cleaned epochs
    rejection_log : Dict
        Rejection statistics
    """
    
    save_dir = Path(save_dir)
    
    # Load epochs
    epochs_path = save_dir / f"{subject}_clean_epo.fif"
    epochs = mne.read_epochs(epochs_path, preload=True)
    
    # Load rejection log
    log_path = save_dir / f"{subject}_rejection_log.json"
    with open(log_path, 'r') as f:
        rejection_log = json.load(f)
    
    return epochs, rejection_log
