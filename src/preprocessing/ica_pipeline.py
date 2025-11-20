"""
ICA Pipeline for EEG Preprocessing
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import mne
from mne_icalabel import label_components


class ICAProcessor:
    """Handle ICA decomposition and component rejection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ica_config = config['preprocessing']['ica']
        self.iclabel_config = config['preprocessing']['iclabel']
        self.logger = logging.getLogger(__name__)
        
    def run_ica(self, raw: mne.io.Raw, subject: str) -> mne.preprocessing.ICA:
        """
        Run ICA decomposition
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Preprocessed raw data (filtered)
        subject : str
            Subject identifier
            
        Returns:
        --------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        """
        
        # Initialize ICA
        fit_params = self.ica_config.get('fit_params')
        if fit_params is not None and not isinstance(fit_params, dict):
            self.logger.warning("Ignoring non-dict ICA fit_params: %s", fit_params)
            fit_params = None

        ica = mne.preprocessing.ICA(
            n_components=self.ica_config['n_components'],
            method=self.ica_config['method'],
            max_iter=self.ica_config['max_iter'],
            random_state=self.ica_config['random_state'],
            fit_params=fit_params,
        )
        
        # Fit ICA on filtered data
        self.logger.info(f"Running ICA for {subject}...")
        ica.fit(raw, picks='eeg', reject_by_annotation=True)
        
        self.logger.info(f"ICA completed for {subject}: {ica.n_components_} components extracted")
        
        return ica
    
    def manual_ica_review(self, raw: mne.io.Raw, ica: mne.preprocessing.ICA, 
                         subject: str) -> Tuple[List[int], Dict]:
        """
        Interactive manual ICA component review
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw data
        ica : mne.preprocessing.ICA
            Fitted ICA object
        subject : str
            Subject identifier
            
        Returns:
        --------
        reject_components : List[int]
            List of component indices to reject
        review_notes : Dict
            Notes from manual review
        """
        
        self.logger.info(f"Starting manual ICA review for {subject}")
        
        # Plot ICA components for review
        fig = ica.plot_components(picks=range(min(20, ica.n_components_)), show=False)
        fig.suptitle(f'ICA Components for {subject} - Manual Review')
        plt.show()
        
        # Plot component time series
        ica.plot_sources(raw, start=0, stop=10, show=False)
        plt.show()
        
        # Interactive component selection
        print(f"\nManual ICA Review for {subject}")
        print(f"Total components: {ica.n_components_}")
        print("Review the plotted components and identify artifacts")
        print("Enter component indices to reject (comma-separated, e.g., 0,1,5):")
        print("Press Enter if no components need rejection")
        
        reject_input = input("Components to reject: ").strip()
        
        if reject_input:
            try:
                reject_components = [int(x.strip()) for x in reject_input.split(',')]
                reject_components = [x for x in reject_components if 0 <= x < ica.n_components_]
            except ValueError:
                self.logger.warning("Invalid input. No components will be rejected.")
                reject_components = []
        else:
            reject_components = []
        
        # Document the review
        review_notes = {
            'subject': subject,
            'total_components': ica.n_components_,
            'rejected_components': reject_components,
            'rejection_reasons': {},
            'review_method': 'manual'
        }
        
        # Get rejection reasons for each component
        for comp in reject_components:
            reason = input(f"Reason for rejecting component {comp} (e.g., 'eye blink', 'muscle', 'line noise'): ")
            review_notes['rejection_reasons'][comp] = reason
        
        self.logger.info(f"Manual review completed for {subject}: {len(reject_components)} components marked for rejection")
        
        return reject_components, review_notes
    
    def automated_ica_rejection(self, raw: mne.io.Raw, ica: mne.preprocessing.ICA, 
                               subject: str) -> Tuple[List[int], Dict]:
        """
        Automated ICA component rejection using ICLabel
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw data
        ica : mne.preprocessing.ICA
            Fitted ICA object
        subject : str
            Subject identifier
            
        Returns:
        --------
        reject_components : List[int]
            List of component indices to reject
        classification_results : Dict
            ICLabel classification results
        """
        
        self.logger.info(f"Running automated ICA classification for {subject}")
        
        try:
            # Run ICLabel classification
            ic_labels = label_components(raw, ica, method='iclabel')

            # Extract labels and probabilities
            labels = ic_labels.get('labels', [])
            probabilities = np.asarray(ic_labels.get('y_pred_proba', []))
            classes = ic_labels.get('classes', [
                'brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other'
            ])
            canonical_classes = [self._canonicalize_label(cls) for cls in classes]

            # Identify components to reject
            reject_components: List[int] = []
            reject_classes = {
                self._canonicalize_label(label) for label in self.iclabel_config['reject_classes']
            }
            threshold = self.iclabel_config['threshold']

            component_probabilities: List[Dict[str, float]] = []
            component_details: List[Dict] = []

            for i, label in enumerate(labels):
                canonical_label = self._canonicalize_label(label)
                prob_map, decision_probability = self._extract_probabilities(
                    probabilities, i, canonical_label, canonical_classes
                )

                component_probabilities.append(prob_map)

                rejected_flag = canonical_label in reject_classes and decision_probability >= threshold
                if rejected_flag:
                    reject_components.append(i)

                component_details.append(
                    {
                        'component': int(i),
                        'predicted_label': str(label),
                        'canonical_label': canonical_label,
                        'decision_probability': float(decision_probability),
                        'probabilities': prob_map,
                        'rejected': bool(rejected_flag),
                    }
                )

            classification_results = {
                'subject': subject,
                'total_components': int(ica.n_components_),
                'labels': list(labels),
                'canonical_labels': [self._canonicalize_label(lbl) for lbl in labels],
                'probabilities': component_probabilities,
                'rejected_components': reject_components,
                'rejection_threshold': threshold,
                'rejected_classes': sorted(reject_classes),
                'classes': canonical_classes,
                'review_method': 'automated_iclabel',
                'component_details': component_details,
            }
            
            self.logger.info(f"Automated classification completed for {subject}: "
                           f"{len(reject_components)} components marked for rejection")
            
        except Exception as e:
            self.logger.error(f"ICLabel classification failed for {subject}: {e}")
            # Fallback: use simple heuristics
            reject_components, classification_results = self._fallback_rejection(ica, subject)
        
        return reject_components, classification_results

    @staticmethod
    def _canonicalize_label(label: Optional[str]) -> str:
        if not label:
            return 'unknown'
        return label.lower().strip().replace(' ', '_')

    def _extract_probabilities(
        self,
        probabilities: np.ndarray,
        index: int,
        canonical_label: str,
        class_order: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """
        Normalize the probability outputs from ICLabel across versions.

        Returns a mapping of canonical class → probability and the decision probability
        for the provided canonical label.
        """
        if probabilities.size == 0:
            return {}, 0.0

        # Older versions return shape (n_components,), storing only the winning probability.
        if probabilities.ndim == 1:
            prob_value = float(probabilities[index])
            return {canonical_label: prob_value}, prob_value

        row = np.asarray(probabilities[index]).flatten()
        prob_map: Dict[str, float] = {}

        for cls_name, prob in zip(class_order, row):
            prob_map[cls_name] = float(prob)

        decision_probability = prob_map.get(canonical_label, float(np.max(row)))
        return prob_map, float(decision_probability)
    
    def _fallback_rejection(self, ica: mne.preprocessing.ICA, subject: str) -> Tuple[List[int], Dict]:
        """Fallback component rejection using simple heuristics"""
        
        self.logger.info(f"Using fallback heuristics for {subject}")
        
        reject_components = []
        
        # Simple heuristics based on component properties
        for i in range(ica.n_components_):
            # Get component time series
            component_data = ica.get_sources(ica.info)._data[i]
            
            # Check for high variance (potential artifacts)
            if np.var(component_data) > np.percentile([np.var(ica.get_sources(ica.info)._data[j]) 
                                                      for j in range(ica.n_components_)], 95):
                reject_components.append(i)
        
        classification_results = {
            'subject': subject,
            'total_components': ica.n_components_,
            'rejected_components': reject_components,
            'review_method': 'fallback_heuristics'
        }
        
        return reject_components, classification_results
    
    def apply_ica_rejection(self, raw: mne.io.Raw, ica: mne.preprocessing.ICA, 
                           reject_components: List[int]) -> mne.io.Raw:
        """
        Apply ICA component rejection to raw data
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw data
        ica : mne.preprocessing.ICA
            Fitted ICA object
        reject_components : List[int]
            Components to reject
            
        Returns:
        --------
        raw_cleaned : mne.io.Raw
            Cleaned raw data
        """
        
        if not reject_components:
            self.logger.info("No components to reject")
            return raw.copy()
        
        # Apply rejection
        raw_cleaned = raw.copy()
        ica.exclude = reject_components
        ica.apply(raw_cleaned)
        
        self.logger.info(f"Applied ICA rejection: {len(reject_components)} components removed")
        
        return raw_cleaned
    
    def plot_ica_results(self, raw_before: mne.io.Raw, raw_after: mne.io.Raw, 
                        ica: mne.preprocessing.ICA, reject_components: List[int],
                        subject: str, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive ICA results visualization
        
        Parameters:
        -----------
        raw_before : mne.io.Raw
            Raw data before ICA
        raw_after : mne.io.Raw
            Raw data after ICA
        ica : mne.preprocessing.ICA
            Fitted ICA object
        reject_components : List[int]
            Rejected components
        subject : str
            Subject identifier
        save_path : Path, optional
            Path to save figure
            
        Returns:
        --------
        fig : plt.Figure
            ICA results figure
        """
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'ICA Results for {subject}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original vs cleaned data (time series)
        ax = axes[0, 0]
        times = raw_before.times
        sample_channel = 'Fz'  # Use Fz as representative channel
        
        if sample_channel in raw_before.ch_names:
            ch_idx = raw_before.ch_names.index(sample_channel)
            data_before = raw_before.get_data()[ch_idx]
            data_after = raw_after.get_data()[ch_idx]
            
            # Plot a 10-second segment
            start_idx = 0
            end_idx = min(int(10 * raw_before.info['sfreq']), len(times))
            
            ax.plot(times[start_idx:end_idx], data_before[start_idx:end_idx] * 1e6, 
                   'b-', alpha=0.7, label='Before ICA')
            ax.plot(times[start_idx:end_idx], data_after[start_idx:end_idx] * 1e6, 
                   'r-', alpha=0.7, label='After ICA')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (µV)')
            ax.set_title(f'Channel {sample_channel}: Before vs After ICA')
            ax.legend()
        
        # Plot 2: Power spectral density comparison
        ax = axes[0, 1]
        psds_before, freqs = mne.time_frequency.psd_array_welch(raw_before.get_data(picks='eeg'), raw_before.info['sfreq'], fmin=1, fmax=50)
        psds_after, _ = mne.time_frequency.psd_array_welch(raw_after.get_data(picks='eeg'), raw_after.info['sfreq'], fmin=1, fmax=50)
        
        ax.loglog(freqs, np.mean(psds_before, axis=0), 'b-', alpha=0.7, label='Before ICA')
        ax.loglog(freqs, np.mean(psds_after, axis=0), 'r-', alpha=0.7, label='After ICA')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (V²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Rejected components
        ax = axes[1, 0]
        if reject_components:
            # Show topographies of rejected components
            n_show = min(6, len(reject_components))
            comp_indices = reject_components[:n_show]
            
            try:
                ica.plot_components(picks=comp_indices, axes=ax, show=False, colorbar=False)
                ax.set_title(f'Rejected Components (showing first {n_show})')
            except:
                ax.text(0.5, 0.5, f'{len(reject_components)} components rejected', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Rejected Components')
        else:
            ax.text(0.5, 0.5, 'No components rejected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rejected Components')
        
        # Plot 4: Component variance explained
        ax = axes[1, 1]
        explained_var = ica.pca_.explained_variance_ratio_[:ica.n_components_]
        ax.bar(range(len(explained_var)), explained_var)
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Component Variance Explained')
        
        # Highlight rejected components
        if reject_components:
            for comp in reject_components:
                if comp < len(explained_var):
                    ax.bar(comp, explained_var[comp], color='red', alpha=0.7)
        
        # Plot 5: Variance comparison
        ax = axes[2, 0]
        var_before = np.var(raw_before.get_data(picks='eeg'), axis=1)
        var_after = np.var(raw_after.get_data(picks='eeg'), axis=1)
        
        ax.scatter(var_before * 1e12, var_after * 1e12, alpha=0.6)
        ax.plot([min(var_before * 1e12), max(var_before * 1e12)], 
                [min(var_before * 1e12), max(var_before * 1e12)], 'r--', alpha=0.7)
        ax.set_xlabel('Variance Before ICA (µV²)')
        ax.set_ylabel('Variance After ICA (µV²)')
        ax.set_title('Channel Variance: Before vs After')
        
        # Plot 6: Summary statistics
        ax = axes[2, 1]
        metrics = {
            'Components\nExtracted': ica.n_components_,
            'Components\nRejected': len(reject_components),
            'Variance\nReduction (%)': (1 - np.mean(var_after) / np.mean(var_before)) * 100,
            'Channels\nProcessed': len(raw_before.ch_names)
        }
        
        bars = ax.bar(metrics.keys(), metrics.values())
        ax.set_title('ICA Summary Statistics')
        ax.set_ylabel('Value')
        
        # Color code the bars
        colors = ['blue', 'red', 'green', 'orange']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save_ica_results(self, ica: mne.preprocessing.ICA, reject_components: List[int], 
                        classification_results: Dict, subject: str, 
                        save_dir: Path) -> Dict[str, Path]:
        """
        Save ICA results to disk
        
        Parameters:
        -----------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        reject_components : List[int]
            Rejected components
        classification_results : Dict
            Classification results
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
        
        # Save ICA object
        ica_path = save_dir / f"{subject}_ica.fif"
        ica.save(ica_path, overwrite=True)
        saved_files['ica'] = ica_path
        
        # Save classification results
        results_path = save_dir / f"{subject}_ica_results.json"
        with open(results_path, 'w') as f:
            json.dump(classification_results, f, indent=2)
        saved_files['results'] = results_path
        
        # Save component rejection info
        rejection_info = {
            'subject': subject,
            'rejected_components': reject_components,
            'total_components': ica.n_components_,
            'rejection_method': classification_results.get('review_method', 'unknown')
        }
        
        rejection_path = save_dir / f"{subject}_component_rejection.json"
        with open(rejection_path, 'w') as f:
            json.dump(rejection_info, f, indent=2)
        saved_files['rejection'] = rejection_path
        
        self.logger.info(f"ICA results saved for {subject}")
        
        return saved_files

def load_ica_results(subject: str, save_dir: Path) -> Tuple[mne.preprocessing.ICA, Dict]:
    """
    Load saved ICA results
    
    Parameters:
    -----------
    subject : str
        Subject identifier
    save_dir : Path
        Directory containing saved results
        
    Returns:
    --------
    ica : mne.preprocessing.ICA
        Loaded ICA object
    results : Dict
        ICA results dictionary
    """
    
    save_dir = Path(save_dir)
    
    # Load ICA object
    ica_path = save_dir / f"{subject}_ica.fif"
    ica = mne.preprocessing.read_ica(ica_path)
    
    # Load results
    results_path = save_dir / f"{subject}_ica_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return ica, results
