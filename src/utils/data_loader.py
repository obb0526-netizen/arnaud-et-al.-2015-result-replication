"""
Data loading utilities for EEG Memory Recognition Analysis
"""

import os
import mne
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict, Optional, Tuple, Union
import logging

from .pathing import find_project_root


class EEGDataLoader:
    """Load and manage EEG data from OpenNeuro ds002680 dataset"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data loader with configuration

        Resolves all paths relative to the project root so notebooks can run
        from any working directory (e.g., `notebooks/`).
        """
        # Initialize logging early
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        project_root = find_project_root()

        # Resolve config path robustly
        if config_path is None:
            candidate = project_root / "config" / "analysis_config.yaml"
        else:
            candidate = Path(config_path)
            if not candidate.is_absolute():
                # Try CWD, then project root
                if not candidate.exists():
                    candidate = project_root / candidate

        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")

        with open(candidate, 'r') as f:
            self.config = yaml.safe_load(f)

        # Normalize paths to absolute based on project root
        def _abs(path_like: Union[str, Path]) -> Path:
            p = Path(path_like)
            return p if p.is_absolute() else (project_root / p)

        self.project_root = project_root
        self.config_path = candidate
        self.raw_dir = _abs(self.config['data']['raw_dir']).resolve()
        self.preprocessed_dir = _abs(self.config['data']['preprocessed_dir']).resolve()
        self.derivatives_dir = _abs(self.config['data']['derivatives_dir']).resolve()

        # Create directories if they don't exist
        for dir_path in [self.preprocessed_dir, self.derivatives_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Diagnostics
        self.logger.info(
            f"EEGDataLoader initialized\n"
            f"  Project root: {self.project_root}\n"
            f"  Config: {self.config_path}\n"
            f"  Raw dir: {self.raw_dir} (exists={self.raw_dir.exists()})\n"
            f"  Preprocessed dir: {self.preprocessed_dir} (exists={self.preprocessed_dir.exists()})\n"
            f"  Derivatives dir: {self.derivatives_dir} (exists={self.derivatives_dir.exists()})"
        )
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subjects from raw data directory"""
        subjects = []
        if self.raw_dir.exists():
            for item in self.raw_dir.iterdir():
                if item.is_dir() and item.name.startswith('sub-'):
                    subjects.append(item.name)
        
        self.logger.info(f"Found {len(subjects)} subjects: {subjects}")
        return sorted(subjects)
    
    def load_raw_eeg(self, subject: str, session: Optional[str] = None,
                     run: Optional[str] = None, task: str = 'gonogo') -> mne.io.Raw:
        """
        Load raw EEG data for a subject
        
        Parameters:
        -----------
        subject : str
            Subject ID (e.g., 'sub-01')
        session : str
            Session name (default: 'familiarization')
            
        Returns:
        --------
        raw : mne.io.Raw
            Raw EEG data
        """
        # Normalize potentially old API usage
        if session is not None and not str(session).startswith('ses-'):
            session = None

        # Enumerate available sessions/runs and pick the first if not provided
        candidates = []
        subj_dir = self.raw_dir / subject
        if subj_dir.exists():
            for ses_dir in sorted(subj_dir.glob('ses-*')):
                eeg_dir = ses_dir / 'eeg'
                if not eeg_dir.exists():
                    continue
                for eeg_file in sorted(eeg_dir.glob(f"{subject}_{ses_dir.name}_task-{task}_run-*_eeg.set")):
                    # Extract run id as the 5th underscore-separated token (.._run-X_..)
                    parts = eeg_file.stem.split('_')
                    run_token = next((p for p in parts if p.startswith('run-')), None)
                    if run_token is not None:
                        candidates.append((ses_dir.name, run_token, eeg_file))

        # Filter by requested session/run if provided
        if session is not None:
            candidates = [c for c in candidates if c[0] == session]
        if run is not None:
            candidates = [c for c in candidates if c[1] == run]

        if not candidates:
            # Diagnostics to help the user
            available = []
            if subj_dir.exists():
                for p in subj_dir.rglob('*.set'):
                    available.append(str(p.relative_to(subj_dir)))
            raise FileNotFoundError(
                f"Could not find EEG file for {subject} with session={session} run={run}. "
                f"Found {len(available)} candidate .set files under {subj_dir}. Examples: {available[:5]}"
            )

        # Pick the first candidate by default
        picked_session, picked_run, eeg_file = candidates[0]
        self.logger.info(f"Loading EEG for {subject}: {picked_session} {picked_run} → {eeg_file}")
        
        # Load EEG data
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        
        self.logger.info(f"Loaded raw data for {subject}: {raw.info['nchan']} channels, "
                        f"{raw.times[-1]:.1f}s duration")
        
        return raw
    
    def load_events(self, subject: str, session: Optional[str] = None,
                    run: Optional[str] = None, task: str = 'gonogo') -> Tuple[np.ndarray, Dict]:
        """
        Load events for a subject
        
        Parameters:
        -----------
        subject : str
            Subject ID
        session : str
            Session name
            
        Returns:
        --------
        events : np.ndarray
            Events array (n_events, 3)
        event_id : Dict
            Event ID mapping
        """
        # Normalize session arg
        if session is not None and not str(session).startswith('ses-'):
            session = None

        # If we can infer file from available candidates, use matching TSV
        raw = self.load_raw_eeg(subject, session=session, run=run, task=task)

        # Determine the expected events TSV path from the raw file path
        raw_file = Path(raw.filenames[0]) if hasattr(raw, 'filenames') else None
        events_file = None
        if raw_file is not None:
            events_file_guess = raw_file.with_name(raw_file.name.replace('_eeg.set', '_events.tsv'))
            if events_file_guess.exists():
                events_file = events_file_guess

        if events_file is not None and events_file.exists():
            events_df = pd.read_csv(events_file, sep='\t')
            
            # DEBUG: Print comprehensive information about the events data
            self.logger.info(f"DEBUG: Events file loaded: {events_file}")
            self.logger.info(f"DEBUG: Events DataFrame shape: {events_df.shape}")
            self.logger.info(f"DEBUG: Events DataFrame columns: {list(events_df.columns)}")
            
            # DEBUG: Check for NaN values in each column
            for col in events_df.columns:
                nan_count = events_df[col].isna().sum()
                self.logger.info(f"DEBUG: Column '{col}' has {nan_count} NaN values")
                if nan_count > 0:
                    self.logger.info(f"DEBUG: Sample values in '{col}': {events_df[col].dropna().head(3).tolist()}")
            
            # DEBUG: Check data types
            self.logger.info(f"DEBUG: DataFrame dtypes: {events_df.dtypes}")

            # Create event ID mapping from 'trial_type' if present, else from 'value'
            if 'trial_type' in events_df.columns:
                self.logger.info(f"DEBUG: Using 'trial_type' column for event mapping")
                unique_events = pd.unique(events_df['trial_type'])
                self.logger.info(f"DEBUG: Unique trial_type values: {unique_events}")
                
                # DEBUG: Check for NaN values in trial_type
                trial_type_nan = events_df['trial_type'].isna().sum()
                self.logger.info(f"DEBUG: trial_type has {trial_type_nan} NaN values")
                
                event_id = {str(event): idx + 1 for idx, event in enumerate(unique_events)}
                self.logger.info(f"DEBUG: Created event_id mapping: {event_id}")
                
                trigger_series = events_df['trial_type'].map(event_id).values
                self.logger.info(f"DEBUG: Created trigger_series with shape: {trigger_series.shape}")
                self.logger.info(f"DEBUG: Trigger series unique values: {np.unique(trigger_series)}")
            else:
                # Handle 'value' column - could be numeric or categorical strings
                self.logger.info(f"DEBUG: Using 'value' column for event mapping")
                unique_events = pd.unique(events_df['value'])
                self.logger.info(f"DEBUG: Unique value values: {unique_events}")
                
                # Check if values are numeric by trying to convert all unique values
                is_numeric = True
                for event in unique_events:
                    try:
                        float(str(event))
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
                
                self.logger.info(f"DEBUG: Value column is_numeric: {is_numeric}")
                
                if is_numeric:
                    # Values are numeric - convert safely
                    try:
                        event_id = {str(event): int(event) for event in unique_events}
                        trigger_series = pd.to_numeric(events_df['value'], errors='coerce').fillna(0).astype(int).values
                    except (ValueError, TypeError):
                        # Fallback to categorical mapping if numeric conversion fails
                        event_id = {str(event): idx + 1 for idx, event in enumerate(unique_events)}
                        trigger_series = events_df['value'].map(event_id).values
                else:
                    # Values are categorical strings
                    event_id = {str(event): idx + 1 for idx, event in enumerate(unique_events)}
                    trigger_series = events_df['value'].map(event_id).values

            # Prefer 'sample' column if present and valid; fallback to onset seconds
            self.logger.info(f"DEBUG: Processing sample column")
            if 'sample' in events_df.columns and not events_df['sample'].isna().all():
                self.logger.info(f"DEBUG: Sample column exists and has some non-NaN values")
                # Check if there are any non-NaN values in the sample column
                valid_samples = events_df['sample'].notna()
                self.logger.info(f"DEBUG: Valid samples count: {valid_samples.sum()}")
                if valid_samples.any():
                    # Use sample column for valid entries, calculate for NaN entries
                    onsets_samples = np.zeros(len(events_df), dtype=int)
                    # Safely convert valid samples to int, handling any remaining NaN values
                    valid_sample_values = events_df.loc[valid_samples, 'sample']
                    self.logger.info(f"DEBUG: Valid sample values shape: {valid_sample_values.shape}")
                    self.logger.info(f"DEBUG: Sample of valid sample values: {valid_sample_values.head(3).tolist()}")
                    
                    # Convert to numeric first, then to int, handling any conversion errors
                    try:
                        self.logger.info(f"DEBUG: Attempting to convert valid sample values to numeric")
                        numeric_samples = pd.to_numeric(valid_sample_values, errors='coerce')
                        self.logger.info(f"DEBUG: Numeric conversion result - NaN count: {numeric_samples.isna().sum()}")
                        
                        # Check for infinite values and replace them
                        if np.any(np.isinf(numeric_samples)):
                            self.logger.info(f"DEBUG: Found infinite values, replacing with NaN")
                            numeric_samples = numeric_samples.replace([np.inf, -np.inf], np.nan)
                        
                        filled_samples = numeric_samples.fillna(0)
                        self.logger.info(f"DEBUG: After fillna - NaN count: {filled_samples.isna().sum()}")
                        
                        # Ensure all values are finite before converting to int
                        if not np.all(np.isfinite(filled_samples)):
                            raise ValueError("Non-finite values found after conversion")
                        
                        onsets_samples[valid_samples] = filled_samples.astype(int).values
                        self.logger.info(f"DEBUG: Successfully converted valid samples to int")
                    except (ValueError, TypeError) as e:
                        self.logger.info(f"DEBUG: Sample conversion failed with error: {e}")
                        # If conversion fails, use onset calculation for all
                        onsets_samples = (events_df['onset'] * raw.info['sfreq']).round().astype(int).values
                        self.logger.info(f"DEBUG: Using onset calculation for all samples")
                    else:
                        # Calculate onset-based samples for invalid entries
                        self.logger.info(f"DEBUG: Calculating onset-based samples for invalid entries")
                        invalid_onsets = events_df.loc[~valid_samples, 'onset']
                        # Ensure onset values are finite
                        if not np.all(np.isfinite(invalid_onsets)):
                            self.logger.info(f"DEBUG: Found non-finite onset values, replacing with 0")
                            invalid_onsets = invalid_onsets.replace([np.inf, -np.inf], 0)
                        onsets_samples[~valid_samples] = (invalid_onsets * raw.info['sfreq']).round().astype(int).values
                else:
                    # All values are NaN, use onset calculation
                    self.logger.info(f"DEBUG: All sample values are NaN, using onset calculation")
                    onset_values = events_df['onset']
                    # Ensure onset values are finite
                    if not np.all(np.isfinite(onset_values)):
                        self.logger.info(f"DEBUG: Found non-finite onset values, replacing with 0")
                        onset_values = onset_values.replace([np.inf, -np.inf], 0)
                    onsets_samples = (onset_values * raw.info['sfreq']).round().astype(int).values
            else:
                self.logger.info(f"DEBUG: Sample column not present or all NaN, using onset calculation")
                onset_values = events_df['onset']
                # Ensure onset values are finite
                if not np.all(np.isfinite(onset_values)):
                    self.logger.info(f"DEBUG: Found non-finite onset values, replacing with 0")
                    onset_values = onset_values.replace([np.inf, -np.inf], 0)
                onsets_samples = (onset_values * raw.info['sfreq']).round().astype(int).values

            self.logger.info(f"DEBUG: Final onsets_samples shape: {onsets_samples.shape}")
            self.logger.info(f"DEBUG: Final onsets_samples range: {onsets_samples.min()} to {onsets_samples.max()}")
            self.logger.info(f"DEBUG: Final trigger_series shape: {trigger_series.shape}")
            
            events = np.column_stack([onsets_samples, np.zeros(len(onsets_samples), dtype=int), trigger_series])
            self.logger.info(f"DEBUG: Final events array shape: {events.shape}")
        else:
            # Fallback: extract from annotations if present
            events, event_id = mne.events_from_annotations(raw)
        
        self.logger.info(f"Loaded {len(events)} events for {subject}")
        return events, event_id
    
    def save_preprocessed(self, raw: mne.io.Raw, subject: str, stage: str) -> Path:
        """
        Save preprocessed data at specific stage
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Preprocessed raw data
        subject : str
            Subject ID
        stage : str
            Processing stage (e.g., 'after_filtering', 'after_ica')
            
        Returns:
        --------
        save_path : Path
            Path where data was saved
        """
        stage_dir = self.preprocessed_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = stage_dir / f"{subject}_{stage}_raw.fif"
        raw.save(save_path, overwrite=True)
        
        self.logger.info(f"Saved {stage} data for {subject} to {save_path}")
        return save_path
    
    def load_preprocessed(self, subject: str, stage: str) -> mne.io.Raw:
        """
        Load preprocessed data from specific stage
        
        Parameters:
        -----------
        subject : str
            Subject ID
        stage : str
            Processing stage
            
        Returns:
        --------
        raw : mne.io.Raw
            Preprocessed raw data
        """
        load_path = self.preprocessed_dir / stage / f"{subject}_{stage}_raw.fif"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {load_path}")
        
        raw = mne.io.read_raw_fif(load_path, preload=True)
        self.logger.info(f"Loaded {stage} data for {subject}")
        
        return raw
    
    def save_epochs(self, epochs: mne.Epochs, subject: str, condition: str = "all") -> Path:
        """
        Save epoched data
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Epoched data
        subject : str
            Subject ID
        condition : str
            Condition name
            
        Returns:
        --------
        save_path : Path
            Path where epochs were saved
        """
        epochs_dir = self.derivatives_dir / "epochs"
        epochs_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = epochs_dir / f"{subject}_{condition}_epo.fif"
        epochs.save(save_path, overwrite=True)
        
        self.logger.info(f"Saved epochs for {subject} ({condition}) to {save_path}")
        return save_path
    
    def load_epochs(self, subject: str, condition: str = "all") -> mne.Epochs:
        """
        Load epoched data
        
        Parameters:
        -----------
        subject : str
            Subject ID
        condition : str
            Condition name
            
        Returns:
        --------
        epochs : mne.Epochs
            Epoched data
        """
        load_path = self.derivatives_dir / "epochs" / f"{subject}_{condition}_epo.fif"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Epochs not found: {load_path}")
        
        epochs = mne.read_epochs(load_path, preload=True)
        self.logger.info(f"Loaded epochs for {subject} ({condition})")
        
        return epochs

    def list_subject_sessions_runs(self, subject: str, task: str = 'gonogo') -> List[Tuple[str, str, Path]]:
        """Return list of (session, run, eeg_file) available for a subject."""
        results: List[Tuple[str, str, Path]] = []
        subj_dir = self.raw_dir / subject
        if not subj_dir.exists():
            return results
        for ses_dir in sorted(subj_dir.glob('ses-*')):
            eeg_dir = ses_dir / 'eeg'
            if not eeg_dir.exists():
                continue
            for eeg_file in sorted(eeg_dir.glob(f"{subject}_{ses_dir.name}_task-{task}_run-*_eeg.set")):
                parts = eeg_file.stem.split('_')
                run_token = next((p for p in parts if p.startswith('run-')), None)
                if run_token is not None:
                    results.append((ses_dir.name, run_token, eeg_file))
        return results

def create_event_mapping() -> Dict[str, int]:
    """Create event ID mapping based on Delorme et al. 2018"""
    return {
        'familiar_animal': 1,
        'familiar_non_animal': 2,
        'new_animal': 3,
        'new_non_animal': 4
    }

def get_condition_mapping() -> Dict[str, List[str]]:
    """Get condition groupings for analysis"""
    return {
        'familiar': ['familiar_animal', 'familiar_non_animal'],
        'new': ['new_animal', 'new_non_animal'],
        'animal': ['familiar_animal', 'new_animal'],
        'non_animal': ['familiar_non_animal', 'new_non_animal']
    }
