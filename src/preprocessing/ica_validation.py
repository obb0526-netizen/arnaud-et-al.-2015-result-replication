"""
ICA validation utilities.

Provides before/after ICA signal-to-noise ratio (SNR) metrics computed per
subject, session, and run using the ERP variance-based formulation defined in
the project plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from ..utils.data_loader import EEGDataLoader


logger = logging.getLogger(__name__)


def _case_insensitive_pick(ch_names: Iterable[str], requested: Iterable[str]) -> List[str]:
    """Return channel names present in ch_names matching requested (case-insensitive)."""
    available_map = {name.upper(): name for name in ch_names}
    picks = []
    for target in requested:
        canonical = available_map.get(target.upper())
        if canonical:
            picks.append(canonical)
    return picks


def _resolve_after_ica_file(pre_path: Path) -> Optional[Path]:
    """
    Find the ICA-cleaned FIF file corresponding to the provided pre-ICA path.

    Parameters
    ----------
    pre_path : Path
        Path to the pre-ICA (after_rereferencing) FIF file.

    Returns
    -------
    Optional[Path]
        Path to the ICA-cleaned FIF if found, otherwise None.
    """
    if "after_rereferencing" not in pre_path.as_posix():
        return None

    subject_dir = pre_path.parents[1]  # .../after_rereferencing/sub-XXX/ses-YY
    root = subject_dir.parents[1] / "after_ica"

    if not root.exists():
        return None

    ses_dir = root / subject_dir.name / pre_path.parent.name
    if not ses_dir.exists():
        return None

    stem = pre_path.name.replace("after_rereferencing", "ica").replace(".fif", "")

    candidates = sorted(
        ses_dir.glob(f"{stem}*cleaned*.fif"), key=lambda p: len(p.name)
    )
    if candidates:
        return candidates[0]

    # Session-level outputs (no run token)
    runless_stem = stem.split("_run-")[0] if "_run-" in stem else stem
    session_candidates = sorted(
        ses_dir.glob(f"{runless_stem}_*cleaned*.fif"), key=lambda p: len(p.name)
    )
    if session_candidates:
        return session_candidates[0]

    fallback = ses_dir / f"{stem}_cleaned.fif"
    if fallback.exists():
        return fallback

    fallback_session = ses_dir / f"{runless_stem}_preprocessed_ica_cleaned.fif"
    return fallback_session if fallback_session.exists() else None


@dataclass
class SNROptions:
    """Options controlling SNR computation."""

    baseline: Tuple[float, float]
    post_window: Tuple[float, float]
    reject_params: Optional[Dict] = None


class ICAValidator:
    """
    Compute before/after ICA validation metrics.

    Currently implements ERP variance-based SNR (in dB) for configured ROIs.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.loader = EEGDataLoader()

        epoch_cfg = config["preprocessing"]["epoching"]
        self.snr_options = SNROptions(
            baseline=tuple(epoch_cfg["baseline"]),
            post_window=(0.0, epoch_cfg["tmax"]),
            reject_params=config["preprocessing"].get("reject_criteria"),
        )

        self.roi_config = config["erp_analysis"]["roi"]

    def compute_snr_summary(
        self,
        subjects: Optional[Iterable[str]] = None,
        summary_csv: Optional[Path] = None,
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Compute before/after ICA SNR for each subject/session/run/ROI.

        Parameters
        ----------
        subjects : Optional[Iterable[str]]
            Subset of subjects to process. Defaults to config selection.
        summary_csv : Optional[Path]
            Preprocessing summary CSV listing runs. Defaults to the generated summary.
        save_path : Optional[Path]
            If provided, writes the resulting dataframe to CSV.

        Returns
        -------
        pd.DataFrame
            Long-format dataframe with columns:
            [`subject`, `session`, `run`, `roi`, `snr_before_db`, `snr_after_db`, `snr_delta_db`]
        """

        if subjects is None:
            subjects = self.config["subjects"]["selected"]
        subjects = list(subjects)

        if summary_csv is None:
            summary_csv = (
                self.loader.preprocessed_dir / "preprocessing_summary.csv"
            )

        if not Path(summary_csv).exists():
            raise FileNotFoundError(f"Preprocessing summary not found: {summary_csv}")

        runs_df = pd.read_csv(summary_csv)
        runs_df = runs_df[runs_df["subject"].isin(subjects)]

        if runs_df.empty:
            raise ValueError("No runs available for specified subjects.")

        records: List[Dict] = []

        for _, row in runs_df.iterrows():
            pre_path = Path(row["reref_path"])
            post_path = _resolve_after_ica_file(pre_path)

            if not pre_path.exists():
                logger.warning("Pre-ICA file missing: %s", pre_path)
                continue

            if post_path is None or not post_path.exists():
                logger.warning(
                    "Skipping %s (no matching ICA-cleaned file found)", pre_path.name
                )
                continue

            try:
                snr_before = self._compute_roi_snr(pre_path)
                snr_after = self._compute_roi_snr(post_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed SNR computation for %s: %s", pre_path, exc)
                continue

            for roi_name in sorted(self.roi_config.keys()):
                row_before = snr_before.get(roi_name)
                row_after = snr_after.get(roi_name)

                if row_before is None or row_after is None:
                    continue

                records.append(
                    {
                        "subject": row["subject"],
                        "session": row["session"],
                        "run": row["run"],
                        "roi": roi_name,
                        "snr_before_db": row_before,
                        "snr_after_db": row_after,
                        "snr_delta_db": row_after - row_before,
                    }
                )

        if not records:
            raise RuntimeError("No SNR records computed. Check data availability.")

        results = pd.DataFrame(records)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(save_path, index=False)

        return results

    def _compute_roi_snr(self, raw_path: Path) -> Dict[str, float]:
        """Compute SNR per ROI for the raw FIF file."""
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(events) == 0:
            raise RuntimeError(f"No events found in {raw_path}")

        roi_snr: Dict[str, float] = {}

        for roi_name, roi_channels in self.roi_config.items():
            picks = _case_insensitive_pick(raw.ch_names, roi_channels)
            if not picks:
                logger.debug(
                    "Skipping ROI %s for %s (channels not found)", roi_name, raw_path
                )
                continue

            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=self.config["preprocessing"]["epoching"]["tmin"],
                tmax=self.config["preprocessing"]["epoching"]["tmax"],
                baseline=self.snr_options.baseline,
                picks=picks,
                preload=True,
                reject=None,
                verbose="ERROR",
            )

            evoked = epochs.average()
            roi_signal = np.mean(evoked.data, axis=0)
            times = evoked.times

            baseline_mask = (times >= self.snr_options.baseline[0]) & (
                times <= self.snr_options.baseline[1]
            )
            post_mask = (times >= self.snr_options.post_window[0]) & (
                times <= self.snr_options.post_window[1]
            )

            if np.sum(baseline_mask) < 2 or np.sum(post_mask) < 2:
                logger.debug(
                    "Insufficient samples for ROI %s in %s", roi_name, raw_path
                )
                continue

            noise_var = np.var(roi_signal[baseline_mask], ddof=1)
            signal_var = np.var(roi_signal[post_mask], ddof=1)

            if noise_var <= 0 or signal_var <= 0:
                logger.debug(
                    "Non-positive variance for ROI %s in %s (noise=%s, signal=%s)",
                    roi_name,
                    raw_path,
                    noise_var,
                    signal_var,
                )
                continue

            snr_db = 10.0 * np.log10(signal_var / noise_var)
            roi_snr[roi_name] = float(snr_db)

        return roi_snr

