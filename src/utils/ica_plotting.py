"""
ICA Component Plotting Utilities

This module provides utilities for plotting ICA components with comprehensive
6-panel visualizations for artifact identification.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import signal


def _compute_roi_snr(raw_obj, raw_cleaned_obj, events_array, roi_map):
    """Compute SNR for supplied ROIs before/after ICA removal."""
    if (
        roi_map is None
        or len(roi_map) == 0
        or events_array is None
        or len(events_array) == 0
    ):
        return {}

    def _snr_for_raw(raw_instance):
        snr_values = {}
        try:
            epochs = mne.Epochs(
                raw_instance,
                events_array,
                event_id=None,
                tmin=-0.2,
                tmax=0.6,
                baseline=None,
                preload=True,
                event_repeated="drop",
                verbose=False,
            )
        except Exception:
            return {}

        if len(epochs) == 0:
            return {}

        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        times = epochs.times
        erp = data.mean(axis=0)

        signal_window = (times >= 0) & (times <= 0.6)
        baseline_window = (times >= -0.2) & (times <= 0)

        if not np.any(signal_window) or not np.any(baseline_window):
            return {}

        for roi_label, roi_channels in roi_map.items():
            if not roi_channels:
                continue

            picks = mne.pick_channels(raw_instance.ch_names, roi_channels)
            if len(picks) == 0:
                continue

            roi_signal = np.var(erp[picks][:, signal_window], axis=1)
            roi_noise = np.var(erp[picks][:, baseline_window], axis=1)

            if roi_noise.size == 0 or np.any(np.isnan(roi_noise)):
                continue

            snr_db = np.mean(10 * np.log10((roi_signal + 1e-12) / (roi_noise + 1e-12)))
            snr_values[roi_label] = snr_db
        return snr_values

    snr_before = _snr_for_raw(raw_obj)
    snr_after = _snr_for_raw(raw_cleaned_obj) if raw_cleaned_obj is not None else {}

    combined = {}
    for roi in roi_map.keys():
        before_val = snr_before.get(roi)
        after_val = snr_after.get(roi)
        if before_val is None and after_val is None:
            continue
        combined[roi] = (before_val, after_val)
    return combined


def _ensure_unique_events(events_array):
    """Make sure event sample indices are unique for Epoch creation."""
    if events_array is None:
        return None

    events_array = np.asarray(events_array)
    if events_array.size == 0:
        return events_array

    if events_array.ndim != 2 or events_array.shape[1] < 3:
        warnings.warn(
            "Events array did not have shape (n_events, 3); skipping "
            "deduplication.",
            RuntimeWarning,
            stacklevel=2,
        )
        return events_array

    unique_samples, unique_indices = np.unique(
        events_array[:, 0], return_index=True
    )
    if len(unique_samples) == len(events_array):
        return events_array

    warnings.warn(
        "Event time samples were not unique; keeping the first occurrence at "
        "each sample for ICA visualization.",
        RuntimeWarning,
        stacklevel=2,
    )
    dedup_indices = np.sort(unique_indices)
    return events_array[dedup_indices]


def plot_component_comprehensive(
    ica, raw, comp_idx, events=None, data_loader=None, roi_config=None
):
    """
    Plot comprehensive component properties in a 6-panel layout.
    
    This function creates a 2x3 grid showing:
    1. Scalp topography (top-left)
    2. Component time series preview (top-middle)
    3. Power spectrum 3-40 Hz (top-right)
    4. Power spectrum 3-80 Hz (bottom-left)
    5. ERP image heatmap (bottom-middle)
    6. Average ERP (bottom-right)
    + ROI-level SNR table (bottom of top-middle column)
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data
    comp_idx : int
        Component index to plot
    events : array or tuple, optional
        Events array or (events, event_id) tuple
    data_loader : EEGDataLoader, optional
        Data loader to load events if not provided
    roi_config : dict, optional
        ROI configuration containing 'frontal' and 'parieto_occipital' channel lists
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    
    # Load events if not provided
    event_id = None
    if events is None:
        try:
            # Try to get events from raw annotations first
            if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
                events, event_id = mne.events_from_annotations(raw)
            elif data_loader is not None:
                # Try data_loader as fallback
                events = data_loader.load_events()
        except Exception as e:
            # Silently fail - events will remain None
            events = None
    elif isinstance(events, tuple):
        events, event_id = events

    events_array = _ensure_unique_events(events)

    if not roi_config:
        try:
            config_path = Path(__file__).resolve().parents[2] / "config" / "analysis_config.yaml"
            with config_path.open("r") as cfg_file:
                config_data = yaml.safe_load(cfg_file)
            roi_config = config_data.get("erp_analysis", {}).get("roi", {})
        except Exception:
            roi_config = None

    roi_map = None
    if roi_config:
        roi_map = {
            "Frontal ROI": roi_config.get("frontal", []),
            "Parieto-Occipital ROI": roi_config.get("parieto_occipital", []),
        }

    if events_array is not None and len(events_array) > 0:
        if event_id is not None:
            events = (events_array, event_id)
        else:
            events = events_array
    else:
        events = None
        events_array = None

    snr_summary = {}
    snr_error = None
    if roi_map and events_array is not None and len(events_array) > 0:
        try:
            raw_full_clean = raw.copy()
            raw_full_clean.load_data()
            ica.apply(raw_full_clean, exclude=[comp_idx])
            snr_summary = _compute_roi_snr(raw, raw_full_clean, events_array, roi_map)
        except Exception as exc:
            snr_error = str(exc)
    else:
        if not roi_map:
            snr_error = "ROI configuration unavailable"
        elif events_array is None or len(events_array) == 0:
            snr_error = "No events available"

    snr_text_lines = []
    if snr_summary:
        snr_text_lines.append("ROI SNR (dB) after removing this IC")
        for roi_label, (before_val, after_val) in snr_summary.items():
            before_txt = f"{before_val:.2f}" if before_val is not None else "—"
            after_txt = f"{after_val:.2f}" if after_val is not None else "—"
            delta_txt = (
                f"{after_val - before_val:+.2f}"
                if before_val is not None and after_val is not None
                else "—"
            )
            snr_text_lines.append(f"{roi_label}: before {before_txt} | after {after_txt} | Δ {delta_txt}")
    else:
        snr_text_lines.append("SNR summary unavailable")
        if snr_error:
            snr_text_lines.append(f"({snr_error})")
    
    # Get component data
    sources = ica.get_sources(raw)
    component_data = sources.get_data()[comp_idx, :]
    component_times = sources.times
    
    # Create figure with 2x3 layout (top-middle split into two rows)
    fig = plt.figure(figsize=(18, 10))
    outer_gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.35)
    
    # 1. Top-left: Scalp Topography with variance
    ax1 = fig.add_subplot(outer_gs[0, 0])
    mne.viz.plot_topomap(
        ica.get_components()[:, comp_idx],
        raw.info,
        axes=ax1,
        show=False,
    )
    title_lines = [f'IC {comp_idx} Topography']
    var_ratio = getattr(ica, 'pca_explained_variance_ratio_', None)
    var_values = getattr(ica, 'pca_explained_variance_', None)
    if var_ratio is not None and len(var_ratio) > comp_idx:
        title_lines.append(f'Var share: {var_ratio[comp_idx] * 100:.2f}%')
    elif var_values is not None and len(var_values) > comp_idx:
        total_var = np.sum(var_values)
        if total_var > 0:
            title_lines.append(f'Var share: {var_values[comp_idx] / total_var * 100:.2f}%')
    if var_values is not None and len(var_values) > comp_idx:
        title_lines.append(f'Var value: {var_values[comp_idx]:.3f}')
    ax1.set_title('\n'.join(title_lines))
    ax1.text(
        0.5,
        1.08,
        "\n".join(snr_text_lines),
        transform=ax1.transAxes,
        ha='center',
        va='bottom',
        fontsize=9,
    )
    
    # 2. Top-middle: Component Time Series (5s preview at max power)
    top_mid_gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[1.0, 1.0], hspace=0.25
    )
    ax2 = fig.add_subplot(top_mid_gs[0, 0])
    ax2b = fig.add_subplot(top_mid_gs[1, 0])
    preview_duration = 5.0  # seconds
    preview_samples = int(preview_duration * raw.info['sfreq'])
    
    # Find the moment of maximum power (RMS in sliding window)
    window_samples = int(0.5 * raw.info['sfreq'])  # 0.5 second window
    max_power_idx = 0
    max_power = 0
    
    for i in range(len(component_data) - window_samples):
        window_data = component_data[i:i + window_samples]
        window_power = np.sqrt(np.mean(window_data**2))  # RMS power
        if window_power > max_power:
            max_power = window_power
            max_power_idx = i
    
    # Center the preview around the maximum power moment
    start_idx = max(0, max_power_idx - preview_samples // 2)
    end_idx = min(len(component_data), start_idx + preview_samples)
    start_idx = max(0, end_idx - preview_samples)  # Adjust if near end
    
    preview_times = component_times[start_idx:end_idx]
    preview_data = component_data[start_idx:end_idx]
    
    ax2.plot(preview_times, preview_data * 1e6, 'b-', linewidth=0.8)  # Convert to µV
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('µV')
    ax2.set_title(f'Component Time Series (5.0s at max power)')
    ax2.grid(True, alpha=0.3)
    
    # Add event markers within preview window
    if events_array is not None and len(events_array) > 0:
        event_times = events_array[:, 0] / raw.info['sfreq']
        for event_time in event_times:
            if preview_times[0] <= event_time <= preview_times[-1]:
                ax2.axvline(
                    x=event_time,
                    color='red',
                    alpha=0.7,
                    linestyle='--',
                    linewidth=1,
                )
    
    # 2b. Top-middle (bottom): Placeholder encouraging channel comparison button
    ax2b.axis('off')
    ax2b.text(
        0.5,
        0.5,
        "Use the 'Compare All Channels' button\nfor a before/after overlay",
        va='center',
        ha='center',
        fontsize=9,
        transform=ax2b.transAxes,
    )


    # 3. Top-right: Power Spectrum (3-40 Hz)
    ax3 = fig.add_subplot(outer_gs[0, 2])
    freqs, psd = signal.welch(component_data, fs=raw.info['sfreq'], 
                           nperseg=2048, noverlap=1024)
    freq_mask = (freqs >= 3) & (freqs <= 40)
    freqs_filtered = freqs[freq_mask]
    psd_filtered = psd[freq_mask]
    psd_db = 10 * np.log10(psd_filtered)
    
    ax3.plot(freqs_filtered, psd_db, 'r-', linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Log Power (10*log10(µV²/Hz))')
    ax3.set_title('Activity Power Spectrum')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([3, 40])
    ax3.set_ylim([psd_db.min() - 5, psd_db.max() + 5])
    
    # 4. Bottom-left: Power Spectrum (3-80 Hz)
    ax4 = fig.add_subplot(outer_gs[1, 0])
    freq_mask = (freqs >= 3) & (freqs <= 80)
    freqs_filtered = freqs[freq_mask]
    psd_filtered = psd[freq_mask]
    psd_db = 10 * np.log10(psd_filtered)
    
    ax4.plot(freqs_filtered, psd_db, 'g-', linewidth=2)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Log Power (10*log10(µV²/Hz))')
    ax4.set_title('Activity Power Spectrum (Extended)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([3, 80])
    ax4.set_ylim([psd_db.min() - 5, psd_db.max() + 5])
    
    # 5. Bottom-middle: ERP Image (Heatmap)
    ax5 = fig.add_subplot(outer_gs[1, 1])
    if events_array is not None and len(events_array) > 0:
        try:
            epochs = mne.Epochs(
                raw,
                events_array,
                tmin=-0.2,
                tmax=1.0,
                baseline=(None, 0),
                preload=True,
                verbose=False,
            )
            component_epochs = ica.get_sources(epochs).get_data()[:, comp_idx, :]

            im = ax5.imshow(
                component_epochs * 1e6,
                aspect='auto',
                origin='lower',
                extent=[epochs.times[0], epochs.times[-1], 0, len(epochs)],
                cmap='RdBu_r',
            )
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Trial')
            ax5.set_title('ERP Image (Trial-by-Trial)')
            ax5.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            plt.colorbar(im, ax=ax5, shrink=0.8, label='µV')
        except Exception as e:
            ax5.text(0.5, 0.5, f'ERP image failed:\n{str(e)[:40]}', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('ERP Image (Error)')
    else:
        ax5.text(0.5, 0.5, 'No events available', ha='center', va='center', 
                transform=ax5.transAxes)
        ax5.set_title('ERP Image (No Events)')
    
    # 6. Bottom-right: Average ERP
    ax6 = fig.add_subplot(outer_gs[1, 2])
    if events_array is not None and len(events_array) > 0:
        try:
            epochs = mne.Epochs(
                raw,
                events_array,
                tmin=-0.2,
                tmax=1.0,
                baseline=(None, 0),
                preload=True,
                verbose=False,
            )
            component_epochs = ica.get_sources(epochs).get_data()[:, comp_idx, :]
            avg_erp = np.mean(component_epochs, axis=0) * 1e6

            ax6.plot(epochs.times, avg_erp, 'b-', linewidth=2)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('µV')
            ax6.set_title('Average ERP')
            ax6.grid(True, alpha=0.3)
            ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        except Exception as e:
            ax6.text(0.5, 0.5, f'Average ERP failed:\n{str(e)[:40]}', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Average ERP (Error)')
    else:
        ax6.text(0.5, 0.5, 'No events available', ha='center', va='center', 
                transform=ax6.transAxes)
        ax6.set_title('Average ERP (No Events)')
    
    fig.suptitle(
        f'Component {comp_idx} - Comprehensive View',
        fontsize=14,
        fontweight='bold',
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
