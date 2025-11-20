# Raw Data Preprocessing Status Investigation

**Date**: October 8, 2025  
**Dataset**: OpenNeuro ds002680 (Delorme et al., 2018)  
**Investigator**: Triggered by user observation of 50 Hz dip in PSD

## Summary

✅ **CONFIRMED**: The "raw" data from OpenNeuro has been **pre-filtered** with a 50 Hz notch filter.

## Evidence

### 1. Spectral Analysis Results

Analysis of `sub-015_ses-01_task-gonogo_run-10_eeg.set`:

- **Power at 45±2 Hz**: 2.36e-12 V²/Hz
- **Power at 50±1 Hz**: 5.16e-15 V²/Hz  ← **Extremely low**
- **Power at 55±2 Hz**: 1.99e-12 V²/Hz
- **50 Hz / Neighbor ratio**: **0.002** (should be ~1.0 for unfiltered data)

**Interpretation**: The power at 50 Hz is **500 times lower** than neighboring frequencies, indicating a notch filter was applied to remove European power line noise.

### 2. Metadata Check

**JSON Sidecar** (`sub-015_ses-01_task-gonogo_run-10_eeg.json`):
```json
{
  "PowerLineFrequency": 50,
  "SoftwareFilters": {
    "FilterDescription": {
      "Description": "n/a"
    }
  }
}
```

**MNE Info**:
- Highpass: 0.0 Hz (documented as none)
- Lowpass: 500.0 Hz (just Nyquist frequency, not actual filter)

**Interpretation**: While metadata says "n/a" for software filters, the spectral analysis clearly shows filtering was applied but not documented in the BIDS metadata.

### 3. Dataset Description

- **Dataset**: ds002680 version 1.2.0
- **Authors**: Arnaud Delorme
- **Institution**: Paul Sabatier University, Toulouse, France
- **Recording**: Neuroscan Synamps 1 (model 5083), 1000 Hz sampling rate
- **Line noise**: 50 Hz (European standard)

## Implications for Analysis

### What This Means

1. **No need to apply 50 Hz notch filter again** - it's already been done
2. **Applying it again could create artifacts** or over-filter the data
3. **Bandpass filtering is still appropriate** for analysis-specific frequency ranges

### Recommended Preprocessing Pipeline

Given that 50 Hz notch filter has already been applied:

```yaml
preprocessing:
  filter:
    l_freq: 0.2   # High-pass for slow drifts
    h_freq: 100.0  # Low-pass (changed from 512 Hz due to Nyquist limit)
    notch_freq: null  # ❌ SKIP - already applied
```

**Updated pipeline steps**:
1. ✅ **Bandpass filter**: 0.2-100 Hz
2. ❌ **Notch filter**: Skip (already applied)
3. ✅ **Re-referencing**: Average reference
4. ✅ **ICA**: For eye blink and muscle artifacts

## Why Was This Not Documented?

Possible reasons:
1. Applied during data acquisition (hardware filter)
2. Applied during EEGLAB export but not tracked in BIDS metadata
3. Common practice in 2018 before strict BIDS compliance
4. Considered part of "standard" acquisition protocol

## Recommendations

### For Current Analysis (Rheumatoid Arthritis Research Context)

The pre-applied 50 Hz notch filter is appropriate and beneficial:
- ✅ Removes line noise artifacts
- ✅ Improves signal quality for ERP analysis
- ✅ Standard practice for European EEG data

### Pipeline Adjustment

**Modify `notebooks/01_preprocessing_pipeline.ipynb`**:

In the `apply_filtering` method, **remove or comment out** the notch filter section:

```python
# SKIP - Already applied to raw data
# raw_filtered.notch_filter(freqs=50, picks='eeg', 
#                          method='iir', 
#                          iir_params=dict(order=4, ftype='butter'),
#                          verbose=False)
```

## Verification

You can verify this finding by:
1. Loading any raw .set file
2. Computing PSD from 40-60 Hz
3. Looking for a dip at 50 Hz

**Code snippet**:
```python
raw = mne.io.read_raw_eeglab('path/to/file.set', preload=True)
spectrum = raw.compute_psd(fmin=40, fmax=60)
spectrum.plot()  # Will show dip at 50 Hz
```

## References

- Delorme, A., et al. (2018). "Briefly Flashed Scenes Can Be Stored in Long-Term Memory." *Frontiers in Neuroscience*, 12, 688.
- OpenNeuro Dataset: https://openneuro.org/datasets/ds002680
- Power line frequency (France): 50 Hz

