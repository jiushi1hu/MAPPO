"""MAPPO advantage preprocessing shared by the optimizer and its diagnostics."""
import numpy as np


NORMALIZATION_LABELS = {
    'all_rows': 'per_reservoir_over_common_rollout_rows',
    'active_rows': 'per_reservoir_over_controllable_rows',
    'none': 'none',
}


def advantage_mode(settings, *, legacy_all_rows=False):
    """Read the explicit mode or resolve a historical boolean setting."""
    mode = settings.get('advantage_normalization')
    if 'advantage_normalization' not in settings:
        return 'all_rows' if settings.get('normalize_shared_advantages', legacy_all_rows) else 'active_rows'
    if mode not in NORMALIZATION_LABELS:
        raise ValueError('advantage_normalization must be all_rows, active_rows, or none')
    return mode


def all_row_advantages(raw):
    """Use separate rollout moments for each final-axis value channel."""
    return (raw - raw.mean(axis=(0, 1), keepdims=True)) / (raw.std(axis=(0, 1), keepdims=True) + 1e-5)


def active_row_advantages(raw, mask):
    """Use controllable rows for one reservoir's moments."""
    selected = raw[mask > 0]
    if not selected.size:
        return np.zeros_like(raw)
    return (raw - selected.mean()) / (selected.std() + 1e-5)
