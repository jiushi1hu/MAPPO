"""Neutral reservoir inventory accounting, without a target or pass/fail test."""
import numpy as np


def storage_change_summary(initial, final=None, prefix=''):
    """Return signed final-minus-initial inventory and its net depletion."""
    if not isinstance(prefix, str) or (prefix and not prefix.isidentifier()):
        raise ValueError('Storage summary prefix must be empty or an identifier')
    start = np.asarray(initial, dtype=np.float64)
    if start.shape != (5,) or not np.isfinite(start).all():
        raise ValueError('Initial storage must be a finite five-vector')
    if final is None:
        return dict({prefix+'initial_storage_m3':start.tolist(), prefix+'final_storage_m3':None,
                    prefix+'storage_change_m3':None, prefix+'net_storage_change_m3':None,
                    prefix+'total_storage_depletion_m3':None,
                    prefix+'total_storage_depletion_billion_m3':None})
    end = np.asarray(final, dtype=np.float64)
    if end.shape != (5,) or not np.isfinite(end).all():
        raise ValueError('Final storage must be a finite five-vector')
    change = end-start
    net = float(change.sum())
    return dict({prefix+'initial_storage_m3':start.tolist(), prefix+'final_storage_m3':end.tolist(),
                prefix+'storage_change_m3':change.tolist(), prefix+'net_storage_change_m3':net,
                prefix+'total_storage_depletion_m3':-net,
                prefix+'total_storage_depletion_billion_m3':-net/1e9})
