"""Shared scalar/batch release dispatch under current physical constraints."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS


FLOW_TOLERANCE = 1e-8

# Release-feasibility ablations. ``backward`` is the original kernel: the
# downstream storage envelope is propagated upstream so a joint release choice
# is feasible for every reservoir at once. ``forward_local`` disables that
# propagation and keeps only each reservoir's own storage interval, applied as
# releases are computed upstream to downstream.
FEASIBILITY_MODES = ('backward', 'forward_local')
DEFAULT_FEASIBILITY_MODE = 'backward'


def validate_feasibility_mode(mode):
    """Return a supported propagation mode; reject everything else."""
    if not isinstance(mode, str) or mode not in FEASIBILITY_MODES:
        raise ValueError('feasibility_mode must be one of '+str(list(FEASIBILITY_MODES)))
    return mode


class InfeasibleCascade(RuntimeError):
    """A physical infeasibility that may carry machine-readable evidence."""

    def __init__(self, message, conflicts=None):
        super().__init__(message)
        self.conflicts = [] if conflicts is None else [dict(value) for value in conflicts]


def _release_conflicts(lower, upper, failed, reason):
    records = []
    for row, reservoir in zip(*np.where(failed)):
        records.append(dict(row=int(row), reservoir=IDS[int(reservoir)],
            reservoir_index=int(reservoir), reason=reason,
            release_lower_m3s=float(lower[row, reservoir]),
            release_upper_m3s=float(upper[row, reservoir])))
    return records


def _matrix(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 5 or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite batch of five-vectors')
    if shape is not None and result.shape != shape:
        raise ValueError('All dispatch matrices must have the same shape')
    return result


def _vectors(value, name, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape not in ((5,), shape) or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite five-vector or matching batch')
    return np.broadcast_to(result, shape)


def _backward(storage, external, storage_lower, storage_upper, release_cap, dt):
    """Propagate downstream storage and outlet compatibility upstream."""
    lower, upper = np.zeros_like(storage), release_cap.copy()
    conflicts = np.zeros(storage.shape, dtype=bool)
    for i in range(3, -1, -1):
        lower[:, i] = np.maximum(lower[:, i], lower[:, i+1]-external[:, i+1]+
                                (storage_lower[:, i+1]-storage[:, i+1])/dt)
        upper[:, i] = np.minimum(upper[:, i], upper[:, i+1]-external[:, i+1]+
                                (storage_upper[:, i+1]-storage[:, i+1])/dt)
        conflicts[:, i] |= lower[:, i] > upper[:, i]+FLOW_TOLERANCE
    # Reservoir zero's own storage interval completes the joint feasibility
    # test; subsequent reservoirs have compatible choices by construction.
    first_lower = np.maximum(lower[:, 0], external[:, 0]+(storage[:, 0]-storage_upper[:, 0])/dt)
    first_upper = np.minimum(upper[:, 0], external[:, 0]+(storage[:, 0]-storage_lower[:, 0])/dt)
    conflicts[:, 0] |= first_lower > first_upper+FLOW_TOLERANCE
    return lower, upper, conflicts


def release_from_normalized_action(lower, upper, action):
    """Linearly interpolate a feasible release for a normalized action."""
    return lower + action * (upper - lower)


def dispatch_release(storage, external, actions, storage_lower, storage_upper,
                     release_cap, dt, feasibility_mode=DEFAULT_FEASIBILITY_MODE):
    """Return release dispatch for independent rows, each with five reservoirs.

    Inputs are ``(N,5)`` arrays; release_cap may also be a five-vector.
    All returned arrays retain the leading row dimension, including diagnostics.
    No input is modified. Empty batches are supported.

    ``feasibility_mode='backward'`` is the original downstream-to-upstream
    propagation. ``feasibility_mode='forward_local'`` is the ablation: each
    reservoir is bounded only by its own storage interval and capacity, with
    the resulting upstream release still passed downstream as inflow.
    Infeasible intervals raise in both modes; nothing is clipped or relaxed.
    """
    feasibility_mode = validate_feasibility_mode(feasibility_mode)
    storage = _matrix(storage, 'Storage')
    shape = storage.shape
    external = _matrix(external, 'External inflow', shape)
    actions = _matrix(actions, 'Normalized actions', shape)
    storage_lower = _matrix(storage_lower, 'Lower storage bounds', shape)
    storage_upper = _matrix(storage_upper, 'Upper storage bounds', shape)
    cap = _vectors(release_cap, 'Release capacity', shape)
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0.:
        raise ValueError('The dispatch time step must be finite and positive')
    if ((external < 0.).any() or (actions < 0.).any() or (actions > 1.).any() or
            (cap <= 0.).any() or (storage_lower > storage_upper).any()):
        raise ValueError('Invalid forcing, actions, capacity or storage bounds')
    if feasibility_mode == 'backward':
        physical_lower, physical_upper, physical_conflict = _backward(
            storage, external, storage_lower, storage_upper, cap, dt)
        if physical_conflict.any():
            raise InfeasibleCascade('Empty physically compatible release interval',
                conflicts=_release_conflicts(physical_lower, physical_upper,
                    physical_conflict, 'cascade_storage_compatibility'))
        lower, upper = physical_lower, physical_upper
    else:
        # Forward-only ablation: no downstream storage compatibility is
        # propagated upstream. Each reservoir keeps its own bounds and capacity.
        lower, upper = np.zeros_like(storage), cap.copy()
    inflow, release, low, high = [np.empty(shape) for _ in range(4)]
    controls = np.ones(shape, dtype=np.float32)
    executed = actions.copy()
    empty = np.zeros(shape, dtype=bool)
    for i in range(5):
        inflow[:, i] = external[:, i]+(release[:, i-1] if i else 0.)
        low[:, i] = np.maximum(lower[:, i], inflow[:, i]+(storage[:, i]-storage_upper[:, i])/dt)
        high[:, i] = np.minimum(upper[:, i], inflow[:, i]+(storage[:, i]-storage_lower[:, i])/dt)
        if (low[:, i] > high[:, i]+FLOW_TOLERANCE).any():
            failed = (low[:, i] > high[:, i]+FLOW_TOLERANCE)
            empty[failed, i] = True
            raise InfeasibleCascade('Empty feasible release interval at reservoir '+str(i),
                conflicts=[dict(row=int(row), reservoir=IDS[i], reservoir_index=int(i),
                    reason='reservoir_release_interval', feasibility_mode=feasibility_mode,
                    release_lower_m3s=float(low[row, i]),
                    release_upper_m3s=float(high[row, i]))
                    for row in np.flatnonzero(failed)])
        collapsed = high[:, i]-low[:, i] <= FLOW_TOLERANCE
        controls[collapsed, i] = 0.
        executed[collapsed, i] = .5
        midpoint = np.minimum(cap[:, i], np.maximum(0., .5*(low[:, i]+high[:, i])))
        release[:, i] = np.where(collapsed, midpoint,
                                 release_from_normalized_action(low[:, i], high[:, i], executed[:, i]))
    return dict(inflow=inflow, release=release, low=low, high=high,
                controls=controls, actions=executed, empty_interval=empty,
                forced_interval=(controls == 0),
                feasibility_mode=feasibility_mode,
                physical_low_m3s=low.copy(), physical_high_m3s=high.copy())
