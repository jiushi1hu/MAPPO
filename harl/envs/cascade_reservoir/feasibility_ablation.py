"""Audit data for the release-feasibility propagation ablation.

Collects executed-day evidence only. It never edits a transition, a reward or a
stored state, so recording cannot change training or evaluation.
"""
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.dispatch_kernel import (
    DEFAULT_FEASIBILITY_MODE, validate_feasibility_mode)


LEVEL_NAMES = tuple('level_min_m_'+name for name in IDS) + tuple(
    'level_max_m_'+name for name in IDS)


def _float(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(name+' must be finite')
    return value


class FeasibilityRecorder:
    """Accumulate release-dispatch diagnostics for one episode.

    ``infeasible_days`` counts transitions the kernel refused. Because the
    environment raises on those, the counter is normally zero; it is recorded so
    a failed episode reports the same fields as a completed one.
    """

    def __init__(self, mode=DEFAULT_FEASIBILITY_MODE):
        self.mode = validate_feasibility_mode(mode)
        self.days = 0
        self.infeasible_days = 0
        self.forced_actor_days = 0
        self.empty_interval_days = 0
        self.max_balance_residual_m3 = 0.
        self.release_sum = np.zeros(len(IDS))
        self.interval_fraction_sum = np.zeros(len(IDS))
        self.interval_fraction_min = np.full(len(IDS), np.inf)
        self.level_min = np.full(len(IDS), np.inf)
        self.level_max = np.full(len(IDS), -np.inf)
        self.level_sum = np.zeros(len(IDS))
        self.level_series = []

    def capture(self, transition):
        """Record one executed transition; raises on malformed evidence."""
        dates = str(transition['date'])
        forced = np.asarray(transition['forced_interval'], dtype=bool)
        empty = np.asarray(transition['empty_interval'], dtype=bool)
        release = np.asarray(transition['release'], dtype=np.float64)
        fraction = np.asarray(transition['release_interval_fraction'], dtype=np.float64)
        levels = np.asarray(transition['levels_start'], dtype=np.float64)
        if (forced.shape != (len(IDS),) or empty.shape != (len(IDS),) or
                release.shape != (len(IDS),) or fraction.shape != (len(IDS),) or
                levels.shape != (len(IDS),) or not np.isfinite(levels).all()):
            raise ValueError('Malformed feasibility transition on '+dates)
        if empty.any():
            raise ValueError('Executed an empty release interval on '+dates)
        self.days += 1
        self.forced_actor_days += int(forced.sum())
        self.empty_interval_days += int(empty.sum())
        self.max_balance_residual_m3 = max(self.max_balance_residual_m3,
            abs(_float(transition['system_residual'], 'system_residual')))
        self.release_sum += release
        self.interval_fraction_sum += fraction
        self.interval_fraction_min = np.minimum(self.interval_fraction_min, fraction)
        self.level_min = np.minimum(self.level_min, levels)
        self.level_max = np.maximum(self.level_max, levels)
        self.level_sum += levels
        self.level_series.append(levels.copy())

    def summary(self):
        """Return JSON-safe aggregates for one episode."""
        if not self.days:
            raise ValueError('No executed transitions were recorded')
        days = float(self.days)
        series = np.asarray(self.level_series, dtype=np.float64)
        return dict(feasibility_mode=self.mode, feasibility_days=int(self.days),
            infeasible_days=int(self.infeasible_days),
            infeasible_interval_rate=float(self.infeasible_days)/days,
            forced_actor_days=int(self.forced_actor_days),
            forced_actor_rate=float(self.forced_actor_days)/(days*len(IDS)),
            empty_interval_days=int(self.empty_interval_days),
            empty_interval_rate=float(self.empty_interval_days)/(days*len(IDS)),
            max_balance_residual_m3=_float(self.max_balance_residual_m3,
                                           'max_balance_residual_m3'),
            mean_release_m3s=(self.release_sum/days).tolist(),
            mean_release_interval_fraction=(self.interval_fraction_sum/days).tolist(),
            min_release_interval_fraction=self.interval_fraction_min.tolist(),
            mean_level_m=(self.level_sum/days).tolist(),
            min_level_m=self.level_min.tolist(), max_level_m=self.level_max.tolist(),
            final_level_m=series[-1].tolist(),
            reservoir_ids=list(IDS))


def level_trajectory_summary(recorders):
    """Trajectory spread across seeds, for the level plot."""
    series = [np.asarray(recorder.level_series, dtype=np.float64) for recorder in recorders]
    if not series or any(value.ndim != 2 or value.shape[1] != len(IDS) for value in series):
        raise ValueError('Level trajectories require matching per-seed series')
    return dict(mean=np.stack(series).mean(axis=0).tolist(),
                min=np.stack(series).min(axis=0).tolist(),
                max=np.stack(series).max(axis=0).tolist())
