"""Hard daily water-level ramp constraints and conflict evidence.

The ramp rule is converted to an end-of-day storage interval before release
dispatch. It therefore changes which releases are physically admissible; it
never clips storage, levels, releases, or rewards after execution.
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.dispatch_kernel import InfeasibleCascade


RAMP_LEVEL_TOLERANCE_M = 1e-8
CONFLICT_FIELDS = (
    'seed', 'phase', 'date', 'reservoir', 'reservoir_index', 'reason',
    'window_start_year', 'episode_step', 'policy_update',
    'current_level_m', 'rise_limit_m_per_day', 'fall_limit_m_per_day',
    'ramp_level_lower_m', 'ramp_level_upper_m',
    'existing_level_lower_m', 'existing_level_upper_m', 'ramp_season',
    'effective_level_lower_m', 'effective_level_upper_m',
    'current_storage_m3', 'existing_storage_lower_m3',
    'existing_storage_upper_m3', 'effective_storage_lower_m3',
    'effective_storage_upper_m3')


def _finite_nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be a finite nonnegative number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be a finite nonnegative number') from exc
    if not np.isfinite(result) or result < 0.:
        raise ValueError(name+' must be a finite nonnegative number')
    return result


def _months(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(name+' must be a nonempty list of months')
    result = set()
    for month in value:
        if isinstance(month, (bool, np.bool_)) or not isinstance(month, (int, np.integer)):
            raise ValueError(name+' must contain integer months')
        month = int(month)
        if month < 1 or month > 12 or month in result:
            raise ValueError(name+' must contain unique months in 1..12')
        result.add(month)
    return result


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(name+' must be a mapping')
    return value


def _stage_curve(data, name):
    try:
        points = np.asarray(data.anchors[name], dtype=np.float64)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError('Missing stage-storage anchors for '+name) from exc
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
            not np.isfinite(points).all() or
            (np.diff(points, axis=0) <= 0.).any()):
        raise ValueError('Stage-storage anchors must be finite and strictly increasing: '+name)
    return points


def _levels_from_storage(data, storage):
    storage = np.asarray(storage, dtype=np.float64)
    if storage.shape != (len(IDS),) or not np.isfinite(storage).all():
        raise ValueError('Ramp diagnostics require a finite five-vector of storage')
    result = np.empty(len(IDS), dtype=np.float64)
    for index, name in enumerate(IDS):
        points = _stage_curve(data, name)
        result[index] = np.interp(storage[index], points[:, 1], points[:, 0])
    return result


def _storage_from_levels(data, levels):
    levels = np.asarray(levels, dtype=np.float64)
    if levels.shape != (len(IDS),) or not np.isfinite(levels).all():
        raise ValueError('Ramp diagnostics require a finite five-vector of levels')
    result = np.empty(len(IDS), dtype=np.float64)
    for index, name in enumerate(IDS):
        points = _stage_curve(data, name)
        if (levels[index] < points[0, 0]-RAMP_LEVEL_TOLERANCE_M or
                levels[index] > points[-1, 0]+RAMP_LEVEL_TOLERANCE_M):
            raise ValueError('Water level lies outside its stage-storage curve: '+name)
        result[index] = np.interp(levels[index], points[:, 0], points[:, 1])
    return result


class DailyLevelRampLimits:
    """Validated constant and seasonal rise/fall limits for the five reservoirs."""

    def __init__(self, config=None):
        self.enabled = False
        self.config = None
        self.month_rise = None
        self.month_fall = None
        self.month_season = None
        if config is None:
            return
        config = _mapping(config, 'daily_level_ramp_limits')
        enabled = config.get('enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('daily_level_ramp_limits.enabled must be a boolean')
        self.enabled = enabled
        if not enabled:
            return
        constant = _mapping(config.get('constant', {}), 'daily_level_ramp_limits.constant')
        seasonal = _mapping(config.get('seasonal', {}), 'daily_level_ramp_limits.seasonal')
        if set(constant) & set(seasonal):
            raise ValueError('Ramp schemes must be unique per reservoir')
        if set(constant) | set(seasonal) != set(IDS):
            raise ValueError('Ramp limits must configure exactly WDD/BHT/XLD/XJB/THR')

        rise = np.zeros((13, len(IDS)), dtype=np.float64)
        fall = np.zeros((13, len(IDS)), dtype=np.float64)
        season = np.empty((13, len(IDS)), dtype=object)
        for index, name in enumerate(IDS):
            if name in constant:
                value = _mapping(constant[name], 'constant ramp limits for '+name)
                rise_value = _finite_nonnegative(value.get('rise_m_per_day'),
                                                  name+'.rise_m_per_day')
                fall_value = _finite_nonnegative(value.get('fall_m_per_day'),
                                                  name+'.fall_m_per_day')
                for month in range(1, 13):
                    rise[month, index] = rise_value
                    fall[month, index] = fall_value
                    season[month, index] = 'constant'
                continue

            value = _mapping(seasonal[name], 'seasonal ramp limits for '+name)
            dry_months = _months(value.get('dry_months'), name+'.dry_months')
            flood_months = _months(value.get('flood_months'), name+'.flood_months')
            if dry_months & flood_months or dry_months | flood_months != set(range(1, 13)):
                raise ValueError(name+' dry and flood months must partition the calendar')
            refill_months = _months(value.get('refill_months'), name+'.refill_months')
            dry_limit = _finite_nonnegative(value.get('dry_m_per_day'), name+'.dry_m_per_day')
            flood_limit = _finite_nonnegative(value.get('flood_m_per_day'), name+'.flood_m_per_day')
            refill_rise = _finite_nonnegative(value.get('refill_max_rise_m_per_day'),
                                               name+'.refill_max_rise_m_per_day')
            for month in range(1, 13):
                base = dry_limit if month in dry_months else flood_limit
                base_name = 'dry' if month in dry_months else 'flood'
                rise[month, index] = refill_rise if month in refill_months else base
                fall[month, index] = base
                season[month, index] = 'refill_'+base_name if month in refill_months else base_name

        self.config = dict(config)
        self.month_rise, self.month_fall, self.month_season = rise, fall, season

    def limits_for_date(self, day):
        if not self.enabled:
            raise RuntimeError('Daily level ramp limits are disabled')
        if not isinstance(day, date):
            raise ValueError('Ramp limits require an actual calendar date')
        month = day.month
        return (self.month_rise[month].copy(), self.month_fall[month].copy(),
                self.month_season[month].copy())

    def effective_storage_bounds(self, data, storage, day, storage_lower, storage_upper,
                                 level_lower=None, level_upper=None):
        """Intersect daily/operating limits with the ramp band before dispatch."""
        storage = np.asarray(storage, dtype=np.float64)
        storage_lower = np.asarray(storage_lower, dtype=np.float64)
        storage_upper = np.asarray(storage_upper, dtype=np.float64)
        if (storage.shape != (len(IDS),) or storage_lower.shape != (len(IDS),) or
                storage_upper.shape != (len(IDS),) or
                not np.isfinite(storage).all() or
                not np.isfinite(storage_lower).all() or
                not np.isfinite(storage_upper).all() or
                (storage_lower > storage_upper).any()):
            raise ValueError('Ramp bounds require ordered finite daily five-vectors')

        rise, fall, season = self.limits_for_date(day)
        current_level = _levels_from_storage(data, storage)
        operating_lower = _levels_from_storage(data, np.asarray(data.vmin, dtype=np.float64))
        operating_upper = _levels_from_storage(data, np.asarray(data.vmax, dtype=np.float64))
        if level_lower is None:
            level_lower = _levels_from_storage(data, storage_lower)
        else:
            level_lower = np.asarray(level_lower, dtype=np.float64)
        if level_upper is None:
            level_upper = _levels_from_storage(data, storage_upper)
        else:
            level_upper = np.asarray(level_upper, dtype=np.float64)
        if (level_lower.shape != (len(IDS),) or level_upper.shape != (len(IDS),) or
                not np.isfinite(level_lower).all() or not np.isfinite(level_upper).all()):
            raise ValueError('Ramp diagnostics require finite daily level bounds')

        ramp_lower = current_level-fall
        ramp_upper = current_level+rise
        existing_lower = np.maximum(level_lower, operating_lower)
        existing_upper = np.minimum(level_upper, operating_upper)
        effective_lower = np.maximum(existing_lower, ramp_lower)
        effective_upper = np.minimum(existing_upper, ramp_upper)
        level_empty = effective_lower > effective_upper+RAMP_LEVEL_TOLERANCE_M

        lower = np.full(len(IDS), np.nan)
        upper = np.full(len(IDS), np.nan)
        if not level_empty.any():
            converted_lower = _storage_from_levels(data, effective_lower)
            converted_upper = _storage_from_levels(data, effective_upper)
            lower = np.maximum(storage_lower, converted_lower)
            upper = np.minimum(storage_upper, converted_upper)
            tolerance = np.maximum(1., 1e-10*np.maximum(np.abs(lower), np.abs(upper)))
            storage_empty = lower > upper+tolerance
        else:
            storage_empty = level_empty

        base = dict(date=day.isoformat(), reservoir=None, reservoir_index=None, reason=None,
            current_level_m=current_level, rise_limit_m_per_day=rise,
            fall_limit_m_per_day=fall, ramp_level_lower_m=ramp_lower,
            ramp_level_upper_m=ramp_upper, existing_level_lower_m=existing_lower,
            existing_level_upper_m=existing_upper, ramp_season=season,
            effective_level_lower_m=effective_lower,
            effective_level_upper_m=effective_upper,
            current_storage_m3=storage, existing_storage_lower_m3=storage_lower,
            existing_storage_upper_m3=storage_upper,
            effective_storage_lower_m3=lower,
            effective_storage_upper_m3=upper)
        if level_empty.any() or storage_empty.any():
            conflicts = []
            for index in np.flatnonzero(level_empty | storage_empty):
                if level_empty[index]:
                    if ramp_upper[index] < existing_lower[index]-RAMP_LEVEL_TOLERANCE_M:
                        reason = 'ramp_rise_below_required_next_level'
                    elif ramp_lower[index] > existing_upper[index]+RAMP_LEVEL_TOLERANCE_M:
                        reason = 'ramp_fall_above_required_next_level'
                    elif level_lower[index] > level_upper[index]+RAMP_LEVEL_TOLERANCE_M:
                        reason = 'existing_daily_level_bounds_empty'
                    else:
                        reason = 'ramp_and_daily_level_bounds_disjoint'
                else:
                    reason = 'ramp_storage_interval_empty'
                record = {}
                for key, value in base.items():
                    if key == 'reservoir':
                        record[key] = IDS[index]
                    elif key == 'reservoir_index':
                        record[key] = int(index)
                    elif key == 'reason':
                        record[key] = reason
                    elif key == 'date':
                        record[key] = value
                    elif key == 'ramp_season':
                        record[key] = str(value[index])
                    else:
                        record[key] = float(value[index])
                conflicts.append(record)
            error = InfeasibleCascade(
                'No end-of-day storage satisfies the hard daily level-ramp constraints',
                conflicts=conflicts)
            error.ramp_metadata = self.transition_metadata(base)
            raise error

        return lower, upper, self.transition_metadata(base)

    @staticmethod
    def transition_metadata(metadata):
        """Public, serializable ramp evidence for a transition."""
        return dict(level_ramp_enabled=True, ramp_date=metadata['date'],
            ramp_rise_limit_m_per_day=metadata['rise_limit_m_per_day'].copy(),
            ramp_fall_limit_m_per_day=metadata['fall_limit_m_per_day'].copy(),
            ramp_season=metadata['ramp_season'].astype(str).copy(),
            current_level_m=metadata['current_level_m'].copy(),
            ramp_level_lower_m=metadata['ramp_level_lower_m'].copy(),
            ramp_level_upper_m=metadata['ramp_level_upper_m'].copy(),
            daily_level_lower_m=metadata['existing_level_lower_m'].copy(),
            daily_level_upper_m=metadata['existing_level_upper_m'].copy(),
            effective_level_lower_m=metadata['effective_level_lower_m'].copy(),
            effective_level_upper_m=metadata['effective_level_upper_m'].copy(),
            daily_storage_lower_m3=metadata['existing_storage_lower_m3'].copy(),
            daily_storage_upper_m3=metadata['existing_storage_upper_m3'].copy(),
            ramp_storage_lower_m3=metadata['effective_storage_lower_m3'].copy(),
            ramp_storage_upper_m3=metadata['effective_storage_upper_m3'].copy())


def daily_level_ramp_limits(config):
    """Build the validated ramp rule from an environment configuration."""
    if not isinstance(config, dict):
        raise ValueError('Physical configuration must be a mapping')
    return DailyLevelRampLimits(config.get('daily_level_ramp_limits'))


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return ''
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), '.15g')
    return str(value)


def conflicts_from_exception(exc):
    """Return structured conflict rows, or an empty list for other errors."""
    conflicts = getattr(exc, 'conflicts', None)
    if conflicts is None:
        return []
    if not isinstance(conflicts, (list, tuple)):
        raise ValueError('Exception conflicts must be a list')
    return [dict(value) for value in conflicts if isinstance(value, dict)]


class RampConflictWriter:
    """One per-seed CSV that exists even when no conflict occurred."""

    def __init__(self, path, seed):
        self.path = Path(path)
        self.seed = int(seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=CONFLICT_FIELDS)
        self.writer.writeheader()
        self.stream.flush()

    def write(self, conflicts, phase, **context):
        for conflict in conflicts:
            row = {field:'' for field in CONFLICT_FIELDS}
            row.update({key:_csv_value(value) for key, value in conflict.items()
                        if key in CONFLICT_FIELDS})
            row.update({key:_csv_value(value) for key, value in context.items()
                        if key in CONFLICT_FIELDS})
            row['seed'] = self.seed
            row['phase'] = phase
            if not row['reservoir']:
                row['reservoir'] = 'CASCADE'
            self.writer.writerow(row)
        if conflicts:
            self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def ensure_ramp_conflict_csv(path, seed):
    path = Path(path)
    if path.exists():
        return path
    writer = RampConflictWriter(path, seed)
    writer.close()
    return path
