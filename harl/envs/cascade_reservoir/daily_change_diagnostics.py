"""Observational daily level and release changes from evaluation trajectories.

The diagnostic layer only reads executed end-of-day levels and daily releases.
It never feeds differences back into observations, rewards, or optimization.
Source-year joins remain part of one continuous episode sequence, so a change is
computed across a join and only the very first daily step is missing.
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS


SCHEMA_VERSION = 2
DAILY_CHANGE_FIELDS = (
    'seed', 'date', 'year', 'day_of_year', 'reservoir', 'level_m',
    'delta_level_m', 'abs_delta_level_m', 'release_m3s', 'delta_release_m3s',
    'rise_limit_m_per_day', 'fall_limit_m_per_day', 'within_ramp_limit')
STATISTICS_FIELDS = (
    'seed', 'reservoir', 'sample_count', 'mean_abs_delta_level_m',
    'p95_abs_delta_level_m', 'max_abs_delta_level_m',
    'max_abs_allowed_level_change_m', 'ramp_violation_count')
LEVEL_BOUND_FIELDS = ('next_state_level_lower_m', 'next_state_level_upper_m')


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    return int(value)


def _calendar_date(value):
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError('trajectory dates must be ISO calendar dates') from exc


def _matrix(value, name, days=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(name+' must be a finite (days, 5) matrix') from exc
    if result.ndim != 2 or result.shape[1] != len(IDS) or (days is not None and len(result) != days):
        raise ValueError(name+' must have shape (days, 5)')
    if not np.isfinite(result).all():
        raise ValueError(name+' must contain only finite executed values')
    return result


def _optional_limits(value, name, days):
    if value is None:
        return None
    result = _matrix(value, name, days)
    if (result < 0.).any():
        raise ValueError(name+' must be nonnegative')
    return result


def ramp_compliance(level_change, rise_limit, fall_limit):
    """Return one explicit within-limit flag per reservoir-day."""
    level_change = np.asarray(level_change, dtype=np.float64)
    if (level_change.ndim != 2 or level_change.shape[1] != len(IDS) or
            np.isinf(level_change).any()):
        raise ValueError('delta_level_m must be a (days, 5) matrix with no infinities')
    rise_limit = _matrix(rise_limit, 'rise_limit_m_per_day', len(level_change))
    fall_limit = _matrix(fall_limit, 'fall_limit_m_per_day', len(level_change))
    result = np.zeros(level_change.shape, dtype=bool)
    finite = np.isfinite(level_change)
    result[finite] = ((level_change[finite] <= rise_limit[finite]+1e-8) &
                      (level_change[finite] >= -fall_limit[finite]-1e-8))
    return result


def validate_daily_series(dates, levels, releases, expected_steps=None,
                          rise_limits=None, fall_limits=None):
    """Validate one reservoir-major daily sequence without imposing year order.

    Held-out years may be nonconsecutive and even move backwards at a source-year
    join. Their source dates stay authoritative, while differences remain
    continuous in episode order across each join.
    """
    parsed = tuple(_calendar_date(day) for day in dates)
    if not parsed:
        raise ValueError('A daily diagnostic sequence must contain at least one day')
    if len(set(parsed)) != len(parsed):
        raise ValueError('Daily diagnostic dates must be unique')
    if expected_steps is not None and len(parsed) != _integer(expected_steps, 'expected_steps'):
        raise ValueError('Daily diagnostic sequence does not contain every expected evaluation day')
    levels = _matrix(levels, 'levels_end', len(parsed))
    releases = _matrix(releases, 'release', len(parsed))
    rise_limits = _optional_limits(rise_limits, 'rise_limit_m_per_day', len(parsed))
    fall_limits = _optional_limits(fall_limits, 'fall_limit_m_per_day', len(parsed))
    if (rise_limits is None) != (fall_limits is None):
        raise ValueError('Rise and fall ramp limits must be saved together')
    return parsed, levels, releases, rise_limits, fall_limits


def daily_changes(levels, releases):
    """Return level and release differences in episode order, with step 0 missing."""
    levels = _matrix(levels, 'levels_end')
    releases = _matrix(releases, 'release')
    if len(levels) != len(releases):
        raise ValueError('levels_end and release must contain the same number of days')
    level_change = np.full_like(levels, np.nan)
    release_change = np.full_like(releases, np.nan)
    if len(levels) > 1:
        level_change[1:] = np.diff(levels, axis=0)
        release_change[1:] = np.diff(releases, axis=0)
    return level_change, release_change


def level_change_statistics(levels, rise_limits=None, fall_limits=None):
    """Summarize finite absolute level changes independently for five reservoirs."""
    levels = _matrix(levels, 'levels_end')
    change, _ = daily_changes(levels, np.zeros_like(levels))
    magnitude = np.abs(change)
    rise_limits = _optional_limits(rise_limits, 'rise_limit_m_per_day', len(levels))
    fall_limits = _optional_limits(fall_limits, 'fall_limit_m_per_day', len(levels))
    if (rise_limits is None) != (fall_limits is None):
        raise ValueError('Rise and fall ramp limits must be supplied together')
    within = (ramp_compliance(change, rise_limits, fall_limits)
              if rise_limits is not None else None)
    rows = []
    for index, reservoir in enumerate(IDS):
        values = magnitude[1:, index]
        values = values[np.isfinite(values)]
        rows.append(dict(reservoir=reservoir, sample_count=int(len(values)),
            mean_abs_delta_level_m=(float(np.mean(values)) if len(values) else None),
            p95_abs_delta_level_m=(float(np.percentile(values, 95)) if len(values) else None),
            max_abs_delta_level_m=(float(np.max(values)) if len(values) else None),
            max_abs_allowed_level_change_m=(float(np.max(np.maximum(
                rise_limits[1:, index], fall_limits[1:, index])))
                if rise_limits is not None and len(levels) > 1 else None),
            ramp_violation_count=(int(np.sum(~within[1:, index]))
                if within is not None else None)))
    return rows


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


def write_daily_change_csv(path, trajectory):
    """Write one exclusive per-seed table; the first delta in each channel is empty."""
    path = Path(path)
    seed = _integer(trajectory.get('seed'), 'seed')
    dates, levels, releases, rise_limits, fall_limits = validate_daily_series(
        trajectory.get('dates', ()), trajectory.get('levels_end', ()),
        trajectory.get('release', ()), trajectory.get('steps'),
        trajectory.get('ramp_rise_limit_m_per_day'),
        trajectory.get('ramp_fall_limit_m_per_day'))
    level_change, release_change = daily_changes(levels, releases)
    within = (ramp_compliance(level_change, rise_limits, fall_limits)
              if rise_limits is not None else None)
    bounds = {name: _matrix(trajectory[name], name, len(dates))
              for name in LEVEL_BOUND_FIELDS if trajectory.get(name) is not None}
    if bounds and len(bounds) != len(LEVEL_BOUND_FIELDS):
        raise ValueError('Both permitted level bounds are required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=DAILY_CHANGE_FIELDS + tuple(bounds))
        writer.writeheader()
        for day_index, day in enumerate(dates):
            for reservoir_index, reservoir in enumerate(IDS):
                level_delta = level_change[day_index, reservoir_index]
                writer.writerow(dict(seed=seed, date=day.isoformat(), year=day.year,
                    day_of_year=day.timetuple().tm_yday, reservoir=reservoir,
                    level_m=_csv_value(levels[day_index, reservoir_index]),
                    delta_level_m=_csv_value(level_delta),
                    abs_delta_level_m=_csv_value(abs(level_delta) if np.isfinite(level_delta) else None),
                    release_m3s=_csv_value(releases[day_index, reservoir_index]),
                    delta_release_m3s=_csv_value(release_change[day_index, reservoir_index]),
                    rise_limit_m_per_day=(_csv_value(rise_limits[day_index, reservoir_index])
                        if rise_limits is not None else ''),
                    fall_limit_m_per_day=(_csv_value(fall_limits[day_index, reservoir_index])
                        if fall_limits is not None else ''),
                    within_ramp_limit=(_csv_value(int(within[day_index, reservoir_index]))
                        if within is not None and np.isfinite(level_delta) else ''),
                    **{name: _csv_value(values[day_index, reservoir_index])
                       for name, values in bounds.items()}))
    return path


def write_level_change_statistics_csv(path, seed, levels, rise_limits=None, fall_limits=None):
    """Write the same mean/P95/max values displayed by the statistics figure."""
    path = Path(path)
    seed = _integer(seed, 'seed')
    rows = level_change_statistics(levels, rise_limits, fall_limits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=STATISTICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(seed=seed, reservoir=row['reservoir'],
                sample_count=row['sample_count'],
                mean_abs_delta_level_m=_csv_value(row['mean_abs_delta_level_m']),
                p95_abs_delta_level_m=_csv_value(row['p95_abs_delta_level_m']),
                max_abs_delta_level_m=_csv_value(row['max_abs_delta_level_m']),
                max_abs_allowed_level_change_m=_csv_value(
                    row['max_abs_allowed_level_change_m']),
                ramp_violation_count=_csv_value(row['ramp_violation_count'])))
    return path


def daily_change_output_paths(directory, seeds):
    """Reserve two data CSVs and one conflict CSV per seed."""
    if directory is None:
        return {}
    directory = Path(directory)
    result = {}
    for seed in seeds:
        seed = _integer(seed, 'seed')
        stem = directory/('mappo_seed'+str(seed))
        paths = dict(csv=stem.with_name(stem.name+'_daily_changes.csv'),
                     statistics_csv=stem.with_name(stem.name+'_level_change_statistics.csv'),
                     ramp_conflicts_csv=stem.with_name(stem.name+'_ramp_conflicts.csv'))
        result[seed] = paths
    return result


def flatten_output_paths(paths):
    return [path for seed_paths in paths.values() for path in seed_paths.values()]


class DailyChangeRecorder:
    """Capture executed daily level and release vectors from one evaluation."""

    def __init__(self, seed):
        self.seed = _integer(seed, 'seed')
        self.dates, self.levels, self.releases = [], [], []
        self.rise_limits, self.fall_limits = [], []
        self.bounds = {name: [] for name in LEVEL_BOUND_FIELDS}

    def capture(self, transition):
        if not isinstance(transition, dict):
            raise ValueError('A daily transition must be a mapping')
        day = _calendar_date(transition.get('date'))
        if self.dates and day == self.dates[-1]:
            raise ValueError('A daily evaluation cannot repeat a calendar date')
        level = np.asarray(transition.get('levels_end'), dtype=np.float64)
        release = np.asarray(transition.get('release'), dtype=np.float64)
        if level.shape != (len(IDS),) or not np.isfinite(level).all():
            raise ValueError('levels_end must be a finite five-vector')
        if release.shape != (len(IDS),) or not np.isfinite(release).all():
            raise ValueError('release must be a finite five-vector')
        rise = transition.get('ramp_rise_limit_m_per_day')
        fall = transition.get('ramp_fall_limit_m_per_day')
        if (rise is None) != (fall is None):
            raise ValueError('A transition must provide both rise and fall ramp limits')
        if rise is not None:
            rise = np.asarray(rise, dtype=np.float64)
            fall = np.asarray(fall, dtype=np.float64)
            if (rise.shape != (len(IDS),) or fall.shape != (len(IDS),) or
                    not np.isfinite(rise).all() or not np.isfinite(fall).all() or
                    (rise < 0.).any() or (fall < 0.).any()):
                raise ValueError('Transition ramp limits must be nonnegative finite five-vectors')
        bounds = {name: transition.get(name) for name in LEVEL_BOUND_FIELDS}
        if any(value is not None for value in bounds.values()):
            bounds = {name: np.asarray(value, dtype=np.float64) for name, value in bounds.items()}
            if any(value.shape != (len(IDS),) or not np.isfinite(value).all()
                   for value in bounds.values()):
                raise ValueError('Permitted levels require two finite five-vectors')
            if (bounds[LEVEL_BOUND_FIELDS[0]] > bounds[LEVEL_BOUND_FIELDS[1]]).any():
                raise ValueError('Permitted level bounds must be ordered')
        for name, value in bounds.items():
            self.bounds[name].append(None if value is None else value.tolist())
        self.dates.append(day.isoformat())
        self.levels.append(level.tolist())
        self.releases.append(release.tolist())
        self.rise_limits.append(None if rise is None else rise.tolist())
        self.fall_limits.append(None if fall is None else fall.tolist())

    def finish(self, expected_steps=None):
        has_limits = bool(self.rise_limits) and self.rise_limits[0] is not None
        if any((value is not None) != has_limits for value in self.rise_limits):
            raise ValueError('Every evaluation transition must use the same ramp-limit availability')
        dates, levels, releases, rise_limits, fall_limits = validate_daily_series(
            self.dates, self.levels, self.releases, expected_steps,
            self.rise_limits if has_limits else None,
            self.fall_limits if has_limits else None)
        bounds = {name: values for name, values in self.bounds.items()
                  if any(value is not None for value in values)}
        if any(any(value is None for value in values) for values in bounds.values()):
            raise ValueError('Permitted level bounds must cover every executed day')
        return dict(schema_version=SCHEMA_VERSION, **bounds,
            record_type='evaluation_daily_change_series', seed=self.seed,
            steps=len(dates), dates=[day.isoformat() for day in dates],
            levels_end=levels.tolist(), release=releases.tolist(),
            ramp_rise_limit_m_per_day=(None if rise_limits is None else rise_limits.tolist()),
            ramp_fall_limit_m_per_day=(None if fall_limits is None else fall_limits.tolist()),
            level_timestamp='levels_end is the actual end-of-day level after each dated daily action',
            release_timestamp='release is the actual total daily release for each dated action',
            difference_rule='current minus previous in episode order; first daily difference is missing; '
                            'differences continue across source-year joins')
