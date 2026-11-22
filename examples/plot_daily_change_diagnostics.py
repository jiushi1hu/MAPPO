"""Render executed reservoir trajectories from held-out daily CSVs."""
import argparse
import csv
from datetime import date
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.daily_change_diagnostics import (  # noqa: E402
    DAILY_CHANGE_FIELDS, IDS, LEVEL_BOUND_FIELDS)


def _number(value, field, allow_missing=False):
    if value in ('', None):
        if allow_missing:
            return None
        raise ValueError(field+' is missing')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(field+' must be numeric') from exc
    if not math.isfinite(result):
        raise ValueError(field+' must be finite')
    return result


def _flag(value, field, allow_missing=False):
    if value in ('', None):
        if allow_missing:
            return None
        raise ValueError(field+' is missing')
    if value not in ('0', '1'):
        raise ValueError(field+' must be 0 or 1')
    return value == '1'


def load_daily_change_csv(path):
    """Load one seed table and require a complete five-reservoir daily layout."""
    path = Path(path)
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if fields not in (DAILY_CHANGE_FIELDS, DAILY_CHANGE_FIELDS + LEVEL_BOUND_FIELDS):
            raise ValueError('Unexpected daily-change CSV columns')
        rows = list(reader)
    if not rows or len(rows) % len(IDS):
        raise ValueError('Daily-change CSV must contain complete five-reservoir days')
    seed = _number(rows[0]['seed'], 'seed')
    if not seed.is_integer() or any(_number(row['seed'], 'seed') != seed for row in rows):
        raise ValueError('Daily-change CSV must contain one integer seed')
    days = len(rows)//len(IDS)
    dates, years, day_of_year = [], np.empty(days, dtype=int), np.empty(days, dtype=int)
    levels = np.empty((days, len(IDS)), dtype=float)
    bounds = {name: np.empty_like(levels) for name in LEVEL_BOUND_FIELDS if name in fields}
    deltas = np.full((days, len(IDS)), np.nan, dtype=float)
    releases = np.empty((days, len(IDS)), dtype=float)
    release_deltas = np.full((days, len(IDS)), np.nan, dtype=float)
    rise_limits = np.full((days, len(IDS)), np.nan, dtype=float)
    fall_limits = np.full((days, len(IDS)), np.nan, dtype=float)
    within_limits = np.full((days, len(IDS)), np.nan, dtype=float)
    for day_index in range(days):
        expected_day = None
        for reservoir_index, reservoir in enumerate(IDS):
            row = rows[day_index*len(IDS)+reservoir_index]
            if row['reservoir'] != reservoir:
                raise ValueError('Daily-change CSV must use WDD/BHT/XLD/XJB/THR order')
            day = date.fromisoformat(row['date'])
            if expected_day is None:
                expected_day = day
                dates.append(day)
                years[day_index] = int(row['year'])
                day_of_year[day_index] = int(row['day_of_year'])
                if years[day_index] != day.year or day_of_year[day_index] != day.timetuple().tm_yday:
                    raise ValueError('Daily-change CSV year or day_of_year disagrees with its date')
            elif day != expected_day:
                raise ValueError('Every reservoir must use the same actual date on a given day')
            levels[day_index, reservoir_index] = _number(row['level_m'], 'level_m')
            releases[day_index, reservoir_index] = _number(row['release_m3s'], 'release_m3s')
            for name, values in bounds.items():
                values[day_index, reservoir_index] = _number(row[name], name)
            delta = _number(row['delta_level_m'], 'delta_level_m', allow_missing=True)
            release_delta = _number(row['delta_release_m3s'], 'delta_release_m3s', allow_missing=True)
            rise = _number(row['rise_limit_m_per_day'], 'rise_limit_m_per_day',
                           allow_missing=True)
            fall = _number(row['fall_limit_m_per_day'], 'fall_limit_m_per_day',
                           allow_missing=True)
            if (rise is None) != (fall is None):
                raise ValueError('Rise and fall ramp limits must be present together')
            if rise is not None:
                if rise < 0. or fall < 0.:
                    raise ValueError('Ramp limits must be nonnegative')
                rise_limits[day_index, reservoir_index] = rise
                fall_limits[day_index, reservoir_index] = fall
            inside = _flag(row['within_ramp_limit'], 'within_ramp_limit',
                           allow_missing=True)
            if day_index == 0:
                if delta is not None or release_delta is not None:
                    raise ValueError('The first daily difference must be missing')
            elif delta is None or release_delta is None:
                raise ValueError('Only the first daily difference may be missing')
            else:
                deltas[day_index, reservoir_index] = delta
                release_deltas[day_index, reservoir_index] = release_delta
                if rise is not None:
                    expected_inside = (delta <= rise+1e-8 and delta >= -fall-1e-8)
                    if inside is None or inside != expected_inside:
                        raise ValueError('within_ramp_limit disagrees with delta and ramp limits')
                    within_limits[day_index, reservoir_index] = int(inside)
                elif inside is not None:
                    raise ValueError('within_ramp_limit requires saved ramp limits')
            if day_index == 0 and inside is not None:
                raise ValueError('The first daily ramp-limit classification must be missing')
            magnitude = _number(row['abs_delta_level_m'], 'abs_delta_level_m', allow_missing=True)
            if delta is None:
                if magnitude is not None:
                    raise ValueError('Missing level change requires missing absolute level change')
            elif magnitude is None or not np.isclose(magnitude, abs(delta), rtol=1e-12, atol=1e-12):
                raise ValueError('Absolute level change must equal abs(delta_level_m)')
    if len(set(dates)) != len(dates):
        raise ValueError('Daily-change dates must be unique')
    if bounds and (bounds[LEVEL_BOUND_FIELDS[0]] > bounds[LEVEL_BOUND_FIELDS[1]]).any():
        raise ValueError('Permitted level bounds must be ordered')
    return dict(**bounds, seed=int(seed), dates=dates, years=years, day_of_year=day_of_year,
                levels=levels, deltas=deltas, releases=releases,
                release_deltas=release_deltas, rise_limits=rise_limits,
                fall_limits=fall_limits, within_limits=within_limits,
                has_ramp_limits=bool(np.isfinite(rise_limits).any()))


def make_operation_figure(series_by_label):
    """Plot levels, effective permitted bounds and release in episode order."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(IDS), 2*len(series_by_label), sharex='col',
                                squeeze=False, figsize=(7*len(series_by_label), 12),
                                constrained_layout=True)
    for column, (label, series) in enumerate(series_by_label.items()):
        if any(name not in series for name in LEVEL_BOUND_FIELDS):
            raise ValueError('Trajectory plots require both permitted level bounds')
        lower, upper = (np.asarray(series[name]) for name in LEVEL_BOUND_FIELDS)
        days = np.arange(1, len(series['dates'])+1)
        starts = np.r_[0, np.flatnonzero(np.diff(series['years']))+1]
        for index, reservoir in enumerate(IDS):
            level_ax, release_ax = axes[index, 2*column:2*column+2]
            level_ax.fill_between(days, lower[:, index], upper[:, index],
                                  color='#bfdbfe', label='Permitted bounds')
            level_ax.plot(days, series['levels'][:, index], color='#1e3a8a', lw=0.85,
                          label='End-of-day level')
            release_ax.plot(days, series['releases'][:, index], color='#047857', lw=0.85)
            level_ax.set_ylabel(reservoir+' level (m)')
            release_ax.set_ylabel(reservoir+' release (m³/s)')
            for ax in (level_ax, release_ax):
                for start in starts[1:]:
                    ax.axvline(days[start], color='0.7', lw=0.7)
                ax.grid(alpha=0.2)
        axes[0, 2*column].set_title(label+' · Water level')
        axes[0, 2*column+1].set_title(label+' · Executed release')
        axes[0, 2*column].legend(fontsize=8)
        for ax in axes[-1, 2*column:2*column+2]:
            ax.set_xticks(days[starts], series['years'][starts])
            ax.set_xlabel('Source year in episode order')
    figure.suptitle('Held-out operation trajectories')
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    try:
        series = load_daily_change_csv(args.input)
        figure = make_operation_figure({'Seed '+str(series['seed']): series})
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as stream:
                figure.savefig(stream, format='png', dpi=160)
        finally:
            plt.close(figure)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
