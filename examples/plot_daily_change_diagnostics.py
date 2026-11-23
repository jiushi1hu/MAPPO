"""Render executed reservoir trajectories from held-out daily CSVs."""
import argparse
import csv
from datetime import date
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

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
    """Plot one column per recorded seed, with consistent scales by reservoir."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    if not series_by_label:
        raise ValueError('No recorded held-out trajectories to plot')
    plot_fields = ('levels', 'releases', *LEVEL_BOUND_FIELDS)
    by_label = {}
    seen_seeds = set()
    for label, series in series_by_label.items():
        dates = series['dates']
        if not len(dates) or series['seed'] in seen_seeds:
            raise ValueError('Trajectory plots require nonempty data for distinct seeds')
        seen_seeds.add(series['seed'])
        if any(name not in series for name in LEVEL_BOUND_FIELDS):
            raise ValueError('Trajectory plots require both permitted level bounds')
        values = {name: np.asarray(series[name], dtype=float) for name in plot_fields}
        if any(matrix.shape != (len(dates), len(IDS)) or not np.isfinite(matrix).all()
               for matrix in values.values()):
            raise ValueError('Trajectory data must contain one finite value per day and reservoir')
        by_label[label] = dict(values, dates=dates)

    levels = np.concatenate([series[name] for series in by_label.values()
                             for name in ('levels', *LEVEL_BOUND_FIELDS)])
    releases = np.concatenate([series['releases'] for series in by_label.values()])
    level_min, level_max = levels.min(axis=0), levels.max(axis=0)
    release_min, release_max = releases.min(axis=0), releases.max(axis=0)
    level_pad = np.maximum((level_max-level_min)*.12, .25)
    release_pad = np.maximum((release_max-release_min)*.08, 1.)

    level_color, bound_color, release_color = '#255eb0', '#637995', '#84bb9b'
    flood_color, boundary_color = '#eeeef0', '#737373'
    columns = len(by_label)
    width = max(10., 9.*columns+1.)
    figure, axes = plt.subplots(len(IDS), columns, sharex='col', squeeze=False, figsize=(width, 14))
    figure.subplots_adjust(left=.86/width, right=1-.87/width, bottom=.080,
                           top=.900, wspace=.27, hspace=.18)
    figure.suptitle('Held-out operation trajectories', y=.981)
    figure.legend(handles=[
        Line2D([], [], color=level_color, label='Water level'),
        Line2D([], [], color=bound_color, ls='--', lw=.8, label='Permitted bounds'),
        Line2D([], [], color=release_color, label='Executed release'),
        Patch(facecolor=flood_color, edgecolor='none', label='June–September'),
        Line2D([], [], color=level_color, marker='o', ls='none', label='Source-year end'),
    ], loc='upper center', bbox_to_anchor=(.5, .954), ncol=5, frameon=False, fontsize=9)

    for column, (label, series) in enumerate(by_label.items()):
        dates = series['dates']
        days = np.arange(1, len(dates)+1)
        years = np.asarray([day.year for day in dates])
        starts = np.r_[0, np.flatnonzero(np.diff(years))+1]
        ends = np.r_[starts[1:], len(dates)]-1
        flood_days = [6 <= day.month <= 9 for day in dates]
        flood_edges = np.flatnonzero(np.diff(flood_days, prepend=False, append=False))+.5
        lower, upper = (series[name] for name in LEVEL_BOUND_FIELDS)

        for index, reservoir in enumerate(IDS):
            level_ax = axes[index, column]
            release_ax = level_ax.twinx()
            release_ax.set_zorder(level_ax.get_zorder()-1)
            level_ax.patch.set_visible(False)
            for start, end in zip(flood_edges[::2], flood_edges[1::2]):
                release_ax.axvspan(start, end, color=flood_color, lw=0, zorder=0)
            release_ax.plot(days, series['releases'][:, index], color=release_color, lw=.85)
            level_ax.plot(days, lower[:, index], '--', color=bound_color, lw=.65, zorder=2)
            level_ax.plot(days, upper[:, index], '--', color=bound_color, lw=.65, zorder=2)
            daily_levels = series['levels'][:, index]
            level_ax.plot(days, daily_levels, color=level_color, lw=1.0, zorder=3)
            for start in starts[1:]:
                level_ax.axvline(days[start]-.5, ls='--', color=boundary_color, lw=.75, zorder=1)
            level_ax.plot(days[ends], daily_levels[ends], 'o', color=level_color, ms=3, zorder=4)
            for end in ends:
                value = daily_levels[end]
                end_day = dates[end]
                value_label = '{:.2f}'.format(value)
                if (end_day.month, end_day.day) != (12, 31):
                    value_label += end_day.strftime(' (%b %d)')
                near_top = value > level_min[index]+.85*(level_max[index]-level_min[index])
                level_ax.annotate(value_label, (days[end], value),
                    xytext=(-4, -6 if near_top else 6), textcoords='offset points',
                    ha='right', va='top' if near_top else 'bottom', fontsize=7, color=level_color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1), zorder=5)

            level_ax.set_ylim(level_min[index]-level_pad[index], level_max[index]+level_pad[index])
            release_ax.set_ylim(max(0., release_min[index]-release_pad[index]),
                                release_max[index]+release_pad[index])
            level_ax.set_ylabel((reservoir+'\n' if column == 0 else '')+'Water level (m)')
            release_ax.set_ylabel('Release (m³/s)')
            for ax, color in ((level_ax, level_color), (release_ax, '#40795d')):
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.tick_params(axis='y', labelsize=8, colors=color)
                ax.yaxis.label.set_color(color)
            level_ax.grid(axis='y', alpha=.16)

        axes[0, column].set_title(label, fontsize=11, pad=8)
        bottom = axes[-1, column]
        month_ticks = [i for i, day in enumerate(dates)
                       if day.day == 1 and day.month in (1, 4, 7, 10)]
        bottom.set_xlim(.5, len(days)+.5)
        bottom.set_xticks((days[starts]+days[ends])/2)
        bottom.set_xticklabels(years[starts], fontsize=9)
        bottom.set_xticks(days[month_ticks], minor=True)
        bottom.set_xticklabels([dates[i].strftime('%b') for i in month_ticks], minor=True, fontsize=7)
        bottom.tick_params(axis='x', which='major', length=0, pad=15)
        bottom.tick_params(axis='x', which='minor', length=2, pad=2)
    return figure


def overwrite_operation_figure(figure, path):
    """Render completely before replacing the existing trajectory PNG."""
    if path.suffix.lower() != '.png' or not path.is_file():
        raise ValueError('No existing trajectory figure at '+str(path))
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            figure.savefig(stream, format='png', dpi=160)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Replace the existing trajectory PNG')
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    try:
        series = [load_daily_change_csv(path) for path in args.input]
        by_label = {'Seed '+str(row['seed']): row for row in sorted(series, key=lambda row: row['seed'])}
        if len(by_label) != len(series):
            raise ValueError('Specify only one daily CSV per seed')
        figure = make_operation_figure(by_label)
        try:
            if args.overwrite:
                overwrite_operation_figure(figure, args.output)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('xb') as stream:
                    figure.savefig(stream, format='png', dpi=160)
        finally:
            plt.close(figure)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
