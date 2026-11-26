"""Report forced commands and realized release authority from saved training CSVs.

Run with python -m analysis.training_control_authority RUN_DIRECTORY.
Each visit to a source date counts separately. No environment or policy is run.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import islice
from operator import itemgetter
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.dispatch_kernel import FLOW_TOLERANCE


COLUMNS = ('seed', 'episode', 'episode_step', 'date', 'reservoir_id',
           'feasible_release_low_m3s', 'feasible_release_high_m3s', 'forced_action',
           'inflow_m3s', 'storage_start_m3', 'storage_lower_m3', 'storage_upper_m3',
           'ecological_threshold_m3s', 'min_ecological_compliant_action_index')


def binding_labels(bound, storage_bound, outlet_bound, storage_label, outlet_label):
    labels = []
    if abs(bound - storage_bound) <= FLOW_TOLERANCE:
        labels.append(storage_label)
    if abs(bound - outlet_bound) <= FLOW_TOLERANCE:
        labels.append(outlet_label)
    return '+'.join(labels) or 'downstream compatibility'


def read_trace(path, seed, cap, dt, horizon, episodes):
    widths = []
    downstream = np.zeros(len(IDS), dtype=np.int64)
    forced_counts = np.zeros(len(IDS), dtype=np.int64)
    ecology = np.zeros((3, len(IDS)), dtype=np.int64)
    forced_by_episode = Counter()
    seen = set()
    with path.open(newline='') as source:
        reader = csv.reader(source)
        header = next(reader)
        select = itemgetter(*(header.index(k) for k in COLUMNS))
        while True:
            rows = list(map(select, islice(reader, 25000)))
            if not rows:
                break
            if len(rows) % len(IDS):
                raise ValueError(f'Incomplete reservoir group in {path}')
            fields = list(zip(*rows))
            seed_col, episode, step, date, reservoir = (
                np.asarray(col).reshape(-1, len(IDS)) for col in fields[:5])
            if (np.any(seed_col != str(seed)) or np.any(reservoir != IDS)
                    or any(np.any(col != col[:, :1]) for col in (episode, step, date))):
                raise ValueError(f'Inconsistent reservoir group in {path}')
            keys = list(zip(episode[:, 0].astype(int), step[:, 0].astype(int)))
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError(f'Duplicate episode day in {path}')
            seen.update(keys)
            low, high, forced, inflow, storage, storage_low, storage_high = (
                np.asarray(col, dtype=float).reshape(-1, len(IDS)) for col in fields[5:12])
            threshold, minimum_bin = (np.asarray(col).reshape(-1, len(IDS)) for col in fields[12:])
            active = np.where(threshold == '', '0', threshold).astype(float) > 0.
            first = np.where(minimum_bin == '', '-1', minimum_bin).astype(int)
            ecology += np.stack([(active & mask).sum(axis=0)
                                 for mask in (first == 0, first > 0, first < 0)])
            width = high - low
            if (not all(np.isfinite(v).all() for v in (width, inflow, storage, storage_low, storage_high))
                    or np.any(width < -FLOW_TOLERANCE)
                    or np.any(forced != (width <= FLOW_TOLERANCE))):
                raise ValueError(f'Invalid release interval or forced flag in {path}')
            # Compare the recorded endpoints to each local mass-balance bound.
            # Effective storage limits already include the day's ramp/level rules.
            from_max_storage = inflow + (storage - storage_high) / dt
            from_min_storage = inflow + (storage - storage_low) / dt
            local_low = np.maximum(0., from_max_storage)
            local_high = np.minimum(cap, from_min_storage)
            if np.any(low < local_low - FLOW_TOLERANCE) or np.any(high > local_high + FLOW_TOLERANCE):
                raise ValueError(f'Recorded interval outside local bounds in {path}')
            downstream += ((low > local_low + FLOW_TOLERANCE) |
                           (high < local_high - FLOW_TOLERANCE)).sum(axis=0)
            widths.append(np.maximum(width, 0.))
            forced_counts += forced.sum(axis=0).astype(int)
            for day, i in np.argwhere(forced):
                ep = int(episode[day, i])
                forced_by_episode[ep] += 1
                lo = binding_labels(low[day, i], from_max_storage[day, i], 0.,
                                    'effective maximum storage', 'zero release')
                hi = binding_labels(high[day, i], from_min_storage[day, i], cap[i],
                                    'effective minimum storage', 'outlet capacity')
                print(f'Forced: seed={seed} episode={ep} step={step[day,i]} '
                      f'date={date[day,i]} reservoir={IDS[i]} interval='
                      f'[{low[day,i]:.12g}, {high[day,i]:.12g}] m3/s '
                      f'lower={lo} upper={hi}')
    if (len(seen) != horizon * len(episodes)
            or any(ep not in episodes or not 1 <= step <= horizon for ep, step in seen)
            or any(forced_by_episode[ep] != count for ep, count in episodes.items())):
        raise ValueError(f'Trace coverage or forced counts disagree with episode log: {path}')
    return np.concatenate(widths), forced_counts, downstream, ecology


def report(label, widths, counts, downstream, ecology, cap):
    for i, name in enumerate(IDS):
        w = widths[:, i]
        pct = 100 * w / cap[i]
        quantiles = np.percentile(w, [0, 5, 50, 95, 100])
        cells = [label, name, len(w), int(counts[i]), f'{100*counts[i]/len(w):.6f}']
        cells += [f'{q:.2f}' for q in quantiles]
        cells += [f'{np.median(pct):.3f}', *(f'{100*np.mean(pct < k):.3f}' for k in (1, 5, 10)),
                  f'{100*downstream[i]/len(w):.3f}']
        eligible = ecology[:, i].sum()
        cells += [f'{100*n/eligible:.3f}' if eligible else 'n/a' for n in ecology[:, i]]
        print(*cells, sep='\t')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/cascade_reservoir_new')
    args = parser.parse_args()
    root = args.run_directory
    episodes = {}
    with (root / 'training.jsonl').open() as log:
        config = json.loads(next(log))
        last_type = config['record_type']
        for line in log:
            row = json.loads(line)
            last_type = row['record_type']
            if last_type == 'training_episode':
                episodes.setdefault(row['seed'], {})[row['episode']] = row['forced_actor_days']
    if last_type != 'run_complete' or not episodes:
        raise ValueError('Expected a completed training run with episode records')
    parameters = args.data_dir / 'reservoir_parameters.csv'
    if hashlib.sha256(parameters.read_bytes()).hexdigest() != config['data_hashes'][parameters.name]:
        raise ValueError('Release-capacity data differs from the recorded run')
    with parameters.open(newline='', encoding='utf-8-sig') as source:
        capacities = {r['reservoir_id']: float(r['max_total_release_m3s']) for r in csv.DictReader(source)}
    cap = np.asarray([capacities[name] for name in IDS])
    files = config['trajectory_diagnostics']['files']
    print('Widths are upper minus lower in m3/s. Relative widths use total outlet capacity. '
          'Downstream-tightened means a recorded endpoint is stricter than local storage and outlet bounds.')
    print('seed reservoir visits forced forced_% min p05 p50 p95 max median_%cap '
          'below_1%cap_% below_5%cap_% below_10%cap_% downstream_tightened_% '
          'ecology_all_bins_% ecology_some_bins_% ecology_no_bin_%')
    all_widths, counts, downstream = [], np.zeros(5, dtype=int), np.zeros(5, dtype=int)
    ecology = np.zeros((3, len(IDS)), dtype=np.int64)
    for seed, expected in sorted(episodes.items()):
        path = root / Path(files[str(seed)]).name
        w, n, d, eco = read_trace(path, seed, cap, config['physical_config']['timestep_seconds'],
                             config['episode_horizon_days'], expected)
        report(seed, w, n, d, eco, cap)
        all_widths.append(w)
        counts += n
        downstream += d
        ecology += eco
    report('all', np.concatenate(all_widths), counts, downstream, ecology, cap)
    if not counts.any():
        print('No forced training visits. There are no forced dates or binding causes to attribute.')


if __name__ == '__main__':
    main()
