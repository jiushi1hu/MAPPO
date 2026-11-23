"""Classify ecological-flow shortfalls on the executed held-out trajectory."""
import csv

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reward import ecological_flow_shortfall


FIELDS = (
    'seed', 'date', 'source_year', 'reservoir', 'inflow_m3s',
    'feasible_release_lower_m3s', 'feasible_release_upper_m3s',
    'ecological_threshold_m3s', 'executed_release_m3s', 'deficit_m3s',
    'deficit_volume_m3', 'requested_action_index', 'requested_normalized_action',
    'classification',
)
CLASSIFICATIONS = ('avoidable', 'unavoidable')


class EcologicalViolationRecorder:
    """Record reward-defined violations against the dispatch interval for that day."""

    def __init__(self, seed, requirements):
        self.seed = seed
        self.thresholds = None if requirements is None else requirements.ecological_flow_m3s
        self.rows = []

    def capture(self, transition):
        if self.thresholds is None:
            return
        deficits = transition['service_shortfalls']['ecological_flow']
        violations = np.flatnonzero(deficits > 0.)
        if not violations.size:
            return
        release = transition['release']
        forced = transition['forced_interval']
        # A collapsed interval executes its midpoint for every command.
        lower = np.where(forced, release, transition['low'])
        upper = np.where(forced, release, transition['high'])
        unavoidable = ecological_flow_shortfall(self.thresholds, upper) > 0.
        volumes = transition['service_shortfall_integrals']['ecological_flow']
        day = transition['date']
        for i in violations:
            self.rows.append(dict(seed=self.seed, date=day, source_year=int(day[:4]),
                reservoir=IDS[i], inflow_m3s=float(transition['inflow'][i]),
                feasible_release_lower_m3s=float(lower[i]), feasible_release_upper_m3s=float(upper[i]),
                ecological_threshold_m3s=float(self.thresholds[i]), executed_release_m3s=float(release[i]),
                deficit_m3s=float(deficits[i]), deficit_volume_m3=float(volumes[i]),
                requested_action_index=int(transition['sampled_actions'][i]),
                requested_normalized_action=float(transition['requested_normalized_actions'][i]),
                classification='unavoidable' if unavoidable[i] else 'avoidable'))

    def print_summary(self, active_days):
        print('Held-out ecological-flow violations for seed {} (rates per active day)'.format(self.seed),
              flush=True)
        for reservoir, days in zip(IDS, active_days):
            rows = [row for row in self.rows if row['reservoir'] == reservoir]
            groups = [('total', rows)] + [(kind, [row for row in rows if row['classification'] == kind])
                                         for kind in CLASSIFICATIONS]
            cells = []
            for kind, group in groups:
                count = len(group)
                rate = '{:.2%}'.format(count/days) if days else 'n/a'
                volume = sum(row['deficit_volume_m3'] for row in group)
                cells.append('{} {} ({}), {:,.3f} m3'.format(kind, count, rate, volume))
            print('  {} ({} active days) | {}'.format(reservoir, days, ' | '.join(cells)), flush=True)


def write_ecological_violations(path, recorders):
    """Write one CSV across seeds, including a header for zero violations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for recorder in recorders:
            writer.writerows(recorder.rows)
