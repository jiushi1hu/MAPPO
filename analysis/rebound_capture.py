"""Capture a historical seed-1 prefix without changing the training entry point."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.checkpoint import save_checkpoint
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'runs/mappo_seed123_scale5_advnone_20261126_142022'


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def history():
    with (HISTORY / 'training.jsonl').open() as stream:
        return [json.loads(line) for line in stream]


def make_runner(records, cls=CascadeMAPPORunner):
    config = records[0]
    args, phys = config['config'], config['physical_config']
    data = CascadeData(ROOT / 'data/cascade_reservoir_new', phys, args['experiment'])
    assert data.hashes == config['data_hashes']
    for name, digest in config['source_code_hashes'].items():
        if name.endswith('mappo_cascade.yaml'):
            continue  # Effective algorithm settings come directly from the historical log.
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    runner = cls(data, phys, args, 1)
    runner.run_id = config['run_id']
    return runner


class PrefixComplete(Exception):
    pass


class CaptureRunner(CascadeMAPPORunner):
    def train(self):
        update = self.update_count + 1
        assert self.batch_window_starts == self.expected_batches[update]
        if 40 <= update <= 46:
            cb = self.critic_buffer
            advantages = cb.returns[:-1] - self.value_normalizer.denormalize(cb.value_preds[:-1])
            torch.save(dict(actors=self.actor_buffer, critic=cb, advantages=advantages,
                            windows=self.batch_window_starts, rng=rng_state()),
                       self.output / f'batch_{update}.pt')
        return super().train()

    def periodic_evaluation(self):
        row = super().periodic_evaluation()
        update = self.update_count
        expected = self.expected_eval[update]
        keys = ['ecological_flow_violation_count', 'ecological_deficit_volume_m3',
                'generation_TWh', 'system_return', 'net_storage_change_m3']
        for key in keys:
            np.testing.assert_allclose(row[key], expected[key], rtol=1e-10, atol=1e-6,
                                       err_msg=f'update {update} {key}')
        with (self.output / 'prefix.jsonl').open('a') as stream:
            print(json.dumps(dict(update=update, **{key: row[key] for key in keys})), file=stream)
        if 40 <= update <= 46:
            save_checkpoint(self, self.output / f'policy_{update}.pt')
            torch.save(dict(actor=self.actor[0].actor_optimizer.state_dict(),
                            critic=self.critic.critic_optimizer.state_dict(),
                            rng=rng_state(), sampler=self.training_sampler,
                            steps=self.training_steps, episodes=self.completed_episodes),
                       self.output / f'optim_{update}.pt')
        if update == 46:
            raise PrefixComplete()
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    out = parser.parse_args().output
    records = history()
    runner = make_runner(records, CaptureRunner)
    out.mkdir(parents=True, exist_ok=False)
    runner.output = out
    runner.expected_eval = {x['evaluation_round']: x for x in records
                            if x['record_type'] == 'periodic_evaluation' and x['seed'] == 1}
    runner.expected_batches = {x['update']: x['window_start_years'] for x in records
                               if x['record_type'] == 'training_update' and x['seed'] == 1}
    try:
        runner.run(300, 1)
    except PrefixComplete:
        pass


if __name__ == '__main__':
    main()
