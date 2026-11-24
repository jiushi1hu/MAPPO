"""Observational physical/learning traces for one complete training batch.

Only the current batch's selected transition columns are retained. Learning
quantities come exclusively from ``diagnostics.before_update``: writing after
optimization must not reinterpret old values with new ValueNorm moments.
"""
import csv
from datetime import date
import json
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.dispatch_kernel import release_from_normalized_action
from harl.envs.cascade_reservoir.learning_signal import critic_reward_spec
from harl.envs.cascade_reservoir.reward import ecological_flow_shortfall


_PHYSICAL_COLUMNS = {
    'storage_start_m3': 'storage_start', 'storage_end_m3': 'storage_end',
    'level_start_m': 'levels_start', 'level_end_m': 'levels_end',
    'storage_lower_m3': 'storage_lower_m3', 'storage_upper_m3': 'storage_upper_m3',
    'next_state_level_lower_m': 'next_state_level_lower_m',
    'next_state_level_upper_m': 'next_state_level_upper_m',
    'external_inflow_m3s': 'external', 'inflow_m3s': 'inflow',
    'release_m3s': 'release', 'turbine_m3s': 'turbine', 'spill_m3s': 'spill',
    'net_head_m': 'net_head_m', 'power_mw': 'power', 'generation_MWh': 'energy',
    'feasible_release_low_m3s': 'low', 'feasible_release_high_m3s': 'high',
    'mass_balance_residual_m3': 'residual', 'normalized_action': 'normalized_actions',
    'local_generation_reward': 'local_generation_rewards',
    'local_reward_penalty': 'local_reward_penalties',
    'local_reward': 'local_rewards', 'mixed_reward': 'mixed_rewards',
}
_SERVICE_COLUMNS = {
    'ecological_flow_shortfall_m3s': ('service_shortfalls', 'ecological_flow'),
    'ecological_flow_deficit_m3': ('service_shortfall_integrals', 'ecological_flow'),
}
_SCALAR_COLUMNS = ('global_reward', 'system_reward', 'reward', 'generation_reward')
ECOLOGICAL_ACTION_DIAGNOSTIC_FIELDS = (
    'argmax_action_index', 'argmax_action_probability', 'ecological_threshold_m3s',
    'min_ecological_compliant_action_index', 'min_ecological_compliant_action_probability',
    'compliant_action_probability_mass',
)
TRAJECTORY_DIAGNOSTIC_FIELDS = (
    'seed', 'update', 'episode', 'parallel_env_index', 'window_start_year',
    'source_years', 'episode_step', 'date', 'reservoir_id', 'value_channel',
    'collection_policy_update', 'probability_source', 'action_selection',
    *_PHYSICAL_COLUMNS,
    'local_ecological_flow_penalty',
    *_SERVICE_COLUMNS, 'global_reward', 'system_reward', 'legacy_team_reward',
    'legacy_generation_reward', 'sampled_action_index', 'effective_action_index',
    'action_bins', 'forced_action', 'sampled_action_probability',
    'sampled_action_log_probability', 'effective_action_probability',
    'effective_action_log_probability', 'entropy', *ECOLOGICAL_ACTION_DIAGNOSTIC_FIELDS,
    'learning_reward',
    'learning_reward_source', 'value_basis', 'critic_reward_layout', 'own_reward_for_learning',
    'own_value_frozen', 'own_next_value_frozen', 'next_state_mask', 'td_residual',
    'gae_target', 'raw_advantage', 'normalized_advantage', 'realized_reward_to_go',
    'episode_time_weight', 'policy_objective_weight', 'actor_advantage_estimator',
)
_OPTIONAL_COLUMNS = set(_SERVICE_COLUMNS) | set(ECOLOGICAL_ACTION_DIAGNOSTIC_FIELDS[2:]) | {
    'effective_action_probability', 'effective_action_log_probability'}
_TEXT_COLUMNS = {'date', 'reservoir_id', 'source_years',
                 'learning_reward_source', 'value_basis', 'critic_reward_layout', 'actor_advantage_estimator',
                 'probability_source', 'action_selection'}
_SUMMARY_FIELDS = ('local_reward', 'mixed_reward', 'learning_reward', 'own_value_frozen', 'td_residual',
                   'raw_advantage', 'normalized_advantage', 'realized_reward_to_go')


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _array(value, shape, name):
    result = np.asarray(value)
    real_numeric = (np.issubdtype(result.dtype, np.integer) or
                    np.issubdtype(result.dtype, np.floating))
    if result.shape != shape or not real_numeric or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite numeric array with shape '+str(shape))
    return result


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value):
        raise ValueError(name+' must be a finite scalar')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be a finite scalar') from exc
    if not np.isfinite(number):
        raise ValueError(name+' must be a finite scalar')
    return number


def ecological_action_diagnostics(probabilities, lower, upper, threshold, *, forced_release=None):
    """Measure a policy against realized same-day bounds, without changing actions.

    Compliance uses the reward's tolerance and includes zero-probability commands
    in the physical minimum. A forced interval executes the same release for every
    index. These are conditional diagnostics, never an actor action mask.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 1 or probabilities.size < 2 or
            not np.isfinite(probabilities).all() or (probabilities < 0.).any() or
            not np.isclose(probabilities.sum(), 1., rtol=1e-5, atol=1e-7)):
        raise ValueError('Action probabilities must be a normalized finite categorical vector')
    argmax = int(probabilities.argmax())
    result = dict(argmax_action_index=argmax, argmax_action_probability=float(probabilities[argmax]),
        ecological_threshold_m3s=None, min_ecological_compliant_action_index=None,
        min_ecological_compliant_action_probability=None, compliant_action_probability_mass=None)
    if threshold is None:
        return result
    threshold = _number(threshold, 'ecological threshold')
    if threshold < 0.:
        raise ValueError('Ecological threshold must be nonnegative')
    if forced_release is None:
        lower, upper = _number(lower, 'release lower bound'), _number(upper, 'release upper bound')
        if lower > upper:
            raise ValueError('Diagnostic release bounds must be ordered')
        releases = release_from_normalized_action(lower, upper,
            np.arange(probabilities.size, dtype=np.float64)/(probabilities.size-1))
    else:
        releases = np.full(probabilities.size, _number(forced_release, 'forced release'))
    compliant = ecological_flow_shortfall(threshold, releases) == 0.
    indices = np.flatnonzero(compliant)
    minimum = int(indices[0]) if indices.size else None
    result.update(ecological_threshold_m3s=threshold,
        min_ecological_compliant_action_index=minimum,
        min_ecological_compliant_action_probability=None if minimum is None else float(probabilities[minimum]),
        compliant_action_probability_mass=float(probabilities[compliant].sum()))
    return result


class _Summary:
    """Constant-memory aggregates; rows are never retained a second time."""

    def __init__(self):
        self.count = self.forced = self.spilling = self.positive_spill_advantage = 0
        self.generation = self.spill = self.weight = 0.
        self.sums = dict.fromkeys(_SUMMARY_FIELDS, 0.)
        self.spill_advantage = 0.

    def update(self, row, dt):
        self.count += 1
        self.forced += int(row['forced_action'])
        spilling = row['spill_m3s'] > 0.
        self.spilling += int(spilling)
        self.positive_spill_advantage += int(spilling and row['raw_advantage'] > 0.)
        self.spill_advantage += row['raw_advantage'] if spilling else 0.
        self.generation += row['generation_MWh']
        self.spill += row['spill_m3s']*dt
        self.weight += row['policy_objective_weight']
        for name in self.sums:
            self.sums[name] += row[name]

    def finish(self):
        return dict(rows=self.count, forced_rows=self.forced,
            controllable_rows=self.count-self.forced, spill_days=self.spilling,
            generation_MWh=self.generation, spill_volume_m3=self.spill,
            policy_objective_weight_sum=self.weight,
            mean={key:value/self.count if self.count else None for key,value in self.sums.items()},
            spilling_days_mean_raw_advantage=self.spill_advantage/self.spilling if self.spilling else None,
            spilling_days_positive_raw_advantage_fraction=(
                self.positive_spill_advantage/self.spilling if self.spilling else None))


class TrajectoryDiagnosticsWriter:
    """Exclusive CSV owner, retaining at most one batch of copied physical data.

    The sampled probability belongs to the frozen collection distribution.
    Effective probabilities on forced rows are left blank, excluding synthetic
    ``exp(0)=1`` buffer placeholders. Actual sampled-command likelihoods and
    policy entropy remain available; their objective weight is zero.
    Each actor row has uniform temporal weight and its reservoir's mixed-return
    value and advantage.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.writer = None
        self._batch = None
        self._transitions = []
        self.rows_written = 0
        self._last_update = None

    def __enter__(self):
        if self.stream is not None:
            raise RuntimeError('Trajectory writer is already open')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8', newline='')
        self.writer = csv.writer(self.stream, lineterminator='\n')
        self.writer.writerow(TRAJECTORY_DIAGNOSTIC_FIELDS)
        self.stream.flush()
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()
        self.stream = self.writer = None
        self._batch = None
        self._transitions.clear()

    def _require_open(self):
        if self.stream is None or self.stream.closed:
            raise RuntimeError('Trajectory writer must be used in its context manager')

    def begin_batch(self, *, seed, update, episode_ids, window_starts, source_years):
        self._require_open()
        if self._batch is not None:
            raise RuntimeError('Write the current trajectory batch before beginning another')
        seed, update = _integer(seed, 'seed'), _integer(update, 'update', 1)
        ids = [_integer(value, 'episode', 1) for value in episode_ids]
        starts = [_integer(value, 'window_start_year', 1) for value in window_starts]
        years = [[_integer(year, 'source year', 1) for year in group] for group in source_years]
        if (not ids or len(ids) != len(set(ids)) or len(ids) != len(starts) or
                len(ids) != len(years) or any(not group or len(set(group)) != len(group) for group in years)):
            raise ValueError('Batch metadata requires one unique episode and nonempty year sequence per lane')
        if any(group[0] != start for start, group in zip(starts, years)):
            raise ValueError('Source-year sequences must begin at their window start')
        if self._last_update is not None and update <= self._last_update:
            raise ValueError('Trajectory update numbers must increase')
        self._batch = dict(seed=seed, update=update, episode_ids=ids,
                           window_starts=starts, source_years=years)
        self._transitions = []

    def capture(self, t, transitions):
        """Copy selected columns before worker results or caller arrays can change."""
        self._require_open()
        if self._batch is None:
            raise RuntimeError('begin_batch must precede capture')
        if _integer(t, 'step') != len(self._transitions):
            raise ValueError('Capture steps must be consecutive and zero-based')
        if len(transitions) != len(self._batch['episode_ids']):
            raise ValueError('Capture requires one transition per batch lane')
        captured = []
        for lane, transition in enumerate(transitions):
            day = date.fromisoformat(transition['date'])
            if day.year not in self._batch['source_years'][lane]:
                raise ValueError('Transition date is outside its source-year sequence')
            if self._transitions:
                previous = date.fromisoformat(self._transitions[-1][lane]['date'])
                if (day-previous).days != 1:
                    raise ValueError('Training captures require consecutive source dates')
            row = dict(date=day.isoformat(),
                       action_bins=_integer(transition['action_bins'], 'action_bins', 2),
                       timestep_seconds=_number(transition['timestep_seconds'], 'timestep_seconds'))
            if row['timestep_seconds'] <= 0.:
                raise ValueError('timestep_seconds must be positive')
            for name in (*_PHYSICAL_COLUMNS.values(), 'controls', 'sampled_actions', 'actions'):
                row[name] = _array(transition[name], (len(IDS),), name).copy()
            if not np.isin(row['controls'], (0., 1.)).all():
                raise ValueError('controls must be binary')
            for name in ('sampled_actions', 'actions'):
                values = row[name]
                if ((values != np.floor(values)) | (values < 0.) | (values >= row['action_bins'])).any():
                    raise ValueError(name+' must contain valid discrete command indices')
            for column, (group, term) in _SERVICE_COLUMNS.items():
                value = transition.get(group, {}).get(term)
                row[column] = None if value is None else _array(value, (len(IDS),), column).copy()
            for term in ('ecological_flow',):
                row['local_'+term+'_penalty'] = _array(
                    transition['local_reward_penalty_components'][term], (len(IDS),), term).copy()
            row.update({key:_number(transition.get(key, transition['reward'])
                                    if key == 'generation_reward' else transition[key], key)
                        for key in _SCALAR_COLUMNS})
            captured.append(row)
        self._transitions.append(captured)

    def _learning_arrays(self, runner, before):
        cb = runner.critic_buffer
        shape = cb.rewards.shape
        days, lanes, channels = shape
        if (channels != len(IDS) or days != len(self._transitions) or
                lanes != len(self._batch['episode_ids']) or len(runner.actor_buffer) != len(IDS) or
                len(runner.actor) != len(IDS)):
            raise ValueError('Trajectory captures must match a complete five-reservoir rollout')
        spec = critic_reward_spec(runner, channels)
        rewards = _array(cb.rewards, shape, 'learning rewards')
        frozen = {key:_array(before[key], shape, key) for key in
                  ('prediction', 'next_prediction', 'gae_targets', 'mc')}
        masks = _array(cb.masks, (days+1, lanes, 1), 'masks')
        if (not np.isin(masks, (0., 1.)).all() or not np.all(masks[-1] == 0.) or
                not np.all(masks[1:-1] == 1.) or not np.all(cb.bad_masks == 1.)):
            raise ValueError('Trajectory diagnostics require complete terminal episodes without truncation')
        gamma = _number(cb.gamma, 'gamma')
        if not 0. <= gamma <= 1.:
            raise ValueError('gamma must be between zero and one')
        raw = frozen['gae_targets']-frozen['prediction']
        shared = runner.algo_args['algo'].get('normalize_shared_advantages', False)
        common = ((raw-raw.mean(axis=(0, 1), keepdims=True))/
                  (raw.std(axis=(0, 1), keepdims=True)+1e-5) if shared else None)
        if len(before['distributions']) != len(IDS):
            raise ValueError('One frozen collection distribution per reservoir is required')
        actors = []
        for i, (buffer, actor, distribution) in enumerate(zip(
                runner.actor_buffer, runner.actor, before['distributions'])):
            active = _array(buffer.active_masks[:-1], (days, lanes, 1), 'active masks')
            if not np.isin(active, (0., 1.)).all() or not np.array_equal(buffer.masks, masks):
                raise ValueError('Actor masks must match complete rollout boundaries')
            channel = spec['actor_channels'][i]
            advantage = raw[..., channel:channel+1]
            if shared:
                normalized = common[..., channel:channel+1]
            elif active.any():
                selected = advantage[active > 0.]
                normalized = (advantage-selected.mean())/(selected.std()+1e-5)
            else:
                normalized = np.zeros_like(advantage)
            # Keep the existing CSV column, with uniform temporal weight for
            # every row. Gamma is used below only in the frozen TD residual.
            time = np.ones((days, lanes, 1), dtype=np.float32)
            if not actor.use_policy_active_masks:
                raise ValueError('Forced trajectory commands require policy active masks')
            weights = time*active
            actions = _array(buffer.actions, (days, lanes, 1), 'effective actions')
            logs = _array(buffer.action_log_probs, (days, lanes, 1), 'effective action logs')
            probabilities = distribution.probs.detach().cpu().numpy().reshape(days, lanes, -1)
            logits = distribution.logits.detach().cpu().numpy().reshape(days, lanes, -1)
            entropy = distribution.entropy().detach().cpu().numpy().reshape(days, lanes)
            _array(probabilities, probabilities.shape, 'frozen action probabilities')
            _array(logits, probabilities.shape, 'frozen action logs')
            _array(entropy, (days, lanes), 'frozen action entropy')
            actors.append(dict(channel=channel, raw=advantage, normalized=normalized,
                advantage_estimator='gae' if cb.use_gae else 'discounted_return',
                learning_reward_source=spec['actor_reward_sources'][i],
                value_basis=spec['actor_value_bases'][i],
                critic_reward_layout=spec['layout'],
                time=time, weights=weights, actions=actions, logs=logs,
                probabilities=probabilities, logits=logits, entropy=entropy, active=active))
        return rewards, frozen, masks, gamma, actors

    def write_batch(self, runner, before):
        """Write frozen training evidence without model calls or state updates."""
        self._require_open()
        if self._batch is None or not self._transitions or before is None:
            raise ValueError('A captured batch and frozen before-update diagnostics are required')
        collection_update = _integer(before['collection_policy_update'], 'collection policy update')
        if collection_update != self._batch['update']-1:
            raise ValueError('Collection policy must precede the trajectory optimization update')
        rewards, frozen, masks, gamma, actors = self._learning_arrays(runner, before)
        requirements = getattr(runner.data, 'service_requirements', None)
        thresholds = (None if requirements is None else
                      _array(requirements.ecological_flow_m3s, (len(IDS),), 'ecological thresholds'))
        spec = critic_reward_spec(runner, rewards.shape[-1])
        captured = np.asarray([[row['mixed_rewards'] for row in transitions]
                               for transitions in self._transitions], dtype=np.float32)
        if not np.array_equal(rewards.astype(np.float32), captured):
            raise ValueError('Learning rewards disagree with the captured mixed rewards')
        windows = {name:{reservoir:_Summary() for reservoir in IDS}
                   for name in ('all_days', 'first_14_days', 'october_may')}
        batch = self._batch
        count = 0
        for t, transitions in enumerate(self._transitions):
            for lane, transition in enumerate(transitions):
                day = date.fromisoformat(transition['date'])
                for i, reservoir in enumerate(IDS):
                    actor = actors[i]
                    channel, bins = actor['channel'], transition['action_bins']
                    sampled, effective = int(transition['sampled_actions'][i]), int(transition['actions'][i])
                    if actor['probabilities'].shape[-1] != bins:
                        raise ValueError('Captured and frozen-distribution action grids disagree')
                    forced = not bool(transition['controls'][i])
                    if (actor['actions'][t,lane,0] != effective or
                            actor['active'][t,lane,0] != transition['controls'][i] or
                            (forced and (effective != 0 or actor['logs'][t,lane,0] != 0.))):
                        raise ValueError('Captured commands/controls disagree with effective rollout placeholders')
                    row = dict(seed=batch['seed'], update=batch['update'], episode=batch['episode_ids'][lane],
                        collection_policy_update=collection_update,
                        probability_source='collection_before_update', action_selection='sample',
                        parallel_env_index=lane, window_start_year=batch['window_starts'][lane],
                        source_years=';'.join(map(str, batch['source_years'][lane])),
                        episode_step=t+1, date=day.isoformat(), reservoir_id=reservoir, value_channel=channel)
                    row.update({name:float(transition[source][i]) for name,source in _PHYSICAL_COLUMNS.items()})
                    row.update({name:None if transition[name] is None else float(transition[name][i])
                                for name in _SERVICE_COLUMNS})
                    for term in ('ecological_flow',):
                        key = 'local_'+term+'_penalty'
                        row[key] = float(transition[key][i])
                    row.update(global_reward=transition['global_reward'], system_reward=transition['system_reward'],
                        legacy_team_reward=transition['reward'], legacy_generation_reward=transition['generation_reward'],
                        sampled_action_index=sampled, effective_action_index=effective, action_bins=bins,
                        forced_action=int(forced), sampled_action_probability=float(actor['probabilities'][t,lane,sampled]),
                        sampled_action_log_probability=float(actor['logits'][t,lane,sampled]),
                        effective_action_log_probability=None if forced else float(actor['logs'][t,lane,0]),
                        effective_action_probability=None if forced else float(np.exp(actor['logs'][t,lane,0])),
                        entropy=float(actor['entropy'][t,lane]), learning_reward=float(rewards[t,lane,channel]),
                        learning_reward_source=actor['learning_reward_source'], value_basis=actor['value_basis'],
                        critic_reward_layout=actor['critic_reward_layout'],
                        own_reward_for_learning=float(rewards[t,lane,channel]),
                        own_value_frozen=float(frozen['prediction'][t,lane,channel]),
                        own_next_value_frozen=float(frozen['next_prediction'][t,lane,channel]),
                        next_state_mask=float(masks[t+1,lane,0]),
                        td_residual=float(rewards[t,lane,channel]+gamma*frozen['next_prediction'][t,lane,channel]*
                                          masks[t+1,lane,0]-frozen['prediction'][t,lane,channel]),
                        gae_target=float(frozen['gae_targets'][t,lane,channel]),
                        raw_advantage=float(actor['raw'][t,lane,0]),
                        normalized_advantage=float(actor['normalized'][t,lane,0]),
                        actor_advantage_estimator=actor['advantage_estimator'],
                        realized_reward_to_go=float(frozen['mc'][t,lane,channel]),
                        episode_time_weight=float(actor['time'][t,lane,0]),
                        policy_objective_weight=float(actor['weights'][t,lane,0]))
                    row.update(ecological_action_diagnostics(actor['probabilities'][t,lane],
                        transition['low'][i], transition['high'][i],
                        None if thresholds is None else thresholds[i],
                        forced_release=transition['release'][i] if forced else None))
                    cells = []
                    for key in TRAJECTORY_DIAGNOSTIC_FIELDS:
                        value = row[key]
                        if value is None and key in _OPTIONAL_COLUMNS:
                            value = ''
                        elif key not in _TEXT_COLUMNS:
                            _number(value, key)
                        cells.append(value)
                    self.writer.writerow(cells)
                    count += 1
                    windows['all_days'][reservoir].update(row, transition['timestep_seconds'])
                    if t < 14:
                        windows['first_14_days'][reservoir].update(row, transition['timestep_seconds'])
                    if day.month >= 10 or day.month <= 5:
                        windows['october_may'][reservoir].update(row, transition['timestep_seconds'])
        self.stream.flush()
        self.rows_written += count
        summary = dict(schema='cascade_trajectory_v4', file=str(self.path.resolve()), rows=count,
            collection_policy_update=collection_update, optimization_update=batch['update'],
            total_rows_written=self.rows_written, episode_horizon_days=len(self._transitions),
            parallel_environments=len(batch['episode_ids']), gamma=gamma,
            critic_reward_layout=spec['layout'], critic_fitted_reward_basis=spec['fitted_reward_basis'],
            actor_advantage_estimator='gae' if runner.critic_buffer.use_gae else 'discounted_return',
            actor_value_channels=list(spec['actor_channels']),
            legacy_team_value_available=spec['legacy_value_available'],
            actor_temporal_weighting='uniform_rollout_rows',
            actor_objective_reduction='mean over all rollout rows; policy active masks zero forced rows',
            episode_time_weight='always 1.0; column retained for CSV compatibility',
            policy_objective_weight='policy active mask only',
            value_source='before_update snapshot in reward units before optimization and ValueNorm changes',
            own_column_interpretation='reservoir-specific mixed reward and value',
            probability_source='collection_before_update; forced effective probability/log-probability are blank; '
                               'sampled probabilities and entropy are real policy measurements',
            ecological_action_scope='realized same-day bounds conditional on current upstream releases; '
                                    'reward tolerance; blank minimum if unattainable; blank ecological fields if unavailable',
            repeated_accounting_scalars=['global_reward', 'system_reward', 'legacy_team_reward', 'legacy_generation_reward'],
            interpretation='observational correlated training rows; realized reward-to-go is not expected value; '
                           'spill and advantage association does not establish avoidability or causation',
            windows={name:{reservoir:value.finish() for reservoir,value in group.items()}
                     for name,group in windows.items()})
        # Finite daily inputs can still overflow accumulated summary totals.
        json.dumps(summary, allow_nan=False)
        self._last_update = batch['update']
        self._batch = None
        self._transitions.clear()
        return summary
