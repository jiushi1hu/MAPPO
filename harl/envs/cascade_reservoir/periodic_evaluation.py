"""A fixed training-only sequence for observational MAPPO policy evaluation."""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import positive_integer, training_windows
from harl.envs.cascade_reservoir.evaluation import (
    STANDARDIZED_EVALUATION_FIELDS, evaluate_policy, episode_constraint_status)
from harl.envs.cascade_reservoir.storage_accounting import storage_change_summary
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import REWARD_AGGREGATION, validate_discount_factor


PERIODIC_EVALUATION_DAYS = 1825
_EVALUATION_METRICS = (
    'evaluator_return', 'system_evaluator_return', 'global_evaluator_return',
    'mixed_evaluator_returns', 'local_evaluator_returns', 'generation_MWh',
    'generation_TWh', 'ecological_flow_penalty', 'final_storage_m3',
    'initial_storage_m3', 'total_storage_depletion_m3',
) + STANDARDIZED_EVALUATION_FIELDS
_VIOLATION_TERMS = ('ecological_flow',)
_VIOLATION_METRICS = tuple(term+suffix for term in _VIOLATION_TERMS
    for suffix in ('_violation_rate', '_violation_count', '_active_reservoir_days'))


def _violation_metrics(shortfalls):
    """Report service frequency from undiscounted active reservoir-day counts.

    Relative shortfall magnitudes and weighted reward penalties cannot recover
    these frequencies. A disabled requirement contributes no eligible days;
    a term with no eligible reservoir-days has an undefined (null) rate.
    """
    result = {}
    for term in _VIOLATION_TERMS:
        summary = shortfalls[term]
        count, eligible = summary['reservoir_day_count'], summary['active_reservoir_days']
        if any(isinstance(value, (bool, np.bool_)) or
               not isinstance(value, (int, np.integer)) or value < 0
               for value in (count, eligible)) or count > eligible:
            raise ValueError('Service violation counts must satisfy 0 <= count <= active reservoir-days')
        result.update({term+'_violation_count':int(count),
                       term+'_active_reservoir_days':int(eligible),
                       term+'_violation_rate':float(count/eligible) if eligible else None})
    return result


def _window_descriptor(data, start_year, years, episode_days, label):
    start, _ = data.indices(start_year, start_year)
    stop = start+episode_days
    external = np.asarray(data.external[start:stop], dtype=np.float64)
    initial = data.initial_storage(start_year)
    return dict(window_id=label+'_'+str(start_year), label=label,
        window_start_year=int(start_year), source_years=[int(year) for year in years],
        episode_horizon_days=episode_days, start_date=data.dates[start].isoformat(),
        end_date=data.dates[stop-1].isoformat(), initial_storage_m3=initial.tolist(),
        initial_level_m=data.levels(initial).tolist(),
        mean_total_external_inflow_m3s=float(external.sum(axis=1).mean()),
        external_inflow_sha256=hashlib.sha256(external.tobytes()).hexdigest())


def select_periodic_window(data, episode_days=PERIODIC_EVALUATION_DAYS):
    """Select the median-inflow training sequence once for all seeds and rounds."""
    days = positive_integer(episode_days, 'periodic_evaluation_days')
    candidates = training_windows(data, days)
    ranked = []
    for year, years in candidates.items():
        start, _ = data.indices(year, year)
        mean = float(data.external[start:start+days].sum(axis=1).mean())
        ranked.append((mean, int(year), years))
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, year, years = ranked[len(ranked)//2]
    return _window_descriptor(data, year, years, days, 'median')


def _validate_window(data, window):
    if not isinstance(window, dict):
        raise ValueError('Periodic evaluation requires one fixed window descriptor')
    days = positive_integer(window['episode_horizon_days'], 'periodic_evaluation_days')
    start = positive_integer(window['window_start_year'], 'window_start_year')
    years = training_windows(data, days).get(start)
    if years is None or years != window['source_years'] or set(years) & set(data.evaluation_years):
        raise ValueError('Periodic evaluation requires one complete consecutive training-only sequence')
    if _window_descriptor(data, start, years, days, window['label']) != window:
        raise ValueError('Periodic evaluation dates, inflows or initial conditions changed')


@contextmanager
def _preserve_training_state(actors):
    """Evaluation consumes no training RNG draws and restores every module mode."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for actor in actors for module in actor.actor.modules()]
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_periodic_policy(actors, data, physical_config, *, seed, evaluation_round,
                             training_steps, window, gamma):
    """Evaluate the current policy in a fresh environment without training it."""
    if (isinstance(evaluation_round, (bool, np.bool_)) or
            not isinstance(evaluation_round, (int, np.integer)) or evaluation_round < 0):
        raise ValueError('evaluation_round must be a nonnegative integer')
    if (isinstance(training_steps, (bool, np.bool_)) or
            not isinstance(training_steps, (int, np.integer)) or training_steps < 0):
        raise ValueError('training_steps must be a nonnegative integer')
    gamma = validate_discount_factor(gamma)
    _validate_window(data, window)
    metrics = _EVALUATION_METRICS + _VIOLATION_METRICS
    result = dict(window, seed=int(seed), evaluation_round=int(evaluation_round),
                  status='ok', failure=None)
    try:
        with _preserve_training_state(actors):
            measured = evaluate_policy(actors, data, physical_config, seed=int(seed),
                episode=int(evaluation_round), training_steps=int(training_steps),
                gamma=gamma, source_years=window['source_years'],
                episode_days=window['episode_horizon_days'], deterministic=True)
    except InfeasibleCascade as exc:
        progress = getattr(exc, 'evaluation_progress', {})
        result.update({metric: None for metric in metrics})
        result.update(status='failed', failure=dict(error_type=type(exc).__name__,
            reason=str(exc), date=progress.get('date', window['start_date']),
            episode_step=progress.get('episode_step', 1),
            completed_steps=progress.get('completed_steps', 0),
            conflicts=progress.get('conflicts', [])), service_shortfalls=None,
            initial_storage_m3=window['initial_storage_m3'],
            partial_metrics=progress.get('partial_metrics', {}),
            partial_storage_m3=progress.get('partial_storage_m3', window['initial_storage_m3']))
        result.update(storage_change_summary(window['initial_storage_m3'],
            result['partial_storage_m3'], prefix='partial_'))
    else:
        result.update({metric: measured[metric] for metric in STANDARDIZED_EVALUATION_FIELDS})
        result.update(evaluator_return=measured['episode_reward'],
            system_evaluator_return=measured['episode_system_reward'],
            global_evaluator_return=measured['episode_global_reward'],
            mixed_evaluator_returns=measured['episode_mixed_rewards'],
            local_evaluator_returns=measured['episode_local_rewards'],
            generation_MWh=measured['energy_MWh'], generation_TWh=measured['energy_TWh'],
            ecological_flow_penalty=measured['episode_penalty_ecological_flow'],
            final_storage_m3=measured['final_storage_m3'],
            total_storage_depletion_m3=measured['total_storage_depletion_m3'],
            completed_steps=window['episode_horizon_days'],
            violation_count=measured['violation_count'],
            max_daily_bound_excess_m3=measured['max_daily_bound_excess_m3'],
            service_shortfalls=measured['service_shortfalls'], units=measured['units'])
        result.update(_violation_metrics(measured['service_shortfalls']))

    completed = result['status'] == 'ok'
    constraint_status = episode_constraint_status(completed=completed, physical_feasible=completed,
        service_shortfalls=result['service_shortfalls'])
    result.update(constraint_status)
    record = dict(record_type='periodic_evaluation', phase='training_evaluation',
        seed=int(seed), evaluation_round=int(evaluation_round), training_steps=int(training_steps),
        status=result['status'], window_count=1, successful_window_count=int(completed),
        completed_window_count=int(completed), gamma=gamma, deterministic=True,
        reservoir_ids=list(IDS), evaluator_return_aggregation=REWARD_AGGREGATION,
        scenario_aggregation='one fixed evaluation sequence',
        violation_rate_definition='violating active reservoir-days / all active reservoir-days; '
            'undiscounted and independent of reward weights and shortfall magnitudes',
        violation_rate_units='fraction',
        standardized_metrics=list(STANDARDIZED_EVALUATION_FIELDS),
        standardized_metric_role='policy evaluation only',
        windows=[result], **constraint_status)
    record.update({metric: result[metric] if completed else None for metric in metrics})
    return record
