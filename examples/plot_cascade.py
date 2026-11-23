"""Render learning, performance, operation trajectories and water allocation."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples import plot_cascade_parallel as legacy


COLOR = '#2563eb'
# Serialized layout versions are kept here so plotting does not import Gym,
# PyTorch or the simulator in a separate Matplotlib-only interpreter.
ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_next7d_forecast_v6'
VOLUME_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_terminal_target_next7d_forecast_future_volume_v5'
TARGET_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_terminal_target_next7d_forecast_v4'
PREVIOUS_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_next7d_forecast_v3'
CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_next7d_forecasts_v3'
VOLUME_CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_forecasts_targets_future_volumes_v2'
TARGET_CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_next7d_forecasts_terminal_targets_v1'
CASCADE_OBSERVATION_LAYOUTS = {
    'cascade_local_upstream_downstream_system_v7': ('concatenated_five_actor_observations_v1', 22, 110),
    'cascade_local_upstream_downstream_system_v8': ('concatenated_five_actor_observations_v2', 21, 105),
}
FORECAST_INPUT = 'five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv'
SHARED_FIELDS = (
    'episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
    'gamma', 'action_bins', 'seeds', 'training_years', 'evaluation_source_years',
    'window_sampling', 'physical_config', 'periodic_windows', 'evaluation_interval',
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def context_from_configuration(metadata):
    """Read the run settings and data provenance used for plotting."""
    if 'comparison_context' in metadata:
        return metadata['comparison_context']
    config = metadata['config']
    experiment = config['experiment']
    shared = dict(
        episode_horizon_days=metadata['episode_horizon_days'],
        parallel_environments=metadata['parallel_environments'],
        episodes_per_seed=metadata['episodes_per_seed'],
        gamma=config['algo']['gamma'], action_bins=config['model']['action_bins'],
        seeds=experiment['seeds'], training_years=metadata['training_years'],
        evaluation_source_years=metadata['evaluation_source_years'],
        window_sampling=experiment.get('window_sampling', 'coverage_balanced'),
        physical_config=metadata['physical_config'],
        periodic_windows=metadata['periodic_evaluation']['windows'],
        evaluation_interval=experiment.get('eval_interval', 1),
    )
    architecture = metadata.get('learning_reward_architecture', {})
    if 'actor_temporal_weighting' in architecture:
        shared['actor_temporal_weighting'] = architecture['actor_temporal_weighting']
    for key in ('device', 'train', 'model', 'algo'):
        if key in config:
            shared[key] = config[key]
    return dict(shared_settings=shared, data_hashes=metadata["data_hashes"])


def _context(payload):
    context = payload.get('comparison_context')
    if not isinstance(context, dict):
        raise ValueError('Every algorithm payload requires comparison_context')
    if (payload.get('run_id') is not None and
            (not isinstance(payload.get('provenance'), dict) or
             payload['provenance'].get('run_id') != payload['run_id'])):
        raise ValueError('Comparison payload run_id disagrees with its provenance')
    settings, hashes = context.get('shared_settings'), context.get('data_hashes')
    if not isinstance(settings, dict) or any(key not in settings for key in SHARED_FIELDS):
        raise ValueError('comparison_context.shared_settings requires: '+', '.join(SHARED_FIELDS))
    if (not isinstance(hashes, dict) or not hashes or
            any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in hashes.items())):
        raise ValueError('comparison_context requires nonempty input data_hashes')
    _canonical(context)  # Reject NaN and non-JSON values before drawing.
    for key in ('episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
                'action_bins', 'evaluation_interval'):
        legacy.positive_integer(settings[key], key)
    if settings['action_bins'] < 2:
        raise ValueError('action_bins must be at least two')
    if settings['episodes_per_seed'] % settings['parallel_environments']:
        raise ValueError('episodes_per_seed must be divisible by parallel_environments')
    gamma = settings['gamma']
    if isinstance(gamma, bool) or not np.isfinite(float(gamma)) or not 0 <= float(gamma) <= 1:
        raise ValueError('Comparison gamma must be finite and in [0, 1]')
    for key in ('seeds', 'training_years', 'evaluation_source_years'):
        values = settings[key]
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, int) for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(key+' must contain unique integers')
    if set(settings['training_years']) & set(settings['evaluation_source_years']):
        raise ValueError('Training and held-out source years must be disjoint')
    if settings['window_sampling'] not in ('coverage_balanced', 'uniform'):
        raise ValueError('Unknown comparison training-window sampling mode')
    if not isinstance(settings['physical_config'], dict) or not settings['physical_config']:
        raise ValueError('Comparison physical_config must be a nonempty mapping')
    expected_signature = legacy.window_signature(settings['periodic_windows'])
    for window in settings['periodic_windows']:
        if not set(window['source_years']) <= set(settings['training_years']):
            raise ValueError('Periodic comparison windows must contain training source years only')
    if float(payload['gamma']) != float(gamma):
        raise ValueError('Plot gamma disagrees with comparison_context')
    if payload['episode_horizon_days'] != settings['episode_horizon_days']:
        raise ValueError('Plot episode horizon disagrees with comparison_context')
    if ('parallel_envs' in payload and
            payload['parallel_envs'] != settings['parallel_environments']):
        raise ValueError('Plot worker count disagrees with comparison_context')
    updates = settings['episodes_per_seed']//settings['parallel_environments']
    if 'updates_per_seed' in payload and payload['updates_per_seed'] != updates:
        raise ValueError('Plot update budget disagrees with comparison_context')
    seeds = payload['seeds']
    if (not isinstance(seeds, list) or not seeds or
            any(isinstance(v, bool) or not isinstance(v, int) for v in seeds) or
            len(set(seeds)) != len(seeds) or not set(seeds) <= set(settings['seeds'])):
        raise ValueError('Plot seeds must be a nonempty subset of configured comparison seeds')
    if payload.get('configured_seeds', settings['seeds']) != settings['seeds']:
        raise ValueError('Configured plot seeds disagree with comparison_context')
    return expected_signature

def _observation_dimensions(run):
    """Validate explicit layouts, retaining the dimensions of unversioned logs."""
    architecture = run.get('actor_architecture') or {}
    daily_limits = run['comparison_context']['shared_settings']['physical_config'].get(
        'use_daily_level_limits', True)
    has_forecast = FORECAST_INPUT in run['comparison_context']['data_hashes']
    actor_layout = architecture.get('actor_observation_layout')
    critic_layout = architecture.get('critic_state_layout')
    state_size = 38 if daily_limits else 18
    raw_dim = 15 if daily_limits else 11
    if actor_layout in CASCADE_OBSERVATION_LAYOUTS:
        expected_critic, raw_dim, state_size = CASCADE_OBSERVATION_LAYOUTS[actor_layout]
        if not has_forecast:
            raise ValueError('The cascade actor observation layout requires the inflow forecast input hash')
        if critic_layout != expected_critic:
            raise ValueError('The cascade actor observation layout requires the matching critic state layout')
        if architecture.get('critic_state_size') != state_size:
            raise ValueError('Critic state size does not match its observation layout')
        return raw_dim, state_size
    if actor_layout in (ACTOR_OBSERVATION_LAYOUT, TARGET_ACTOR_OBSERVATION_LAYOUT, VOLUME_ACTOR_OBSERVATION_LAYOUT):
        if not has_forecast:
            raise ValueError('The current actor observation layout requires the inflow forecast input hash')
        extra, expected_critic = {ACTOR_OBSERVATION_LAYOUT: (1, CRITIC_STATE_LAYOUT),
            TARGET_ACTOR_OBSERVATION_LAYOUT: (2, TARGET_CRITIC_STATE_LAYOUT),
            VOLUME_ACTOR_OBSERVATION_LAYOUT: (3, VOLUME_CRITIC_STATE_LAYOUT)}[actor_layout]
        raw_dim += extra
        state_size += 5*extra
        if critic_layout != expected_critic:
            raise ValueError('Current actor observation layout requires the matching critic state layout')
        if architecture.get('critic_state_size') != state_size:
            raise ValueError('Critic state size does not match its observation layout')
    else:
        if actor_layout not in (None, PREVIOUS_ACTOR_OBSERVATION_LAYOUT):
            raise ValueError('Unsupported actor observation layout: '+str(actor_layout))
        if actor_layout == PREVIOUS_ACTOR_OBSERVATION_LAYOUT and not has_forecast:
            raise ValueError('The forecast actor observation layout requires the inflow forecast input hash')
        raw_dim += int(has_forecast)
        if critic_layout is not None:
            raise ValueError('Unsupported critic state layout for historical actor observations')
        if 'critic_state_size' in architecture and architecture['critic_state_size'] != state_size:
            raise ValueError('Historical critic state size does not match the cascade observations')
    return raw_dim, state_size


def _validate_architecture(run):
    """Check current and historical MAPPO observation and value layouts."""
    settings = run["comparison_context"]["shared_settings"]
    architecture = run.get('actor_architecture')
    objective = run.get('learning_objective')
    critic_dim = 5
    if objective is not None:
        if not isinstance(objective, dict):
            raise ValueError('learning_objective must be a mapping')
        layout = objective.get('critic_reward_layout')
        legacy_layout = layout == 'mixed_by_reservoir_then_legacy_team'
        if layout not in ('mixed_by_reservoir', 'mixed_by_reservoir_then_legacy_team'):
            raise ValueError('Unsupported MAPPO learning reward layout')
        normalization = ('per_reservoir_over_common_rollout_rows'
            if settings.get('algo', {}).get('normalize_shared_advantages', not legacy_layout)
            else 'per_reservoir_over_controllable_rows')
        expected_objective = dict(advantage_source='individual_mixed',
            reward_formula='0.5 * own local reward + 0.5 * global generation reward',
            critic_reward_layout=layout, actor_value_channels=list(range(5)),
            normalization=normalization)
        if objective != expected_objective:
            raise ValueError('Learning objective does not match the algorithm reward/value layout')
        if objective.get('critic_reward_layout') == 'mixed_by_reservoir_then_legacy_team':
            critic_dim = 6  # Historical runs used five mixed values plus a legacy head.
        else:
            critic_dim = 5
    if architecture is None:  # Historical MAPPO used five independent actors.
        return
    if not isinstance(architecture, dict):
        raise ValueError('actor_architecture must be a mapping')
    shared = architecture.get('parameter_sharing')
    if not isinstance(shared, bool):
        raise ValueError('Actor parameter sharing must be a boolean')
    raw_dim, _ = _observation_dimensions(run)
    identity = architecture.get('identity_features')
    legacy_identity = shared and identity == 'one_hot_reservoir_id'
    if shared and identity not in ('none', 'one_hot_reservoir_id'):
        raise ValueError('Unsupported shared MAPPO actor identity features')
    expected = dict(network_count=1 if shared else 5, optimizer_count=1 if shared else 5,
        raw_observation_dim=raw_dim,
        network_observation_dim=raw_dim+5 if legacy_identity else raw_dim,
        identity_features=identity if shared else 'implicit_in_separate_actor',
        critic_output_dim=critic_dim)
    if legacy_identity and objective is not None and objective.get('critic_reward_layout') != 'mixed_by_reservoir_then_legacy_team':
        raise ValueError('Agent-ID actor inputs are only supported for historical six-head runs')
    if any(architecture.get(key) != value for key, value in expected.items()):
        raise ValueError('Actor architecture dimensions/counts do not match the cascade observations')


def validate_run(run):
    """Check one MAPPO run and its training and evaluation histories."""
    if not isinstance(run, dict) or str(run.get('algorithm', '')).lower() != 'mappo':
        raise ValueError('Plot input requires an explicitly labeled MAPPO run')
    signature = _context(run)
    _validate_architecture(run)
    legacy.available_histories(run, 'histories')
    for rows in run.get('histories', []):
        steps = [legacy.positive_integer(row['training_steps'], 'training_steps') for row in rows]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError('Training steps must increase without duplicates')
        if any(not np.isfinite(float(row[key])) for row in rows
               for key in ('episode_reward', 'episode_energy_TWh')):
            raise ValueError('Training curve metrics must be finite')
        for row in rows:
            legacy.learning_return_value(run, row, 'training')
    _, histories, _ = legacy.periodic_histories(run)
    for rows in histories:
        for row in rows:
            legacy.validate_periodic_row(row, signature)
            if 'gamma' in row and float(row['gamma']) != float(run['gamma']):
                raise ValueError('Periodic gamma differs from plot gamma')
    if not legacy.violation_metrics_available(run):
        raise ValueError('Figures require actual periodic violation counts')


def load_run(path):
    run = legacy.load_episode_log(path)
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                metadata = json.loads(line)
                if metadata.get('record_type') == 'run_configuration':
                    break
        else:
            raise ValueError('A run_configuration is required for plotting')
    run.update(algorithm=metadata.get('algorithm', metadata.get('algorithm_name')),
               comparison_context=context_from_configuration(metadata),
               log_directory=str(Path(path).resolve().parent))
    return run


FIGURE_CATEGORIES = ('learning', 'performance', 'trajectories', 'water_allocation')
IDS = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')


def _make_learning(run):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _, histories, grids = legacy.periodic_histories(run)
    grid = np.unique(np.concatenate(grids))
    aligned = np.full((len(histories), len(grid)), np.nan)
    for index, (rounds, rows) in enumerate(zip(grids, histories)):
        aligned[index, np.searchsorted(grid, rounds)] = [
            legacy._finite_return(row['system_evaluator_return'], 'system_evaluator_return')
            if row['status'] == 'ok' else np.nan for row in rows]
    configured = run['comparison_context']['shared_settings']['seeds']
    valid = np.isfinite(aligned).all(axis=0) & (len(histories) == len(configured))
    mean = np.full(len(grid), np.nan)
    mean[valid] = aligned[:, valid].mean(axis=0)
    color = COLOR
    ax.plot(grid, mean, color=color, lw=2, marker='.',
            label='MAPPO mean')
    if len(histories) > 1:
        sd = np.full(len(grid), np.nan)
        sd[valid] = aligned[:, valid].std(axis=0, ddof=1)
        ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.18,
                        label='MAPPO ±1 SD')
    if (~valid).any():
        ax.scatter(grid[~valid], np.full((~valid).sum(), .07),
                   transform=ax.get_xaxis_transform(), color=color, marker='x',
                   label='MAPPO incomplete round')
    ax.set(xlabel='Evaluation round', ylabel='Discounted system return',
           title='System return on three fixed training windows')
    ax.grid(alpha=.2)
    ax.legend()
    return figure


def _held_out(run):
    rows = [row for row in run.get('final_evaluations', [])
            if row.get('completed') and row.get('physical_feasible')]
    if not rows:
        raise ValueError('No completed held-out MAPPO evaluation')
    return sorted(rows, key=lambda row: row['seed'])


def _make_performance(run):
    import matplotlib.pyplot as plt

    metrics = (
        ('hydropower_generation_TWh', 1., 'Hydropower generation (TWh)'),
        ('ecological_violation_rate', 100., 'Ecological violation rate (%)'),
        ('ecological_deficit_volume_m3', 1e-9, 'Cumulative ecological deficit (billion m³)'),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    rows = _held_out(run)
    for ax, (key, scale, label) in zip(axes, metrics):
        seeds = [row['seed'] for row in rows if row[key] is not None]
        values = [row[key]*scale for row in rows if row[key] is not None]
        ax.scatter(seeds, values, color=COLOR, label='MAPPO')
        ax.set(xlabel='Seed', ylabel=label)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=.2)
    axes[0].legend()
    days = run['episode_horizon_days']
    figure.suptitle('Held-out operational performance · {} days'.format(days))
    return figure


def _make_water_allocation(run):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(IDS))
    rows = _held_out(run)
    change = np.asarray([row['storage_change_m3'] for row in rows], dtype=float)/1e9
    if change.shape != (len(rows), len(IDS)) or not np.isfinite(change).all():
        raise ValueError('Storage change must contain five finite reservoir values per seed')
    ax.bar(positions, change.mean(axis=0), width=.32, color=COLOR, alpha=.7, label='MAPPO mean')
    ax.scatter(np.broadcast_to(positions, change.shape), change,
               color=COLOR, edgecolor='white', linewidth=.5, s=20, zorder=3)
    ax.axhline(0, color='0.3', lw=.8)
    ax.set(xticks=positions, xticklabels=IDS, ylabel='Final − initial storage (billion m³)',
           title='Held-out water allocation by reservoir')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    return figure


def _make_trajectories(run, seed):
    from examples.plot_daily_change_diagnostics import load_daily_change_csv, make_operation_figure

    if seed is None:
        seed = run['comparison_context']['shared_settings']['seeds'][0]
    row = next((row for row in _held_out(run) if row['seed'] == seed), None)
    if row is None:
        raise ValueError('No completed held-out MAPPO trajectory for seed '+str(seed))
    trajectory = run.get('figure_trajectory')
    if trajectory is not None:
        dates = [date.fromisoformat(day) for day in trajectory['dates']]
        series = dict(trajectory, dates=dates, years=[day.year for day in dates],
                      levels=np.asarray(trajectory['levels_end']),
                      releases=np.asarray(trajectory['release']))
    else:
        files = run.get('daily_change_diagnostics', {}).get('files', {}).get(str(seed), {})
        if not files.get('csv'):
            raise ValueError('Run does not record a held-out daily CSV for seed '+str(seed))
        path = Path(files['csv'])
        if not path.is_file() and run.get('log_directory'):
            path = Path(run['log_directory'])/path.name
        series = load_daily_change_csv(path)
    dates = series['dates']
    if (series['seed'] != seed or len(dates) != row['episode_horizon_days'] or
            dates[0].isoformat() != row['start_date'] or dates[-1].isoformat() != row['end_date'] or
            list(dict.fromkeys(series['years'])) != row['source_years']):
        raise ValueError('Daily trajectory disagrees with the recorded held-out evaluation')
    return make_operation_figure({'MAPPO seed '+str(seed): series})


def make_figures(run, categories=FIGURE_CATEGORIES, trajectory_seed=None):
    validate_run(run)
    builders = dict(learning=_make_learning, performance=_make_performance,
                    water_allocation=_make_water_allocation)
    figures = {}
    try:
        for name in categories:
            figures[name] = (_make_trajectories(run, trajectory_seed)
                             if name == 'trajectories' else builders[name](run))
    except Exception:
        import matplotlib.pyplot as plt
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-log', type=Path, help='Saved MAPPO log (default: read a run payload from stdin)')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output-dir', type=Path)
    destination.add_argument('--output-stem', type=Path)
    parser.add_argument('--categories', nargs='+', choices=FIGURE_CATEGORIES,
                        default=FIGURE_CATEGORIES)
    parser.add_argument('--trajectory-seed', type=int)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    categories = tuple(dict.fromkeys(args.categories))
    paths = {name: (args.output_dir/(name+'.png') if args.output_dir is not None else
                    args.output_stem.with_name(args.output_stem.stem+'_'+name+'.png'))
             for name in categories}
    try:
        for path in paths.values():
            if path.exists():
                raise ValueError('Output already exists: '+str(path))
        run = load_run(args.episode_log) if args.episode_log is not None else json.load(sys.stdin)
        figures = make_figures(run, categories, args.trajectory_seed)
        try:
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    figures[name].savefig(stream, format='png', dpi=160)
                print(str(path.resolve()))
        finally:
            for figure in figures.values():
                plt.close(figure)
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
