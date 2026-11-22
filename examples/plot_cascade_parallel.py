"""Read and validate cascade logs, and render one saved run."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from examples.train_cascade_parallel import verify_artifact_manifest
except ImportError:  # A minimal NumPy/Matplotlib interpreter can still replot evidence.
    verify_artifact_manifest = None


VIOLATION_TERMS = ('ecological_flow',)
VIOLATION_SUFFIXES = ('violation_rate', 'violation_count', 'active_reservoir_days')
COMPLETION_PROVENANCE_FIELDS = ('artifact_manifest', 'artifact_inventory',
    'continuation_provenance', 'completion_record', 'artifact_manifest_integrity')


def provenance_core(provenance):
    """Compare immutable run identity without its post-run artifact audit fields."""
    if not isinstance(provenance, dict):
        raise ValueError('Run provenance must be an object')
    return {key:value for key, value in provenance.items()
            if key not in COMPLETION_PROVENANCE_FIELDS}


def learning_return_kind(payload):
    """Resolve MAPPO reward semantics from the saved provenance."""
    objective = payload.get('learning_objective', {})
    if not isinstance(objective, dict):
        raise ValueError('learning_objective must be a mapping')
    layout = objective.get('critic_reward_layout', payload.get('critic_reward_layout'))
    if layout in ('mixed_by_reservoir', 'mixed_by_reservoir_then_legacy_team'):
        return 'individual_mixed'
    source = objective.get('advantage_source')
    if source == 'individual_mixed':
        return 'individual_mixed'
    if layout is not None or source is not None:
        raise ValueError('Unsupported plotting learning objective')
    return 'legacy_team'


def learning_return_label(payload):
    return {'individual_mixed': 'Mean individual mixed learning return',
            'legacy_team': 'Legacy-team return (learning objective unavailable)'}[learning_return_kind(payload)]


def _finite_return(value, field):
    if isinstance(value, bool) or np.ndim(value) != 0:
        raise ValueError(field+' must be a finite scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(field+' must be a finite scalar') from exc
    if not np.isfinite(result):
        raise ValueError(field+' must be a finite scalar')
    return result


def _mixed_returns(value, field):
    result = np.asarray(value, dtype=float)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError(field+' must contain five finite reservoir returns')
    return result


def learning_return_value(payload, row, phase):
    """Select the learned reward; never substitute legacy accounting for it."""
    if phase not in ('training', 'evaluation'):
        raise ValueError('Unknown return plotting phase: '+str(phase))
    if phase == 'evaluation' and row.get('status') == 'failed':
        return np.nan
    kind = learning_return_kind(payload)
    if kind == 'legacy_team':
        field = 'episode_reward' if phase == 'training' else 'evaluator_return'
    else:
        field = 'episode_mixed_rewards' if phase == 'training' else 'mixed_evaluator_returns'
        if field in row:
            return float(_mixed_returns(row[field], field).mean())
        if phase == 'training' and all(key in row for key in
                ('episode_system_reward', 'episode_global_reward')):
            # With five reservoirs and alpha=.5, sum(mixed)=system+2*global.
            # The identity also holds for the lane-averaged in-memory history.
            return (_finite_return(row['episode_system_reward'], 'episode_system_reward')+
                    2.*_finite_return(row['episode_global_reward'], 'episode_global_reward'))/5.
    if field not in row:
        raise ValueError('Missing '+field+' for '+learning_return_label(payload)+
                         '; learning returns cannot be reconstructed from legacy reward alone')
    return _finite_return(row[field], field)


def validate_learning_evaluation(payload, row):
    """Check the selected round return against all three complete windows."""
    if learning_return_kind(payload) == 'legacy_team':
        return
    value = learning_return_value(payload, row, 'evaluation')
    windows = [learning_return_value(payload, window, 'evaluation') for window in row['windows']]
    if row['status'] == 'ok' and not np.isclose(value, np.mean(windows), rtol=1e-10, atol=1e-10):
        raise ValueError('Learning evaluator return must equal the mean of all three fixed windows')


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def evaluation_round(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('evaluation_round must be a nonnegative integer')
    return int(value)


def window_signature(windows):
    """Identify the fixed scenarios, including initial conditions and forcing."""
    if not isinstance(windows, list) or len(windows) != 3:
        raise ValueError('Periodic evaluation must use exactly three fixed windows')
    fields = ('window_id', 'window_start_year', 'source_years', 'episode_horizon_days',
              'start_date', 'end_date', 'initial_storage_m3', 'initial_level_m',
              'external_inflow_sha256')
    identities = [{key:row[key] for key in fields} for row in windows]
    if len({row['window_id'] for row in identities}) != 3:
        raise ValueError('Periodic window identifiers must be unique')
    for row in identities:
        positive_integer(row['episode_horizon_days'], 'periodic episode_horizon_days')
        if not row['source_years'] or len(set(row['source_years'])) != len(row['source_years']):
            raise ValueError('Periodic source years must be nonempty and unique')
        for name in ('initial_storage_m3', 'initial_level_m'):
            values = np.asarray(row[name], dtype=float)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError(name+' must contain five finite values')
    return json.dumps(identities, sort_keys=True, allow_nan=False)


def validate_periodic_row(row, expected_signature=None):
    """Keep failures as failures, and never substitute a survivor-only mean."""
    evaluation_round(row['evaluation_round'])
    signature = window_signature(row['windows'])
    if expected_signature is not None and signature != expected_signature:
        raise ValueError('Periodic evaluation scenarios or initial conditions changed')
    if row.get('window_count') != 3:
        raise ValueError('Periodic window_count must equal three')
    statuses = [window.get('status') for window in row['windows']]
    if any(status not in ('ok', 'failed') for status in statuses):
        raise ValueError('Unknown periodic window status')
    successful = statuses.count('ok')
    if row.get('successful_window_count') != successful:
        raise ValueError('Periodic successful_window_count disagrees with the window results')
    values = []
    for window in row['windows']:
        value = window.get('evaluator_return')
        if window['status'] == 'ok':
            if value is None or isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError('Successful periodic windows require finite evaluator returns')
            values.append(float(value))
        elif value is not None:
            raise ValueError('Failed periodic windows must not report a completed return')
    value = row.get('evaluator_return')
    if successful == 3:
        if row.get('status') != 'ok' or value is None or isinstance(value, bool):
            raise ValueError('A successful periodic evaluation requires a completed mean return')
        if not np.isfinite(float(value)) or not np.isclose(float(value), np.mean(values), rtol=1e-10, atol=1e-10):
            raise ValueError('Evaluator return must equal the mean of all three fixed windows')
    elif row.get('status') != 'failed' or value is not None:
        raise ValueError('A failed periodic evaluation must leave the aggregate return null')
    validate_violation_metrics(row)
    return signature


def validate_violation_metrics(row):
    """Validate actual violation frequencies, keeping legacy records readable.

    Weighted shortfall penalties measure severity, so they cannot reconstruct
    the number of reservoir-days on which a requirement was violated.
    """
    fields = tuple(term+'_'+suffix for term in VIOLATION_TERMS for suffix in VIOLATION_SUFFIXES)
    scopes = [row]+row['windows']
    if not any(field in scope for scope in scopes for field in fields):
        return False
    if any(field not in scope for scope in scopes for field in fields):
        raise ValueError('Incomplete periodic violation metrics: every window and round require rates, counts and active reservoir-days')
    if row.get('violation_rate_units', 'fraction') != 'fraction':
        raise ValueError('Stored periodic violation rates must be fractions, not percentages')
    for scope in scopes:
        for term in VIOLATION_TERMS:
            rate, count, active = [scope[term+'_'+suffix] for suffix in VIOLATION_SUFFIXES]
            if scope['status'] == 'failed':
                if any(value is not None for value in (rate, count, active)):
                    raise ValueError('Failed periodic evaluations must leave completed violation metrics null')
                continue
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
                   for value in (count, active)) or count > active:
                raise ValueError('Violation counts and active reservoir-days must be nonnegative integers with count <= active')
            if scope is not row and active > 5*scope['episode_horizon_days']:
                raise ValueError('Active reservoir-days exceed the five-reservoir episode horizon')
            if active == 0:
                if rate is not None:
                    raise ValueError('A requirement without active reservoir-days has no defined violation rate')
            elif (rate is None or isinstance(rate, bool) or not np.isfinite(float(rate)) or
                  not 0. <= float(rate) <= 1. or
                  not np.isclose(float(rate), count/active, rtol=1e-10, atol=1e-12)):
                raise ValueError('Violation rate must be the fraction of violating active reservoir-days')
            if scope is not row and 'service_shortfalls' in scope:
                summary = scope['service_shortfalls'][term]
                if summary['reservoir_day_count'] != count or summary['active_reservoir_days'] != active:
                    raise ValueError('Violation counts disagree with the service shortfall summary')
    if row['status'] == 'ok':
        for term in VIOLATION_TERMS:
            for suffix in ('violation_count', 'active_reservoir_days'):
                field = term+'_'+suffix
                if row[field] != sum(window[field] for window in row['windows']):
                    raise ValueError('Aggregate violation counts must sum all three fixed windows')
    return True


def violation_metrics_available(payload):
    """Do not turn absent legacy frequencies into zero-valued curves."""
    _, histories = available_histories(payload, 'evaluation_histories')
    available = [validate_violation_metrics(row) for rows in histories for row in rows]
    if any(available) and not all(available):
        raise ValueError('Some evaluation rounds lack violation counts; mixed legacy/new histories cannot provide a complete violation-rate curve')
    return bool(available) and all(available)


def load_episode_log(path):
    """Stream summary records, retaining collector and evaluator histories separately."""
    configuration, histories, updates = None, {}, {}
    evaluations, last_rounds, scenarios = {}, {}, None
    run_identity, completion_identity = None, None
    final_evaluations, daily_change_diagnostics = {}, {}
    pending_episodes = {}
    pending_lanes = {}
    # Step writers put record_type first. Other key orders still use the JSON
    # parser below, so existing logs do not depend on object-key order.
    step_prefix = re.compile(r'^\s*\{\s*"record_type"\s*:\s*"(?:training_step|evaluation_step)"\s*[,}]')
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if step_prefix.match(line):
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                raise ValueError('Invalid JSON at line {}'.format(line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError('Log records must be JSON objects at line {}'.format(line_number))
            kind = row.get('record_type')
            if kind not in ('run_configuration', 'training_episode', 'training_update',
                            'periodic_evaluation', 'final_evaluation_sequence', 'run_complete'):
                continue
            try:
                if kind == 'run_complete':
                    completion_identity = (row.get('run_id'), row.get('provenance'))
                    daily_change_diagnostics = row.get('daily_change_diagnostics', {})
                    continue
                if kind == 'run_configuration':
                    if configuration is not None:
                        raise ValueError('A plot input must contain exactly one run configuration')
                    if str(row.get('algorithm', row.get('algorithm_name', ''))).lower() != 'mappo':
                        raise ValueError('Plot input requires an explicitly labeled MAPPO log')
                    gamma = float(row['config']['algo']['gamma'])
                    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
                        raise ValueError('Gamma must be finite and in [0, 1]')
                    seeds = row['config']['experiment']['seeds']
                    if (not isinstance(seeds, list) or not seeds or
                            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or
                            len(set(seeds)) != len(seeds)):
                        raise ValueError('Configured seeds must be unique integers')
                    configuration = dict(gamma=gamma, configured_seeds=seeds,
                        episode_horizon_days=positive_integer(row['episode_horizon_days'], 'episode_horizon_days'),
                        parallel_envs=positive_integer(row['parallel_environments'], 'parallel_environments'))
                    for name in ('run_id', 'provenance'):
                        if name in row:
                            configuration[name] = row[name]
                    run_identity = (row.get('run_id'), row.get('provenance'))
                    if 'updates_per_seed' in row:
                        configuration['updates_per_seed'] = positive_integer(row['updates_per_seed'], 'updates_per_seed')
                    periodic = row.get('periodic_evaluation', {})
                    if 'windows' in periodic:
                        scenarios = window_signature(periodic['windows'])
                    configuration['training_years'] = row.get('training_years')
                    configuration['evaluation_source_years'] = row.get('evaluation_source_years', [])
                    for name in ('algorithm', 'algorithm_name', 'learning_objective',
                                 'critic_reward_layout', 'actor_architecture'):
                        if name in row:
                            configuration[name] = row[name]
                    continue
                if configuration is None:
                    raise ValueError('A run_configuration must precede training/evaluation results')
                recorded_run_id = row.get('run_id',
                    row.get('learning_rewards', {}).get('run_id')
                    if isinstance(row.get('learning_rewards'), dict) else None)
                if recorded_run_id is not None and recorded_run_id != configuration.get('run_id'):
                    raise ValueError('A result record belongs to a different run_id')
                seed = row['seed']
                if isinstance(seed, bool) or not isinstance(seed, int) or seed not in configuration['configured_seeds']:
                    raise ValueError('Result seed is absent from the run configuration')
                if kind == 'final_evaluation_sequence':
                    if seed in final_evaluations:
                        raise ValueError('Duplicate final evaluation seed')
                    final_evaluations[seed] = row
                    continue
                if kind == 'training_episode':
                    if not row.get('completed') or learning_return_kind(configuration) == 'legacy_team':
                        continue
                    update = positive_integer(row['collecting_for_update'], 'collecting_for_update')
                    episode = positive_integer(row['episode'], 'episode')
                    batch = pending_episodes.setdefault((seed, update), {})
                    if episode in batch or update <= updates.get(seed, 0):
                        raise ValueError('Duplicate or already plotted training episode')
                    lane = row['parallel_env_index']
                    lanes = pending_lanes.setdefault((seed, update), set())
                    if (isinstance(lane, bool) or not isinstance(lane, int) or
                            not 0 <= lane < configuration['parallel_envs'] or lane in lanes):
                        raise ValueError('Completed training episodes must have distinct valid environment lanes')
                    rewards = row['learning_rewards']
                    # Validate the selected objective before retaining its data.
                    learning_return_value(configuration, rewards, 'training')
                    batch[episode] = rewards
                    lanes.add(lane)
                    continue
                if kind == 'periodic_evaluation':
                    number = evaluation_round(row['evaluation_round'])
                    if seed not in last_rounds and number != 0:
                        raise ValueError('Periodic evaluation history must start at round zero')
                    if seed in last_rounds and number <= last_rounds[seed]:
                        raise ValueError('Duplicate or non-increasing evaluation round for seed {}'.format(seed))
                    if number > configuration.get('updates_per_seed', number):
                        raise ValueError('Evaluation round exceeds the configured update budget')
                    signature = validate_periodic_row(row, scenarios)
                    validate_learning_evaluation(configuration, row)
                    if scenarios is None:
                        scenarios = signature
                    training_years = configuration['training_years']
                    held_out = set(configuration['evaluation_source_years'])
                    for window in row['windows']:
                        years = set(window['source_years'])
                        if years & held_out or (training_years is not None and not years <= set(training_years)):
                            raise ValueError('Periodic evaluation windows must use training years only')
                    if 'gamma' in row and float(row['gamma']) != configuration['gamma']:
                        raise ValueError('Evaluator gamma differs from the run configuration')
                    evaluations.setdefault(seed, []).append(row)
                    last_rounds[seed] = number
                    continue
                if (row['episode_horizon_days'] != configuration['episode_horizon_days'] or
                        row['parallel_envs'] != configuration['parallel_envs']):
                    raise ValueError('Training update horizon/workers disagree with the run configuration')
                step = positive_integer(row['training_steps'], 'training_steps')
                update = positive_integer(row['update'], 'update')
                history = histories.setdefault(seed, [])
                if history and (step <= history[-1]['training_steps'] or update <= updates[seed]):
                    raise ValueError('Duplicate or non-increasing training update for seed {}'.format(seed))
                reward = float(row['episode_reward_mean'])
                generation = float(row['episode_generation_TWh_mean'])
                if not np.isfinite([reward, generation]).all():
                    raise ValueError('Non-finite training curve metric')
                point = dict(training_steps=step, episode_reward=reward, episode_energy_TWh=generation)
                if learning_return_kind(configuration) != 'legacy_team':
                    batch = pending_episodes.pop((seed, update), {})
                    pending_lanes.pop((seed, update), None)
                    if len(batch) != configuration['parallel_envs']:
                        raise ValueError('Learning-return plot requires all completed training episodes for update '+str(update))
                    rewards = list(batch.values())
                    for field in ('episode_system_reward', 'episode_global_reward'):
                        if all(field in episode for episode in rewards):
                            point[field] = float(np.mean([_finite_return(episode[field], field)
                                                         for episode in rewards]))
                    if all('episode_mixed_rewards' in episode for episode in rewards):
                        point['episode_mixed_rewards'] = np.mean([
                            _mixed_returns(episode['episode_mixed_rewards'], 'episode_mixed_rewards')
                            for episode in rewards], axis=0).tolist()
                    learning_return_value(configuration, point, 'training')
                history.append(point)
                updates[seed] = update
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid plotting input at line {}: {}'.format(line_number, exc)) from exc
    if configuration is None:
        raise ValueError('No run_configuration found in the episode log')
    if completion_identity is not None:
        if completion_identity[0] != run_identity[0]:
            raise ValueError('run_complete provenance disagrees with run_configuration')
        if provenance_core(completion_identity[1]) != provenance_core(run_identity[1]):
            raise ValueError('run_complete provenance disagrees with run_configuration')
    log_path = Path(path)
    manifest_candidates = [log_path.with_name(log_path.stem+'_manifest.json'),
                           log_path.parent/'run_manifest.json']
    if completion_identity is not None:
        recorded_manifest = (completion_identity[1] or {}).get('artifact_manifest')
        if isinstance(recorded_manifest, str) and recorded_manifest:
            manifest_candidates.append(Path(recorded_manifest))
    manifest_path = next((candidate for candidate in manifest_candidates
                          if candidate.is_file()), None)
    if verify_artifact_manifest is not None and manifest_path is not None:
        manifest = verify_artifact_manifest(manifest_path)
        if configuration.get('run_id') != manifest.get('run_id'):
            raise ValueError('Episode log and artifact manifest belong to different runs')
        def binds_log(row):
            artifact = Path(row['path'])
            if not artifact.is_absolute():
                artifact = manifest_path.parent/artifact
            return artifact.resolve() == log_path.resolve()
        log_artifact = next((row for row in manifest['artifacts'] if binds_log(row)), None)
        if log_artifact is None:
            raise ValueError('Artifact manifest does not bind the episode log')
        if log_artifact['sha256'] != hashlib.sha256(log_path.read_bytes()).hexdigest():
            # The manifest hashes the final closed log. A completed run cannot
            # silently substitute another file with the same run_configuration.
            raise ValueError('Episode log hash disagrees with its artifact manifest')
    if run_identity[0] is not None:
        run_provenance = run_identity[1]
        if (not isinstance(run_provenance, dict) or
                run_provenance.get('run_id') != run_identity[0]):
            raise ValueError('Run identity requires matching provenance')
        source_hashes = run_provenance.get('source_code_hashes', {})
        if not isinstance(source_hashes, dict):
            raise ValueError('Run provenance source code hashes must be an object')
        source_hashes = {str(name).replace('\\', '/'):value
                         for name, value in source_hashes.items()}
        if source_hashes.get('examples/plot_cascade_parallel.py') is None:
            raise ValueError('Run provenance lacks the plotting transport hash')
    seeds = [seed for seed in configuration['configured_seeds'] if histories.get(seed) or evaluations.get(seed)]
    if not seeds:
        raise ValueError('No training updates or periodic evaluations have been recorded yet')
    return dict(configuration, seeds=seeds, histories=[histories.get(seed, []) for seed in seeds],
                evaluation_histories=[evaluations.get(seed, []) for seed in seeds],
                final_evaluations=list(final_evaluations.values()),
                daily_change_diagnostics=daily_change_diagnostics)


def available_histories(payload, name):
    histories = payload.get(name, [])
    if not histories:
        return [], []
    seeds = payload['seeds']
    if len(histories) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('One history per unique seed required')
    selected = [(seed, rows) for seed, rows in zip(seeds, histories) if rows]
    return [seed for seed, _ in selected], [rows for _, rows in selected]


def periodic_histories(payload):
    """Validate one common set of fixed scenarios and evaluation round grids."""
    seeds, histories = available_histories(payload, 'evaluation_histories')
    if not histories:
        raise ValueError('No periodic evaluator records to plot')
    grids, scenarios = [], None
    for seed, rows in zip(seeds, histories):
        grid = np.array([evaluation_round(row['evaluation_round']) for row in rows], dtype=int)
        if grid[0] != 0 or np.any(np.diff(grid) <= 0):
            raise ValueError('Evaluation rounds must start at zero and increase without duplicates')
        for row in rows:
            if row.get('seed') != seed:
                raise ValueError('Periodic history seed disagrees with its curve')
            scenarios = validate_periodic_row(row, scenarios)
            validate_learning_evaluation(payload, row)
        grids.append(grid)
    return seeds, histories, grids


def main():
    """Render the retained figures from one saved run."""
    from examples.plot_cascade import main as plot
    plot()


if __name__ == '__main__':
    main()
