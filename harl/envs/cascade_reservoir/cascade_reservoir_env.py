"""Local observations and physically feasible reservoir episode transitions."""
import calendar
import numpy as np
from gym.spaces import Box, Discrete
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reservoir_physics import (
    ReservoirPhysics, InfeasibleCascade)
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings
from harl.envs.cascade_reservoir.episodes import (
    DEFAULT_EPISODE_DAYS, DEFAULT_ACTION_BINS, positive_integer)
from harl.envs.cascade_reservoir.level_ramp import (
    RAMP_LEVEL_TOLERANCE_M, daily_level_ramp_limits)

ACTOR_OBSERVATION_LAYOUT = 'cascade_local_upstream_downstream_system_v8'
CRITIC_STATE_LAYOUT = 'concatenated_five_actor_observations_v2'

# The order is a public data contract. Keep it identical for every reservoir.
ACTOR_OBSERVATION_FEATURES = (
    'current_water_level_fraction',
    'actual_total_inflow_fraction_of_release_capacity',
    'next7d_mean_incremental_inflow_forecast_standardized',
    'season_sin',
    'season_cos',
    'next_day_max_permitted_water_level_fraction',
    'next_day_min_permitted_water_level_fraction',
    'previous_requested_normalized_action',
    'current_storage_ratio',
    'remaining_regulation_capacity_ratio',
    'actual_total_release_fraction_of_release_capacity',
    'actual_turbine_release_fraction_of_turbine_capacity',
    'actual_power_fraction_of_installed_capacity',
    'upstream_water_level_fraction',
    'upstream_actual_release_fraction_of_release_capacity',
    'upstream_next7d_mean_incremental_inflow_forecast_standardized',
    'downstream_water_level_fraction',
    'downstream_remaining_regulation_capacity_ratio',
    'total_storage_fraction_of_combined_normal_storage',
    'total_power_fraction_of_total_installed_capacity',
    'flood_season_indicator',
)
ACTOR_OBSERVATION_SIZE = len(ACTOR_OBSERVATION_FEATURES)
CRITIC_STATE_SIZE = len(IDS) * ACTOR_OBSERVATION_SIZE


def actor_observation_features(use_daily_level_limits=None):
    """Return the fixed 21-feature layout for either daily-limit setting."""
    return ACTOR_OBSERVATION_FEATURES


def critic_state_size(use_daily_level_limits=None):
    """The critic receives WDD/BHT/XLD/XJB/THR observations in that order."""
    return CRITIC_STATE_SIZE


def _level_curve(data, storage):
    """Evaluate the supplied monotone storage-to-level anchors on row vectors."""
    storage = np.asarray(storage, dtype=np.float64)
    if storage.ndim != 2 or storage.shape[1] != len(IDS):
        raise ValueError('Water-level lookup requires a (rows, 5) storage matrix')
    levels = np.empty_like(storage)
    for i, name in enumerate(IDS):
        points = data.anchors[name]
        levels[:, i] = np.interp(storage[:, i], points[:, 1], points[:, 0])
    return levels


def actor_observations_from_components(data, level_limits, storage, previous_release,
        previous_requested_action, previous_power, previous_turbine, index):
    """Build the 21-feature layout from decision-time components.

    ``actual_total_inflow`` is the physically available inflow at the decision
    instant: external/incremental inflow plus the previous day's executed
    upstream release. The current upstream release is not known yet. The
    forecast is the supplied issued forecast of local incremental inflow and is
    standardized with training-year actual-inflow statistics, never recomputed
    from future observations. Levels use the configured operating-level range;
    flows use release/turbine capacity; power uses installed capacity.
    """
    storage = np.asarray(storage, dtype=np.float64)
    if storage.ndim != 2 or storage.shape[1] != len(IDS):
        raise ValueError('Observation storage must have shape (rows, 5)')
    rows = storage.shape[0]
    index = np.asarray(index, dtype=np.int64).reshape(-1)
    if index.shape != (rows,):
        raise ValueError('Observation index must have one entry per storage row')
    fields = (previous_release, previous_requested_action, previous_power, previous_turbine)
    arrays = [np.asarray(value, dtype=np.float64) for value in fields]
    if any(value.shape != storage.shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Observation components must be finite (rows, 5) matrices')
    previous_release, previous_requested_action, previous_power, previous_turbine = arrays
    if ((previous_requested_action < 0.) | (previous_requested_action > 1.)).any():
        raise ValueError('Previous requested actions must be normalized to [0, 1]')

    levels = _level_curve(data, storage)
    external = np.asarray(data.external[index], dtype=np.float64)
    forecast = np.asarray(data.inflow_forecast[index], dtype=np.float64)
    if external.shape != storage.shape or forecast.shape != storage.shape:
        raise ValueError('Observation forcing arrays must match storage rows')
    total_inflow = external.copy()
    total_inflow[:, 1:] += previous_release[:, :-1]
    forecast_standardized = (forecast-data.flow_mean)/data.flow_std

    storage_span = data.vmax-data.vmin
    level_min, level_max = data.levels(data.vmin), data.levels(data.vmax)
    level_span = level_max-level_min
    storage_ratio = (storage-data.vmin)/storage_span
    remaining_ratio = np.clip((data.vmax-storage)/storage_span, 0., 1.)
    release_fraction = previous_release/data.release_cap
    turbine_fraction = previous_turbine/data.turbine_cap
    power_fraction = previous_power/data.power_cap

    if level_limits is None:
        next_lower = np.broadcast_to(data.levels(data.vmin), storage.shape)
        next_upper = np.broadcast_to(data.levels(data.vmax), storage.shape)
    else:
        next_lower = np.asarray(level_limits.level_lower[index], dtype=np.float64)
        next_upper = np.asarray(level_limits.level_upper[index], dtype=np.float64)
    if next_lower.shape != storage.shape or next_upper.shape != storage.shape:
        raise ValueError('Next-day level bounds must match observation rows')

    phases = np.asarray([2.*np.pi*(data.dates[i].timetuple().tm_yday-1)/
                         (366 if calendar.isleap(data.dates[i].year) else 365)
                         for i in index], dtype=np.float64)
    flood = np.asarray([1. if 6 <= data.dates[i].month <= 9 else 0. for i in index], dtype=np.float64)
    system_storage = storage.sum(axis=1)/data.vmax.sum()
    system_power = previous_power.sum(axis=1)/data.power_cap.sum()

    observations = np.zeros((rows, len(IDS), ACTOR_OBSERVATION_SIZE), dtype=np.float64)
    for i in range(len(IDS)):
        observations[:, i, 0] = (levels[:, i]-level_min[i])/level_span[i]
        observations[:, i, 1] = total_inflow[:, i]/data.release_cap[i]
        observations[:, i, 2] = forecast_standardized[:, i]
        observations[:, i, 3] = np.sin(phases)
        observations[:, i, 4] = np.cos(phases)
        observations[:, i, 5] = (next_upper[:, i]-level_min[i])/level_span[i]
        observations[:, i, 6] = (next_lower[:, i]-level_min[i])/level_span[i]
        observations[:, i, 7] = previous_requested_action[:, i]
        observations[:, i, 8] = storage_ratio[:, i]
        observations[:, i, 9] = remaining_ratio[:, i]
        observations[:, i, 10] = release_fraction[:, i]
        observations[:, i, 11] = turbine_fraction[:, i]
        observations[:, i, 12] = power_fraction[:, i]
        if i:
            observations[:, i, 13] = (levels[:, i-1]-level_min[i-1])/level_span[i-1]
            observations[:, i, 14] = release_fraction[:, i-1]
            observations[:, i, 15] = forecast_standardized[:, i-1]
        if i < len(IDS)-1:
            observations[:, i, 16] = (levels[:, i+1]-level_min[i+1])/level_span[i+1]
            observations[:, i, 17] = remaining_ratio[:, i+1]
        else:
            observations[:, i, 17] = 1.
        observations[:, i, 18] = system_storage
        observations[:, i, 19] = system_power
        observations[:, i, 20] = flood
    if not np.isfinite(observations).all():
        raise FloatingPointError('Non-finite reservoir actor observations')
    return observations.astype(np.float32)


class CascadeReservoirEnv:
    n_agents = 5
    observation_layout = ACTOR_OBSERVATION_LAYOUT
    critic_state_layout = CRITIC_STATE_LAYOUT

    def observation_metadata(self):
        return dict(actor_observation_layout=self.observation_layout,
                    actor_observation_size=self.observation_size,
                    actor_observation_features=list(self.observation_features),
                    critic_state_layout=self.critic_state_layout, critic_state_size=self.state_size,
                    critic_input_layout='concatenated_actor_observations_WDD_BHT_XLD_XJB_THR',
                    feature_normalization=dict(
                        water_level='(level - operating_level_min) / operating_level_span',
                        total_inflow='actual available total inflow / release capacity',
                        forecast='issued next-7-day mean incremental inflow, training-year standardized',
                        release='actual total release / release capacity',
                        turbine='actual turbine release / turbine capacity',
                        power='actual power / installed capacity',
                        storage='(storage - operating_storage_min) / operating_storage_span',
                        remaining_regulation='(normal_storage - storage) / operating_storage_span',
                        system_storage='total storage / combined normal storage',
                        system_power='total power / total installed capacity'),
                    end_storage_policy='unconstrained_within_daily_operating_bounds',
                    feasibility_mode=self.physics.feasibility_mode,
                    feasibility_mode_definition=(
                        'backward: downstream-to-upstream storage compatibility propagated before dispatch' if
                        self.physics.feasibility_mode == 'backward' else
                        'forward_local: each reservoir bounded only by its own storage interval; no upstream propagation'))

    def __init__(self, data, config, start_year=None, action_bins=DEFAULT_ACTION_BINS,
                 episode_days=DEFAULT_EPISODE_DAYS):
        self.data = data
        self.physics = ReservoirPhysics(data, config)
        self.use_daily_level_limits = config.get('use_daily_level_limits', True)
        if not isinstance(self.use_daily_level_limits, bool):
            raise ValueError('use_daily_level_limits must be a boolean')
        self.level_limits = data.load_daily_level_limits() if self.use_daily_level_limits else None
        self.level_ramp_limits = daily_level_ramp_limits(config)
        self.reward_settings = reward_settings(config)
        self.operation_reward = OperationReward(data, self.reward_settings,
                                               timestep_seconds=self.physics.dt)
        start_year = data.dates[0].year if start_year is None else int(start_year)
        self.episode_days = positive_integer(episode_days, 'episode_days')
        self.start, _ = data.indices(start_year, start_year)
        self.stop = self.start+self.episode_days
        if self.stop > len(data.dates):
            raise ValueError('Episode extends beyond the available daily data')
        self.time_scale_days = self.episode_days
        self._storage_span = data.vmax-data.vmin
        self._level_min, self._level_max = data.levels(data.vmin), data.levels(data.vmax)
        self._level_span = self._level_max-self._level_min
        if (self._storage_span <= 0.).any() or (self._level_span <= 0.).any():
            raise ValueError('Operating storage and water-level ranges must be positive')
        self.observation_features = actor_observation_features(self.use_daily_level_limits)
        self.observation_size = len(self.observation_features)
        self.state_size = critic_state_size(self.use_daily_level_limits)
        if self.observation_size != 21 or self.state_size != 105:
            raise ValueError('The cascade observation contract requires 21 actor and 105 critic features')
        self.observation_space = [Box(-np.inf, np.inf, (self.observation_size,), dtype=np.float32) for _ in range(5)]
        self.share_observation_space = [Box(-np.inf, np.inf, (self.state_size,), dtype=np.float32) for _ in range(5)]
        if (isinstance(action_bins, bool) or
                not isinstance(action_bins, (int, np.integer)) or action_bins < 2):
            raise ValueError('action_bins must be an integer >= 2')
        self.action_bins = action_bins
        self.action_space = [Discrete(action_bins) for _ in range(5)]
        self.initial_storage = data.initial_storage(data.dates[self.start].year)
        self.reset()

    def _next_state_storage_bounds(self, index):
        if self.level_limits is None:
            return self.data.vmin, self.data.vmax
        return self.level_limits.storage_lower[index], self.level_limits.storage_upper[index]

    def _next_state_level_bounds(self, index):
        if self.level_limits is None:
            return self._level_min, self._level_max
        return self.level_limits.level_lower[index], self.level_limits.level_upper[index]

    def _current_state_storage_bounds(self, index):
        """The previous executed row constrains the current state; reset uses global limits."""
        if self.level_limits is None or index == self.start:
            return self.data.vmin, self.data.vmax
        return self.level_limits.storage_lower[index-1], self.level_limits.storage_upper[index-1]

    def _check_current_storage(self, storage, index):
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise InfeasibleCascade('Starting storage must be a finite five-vector')
        lower, upper = self._current_state_storage_bounds(index)
        tolerance = np.maximum(1., 1e-10*np.abs(storage))
        if ((storage < lower-tolerance) | (storage > upper+tolerance)).any():
            raise InfeasibleCascade('Starting storage violates the preceding action\'s bounds')

    def _transition_failure(self, exc, day, actions, ramp_info):
        """Preserve structured infeasibility evidence across the environment API."""
        message = '{}; date={} storage_m3={} actions={}'.format(
            exc, day, self.storage.tolist(), actions.reshape(5).tolist())
        if not isinstance(exc, InfeasibleCascade):
            return type(exc)(message)
        conflicts = [dict(value) for value in getattr(exc, 'conflicts', [])]
        if ramp_info and not conflicts:
            conflicts = [dict(reservoir='CASCADE', reservoir_index=-1,
                              reason='physical_transition_infeasible')]
        for conflict in conflicts:
            conflict.setdefault('date', day.isoformat())
            index = conflict.get('reservoir_index')
            if not isinstance(index, int) or not 0 <= index < 5:
                continue
            conflict.setdefault('current_storage_m3', float(self.storage[index]))
            if ramp_info:
                conflict.setdefault('current_level_m',
                    float(ramp_info['current_level_m'][index]))
                conflict.setdefault('rise_limit_m_per_day',
                    float(ramp_info['ramp_rise_limit_m_per_day'][index]))
                conflict.setdefault('fall_limit_m_per_day',
                    float(ramp_info['ramp_fall_limit_m_per_day'][index]))
                conflict.setdefault('ramp_season',
                    str(ramp_info['ramp_season'][index]))
        error = InfeasibleCascade(message, conflicts=conflicts)
        if hasattr(exc, 'ramp_metadata'):
            error.ramp_metadata = exc.ramp_metadata
        error.transition_context = dict(date=day.isoformat(),
            storage_m3=self.storage.tolist(),
            actions=actions.reshape(5).astype(int).tolist())
        return error

    def _check_executed_ramp(self, result):
        """Reject a post-execution ramp violation without clipping state."""
        if not result.get('level_ramp_enabled', False):
            return
        delta = result['levels_end']-result['levels_start']
        within = ((delta <= result['ramp_rise_limit_m_per_day']+
                   RAMP_LEVEL_TOLERANCE_M) &
                  (delta >= -result['ramp_fall_limit_m_per_day']-
                   RAMP_LEVEL_TOLERANCE_M))
        result['level_change_m'] = delta
        result['level_ramp_within_limits'] = within
        if within.all():
            return
        conflicts = []
        for index in np.flatnonzero(~within):
            conflicts.append(dict(date=result['date'], reservoir=IDS[index],
                reservoir_index=int(index), reason='executed_daily_level_ramp_violation',
                current_level_m=float(result['levels_start'][index]),
                rise_limit_m_per_day=float(result['ramp_rise_limit_m_per_day'][index]),
                fall_limit_m_per_day=float(result['ramp_fall_limit_m_per_day'][index]),
                ramp_season=str(result['ramp_season'][index]),
                ramp_level_lower_m=float(result['ramp_level_lower_m'][index]),
                ramp_level_upper_m=float(result['ramp_level_upper_m'][index]),
                existing_level_lower_m=float(result['daily_level_lower_m'][index]),
                existing_level_upper_m=float(result['daily_level_upper_m'][index]),
                effective_level_lower_m=float(result['effective_level_lower_m'][index]),
                effective_level_upper_m=float(result['effective_level_upper_m'][index]),
                current_storage_m3=float(result['storage_start'][index]),
                existing_storage_lower_m3=float(result['daily_storage_lower_m3'][index]),
                existing_storage_upper_m3=float(result['daily_storage_upper_m3'][index]),
                effective_storage_lower_m3=float(result['storage_lower_m3'][index]),
                effective_storage_upper_m3=float(result['storage_upper_m3'][index])))
        raise InfeasibleCascade(
            'Executed transition violates the hard daily water-level ramp limit',
            conflicts=conflicts)

    def reset(self):
        storage = self.initial_storage.copy()
        self._check_current_storage(storage, self.start)
        self.index = self.start
        self.storage = storage
        self.previous_release = np.zeros(5)
        self.previous_requested_action = np.zeros(5)
        self.previous_power = np.zeros(5)
        self.previous_turbine = np.zeros(5)
        self.last_transition = None
        return self._observations()

    def state(self):
        """Concatenate WDD/BHT/XLD/XJB/THR actor observations for the critic."""
        if self.index == self.stop:
            return np.zeros(self.state_size, dtype=np.float32)
        return self.actor_observations().reshape(-1).copy()

    def actor_observations(self):
        """Return the fixed local/upstream/downstream/shared-system layout."""
        if self.index == self.stop:
            return np.zeros((5,self.observation_size),dtype=np.float32)
        return actor_observations_from_components(self.data, self.level_limits,
            self.storage[None], self.previous_release[None],
            self.previous_requested_action[None], self.previous_power[None], self.previous_turbine[None],
            np.asarray([self.index], dtype=np.int64))[0]

    def _observations(self):
        state = np.repeat(self.state()[None,:],5,axis=0)
        return self.actor_observations(), state, None

    def normalize_actions(self, actions):
        actions = np.asarray(actions,dtype=np.float64)
        if actions.shape not in ((5,),(5,1)) or not np.isfinite(actions).all():
            raise ValueError('Five finite scalar actions required')
        actions = actions.reshape(5)
        if ((actions != np.floor(actions)) | (actions < 0) | (actions >= self.action_bins)).any():
            raise ValueError('Discrete actions must be integer indices in [0,action_bins-1]')
        return actions/(self.action_bins-1)

    def step(self, actions):
        if self.index >= self.stop:
            raise RuntimeError('Episode finished; reset explicitly')
        actions = np.asarray(actions)
        normalized = self.normalize_actions(actions)
        ramp_info = {}
        try:
            self._check_current_storage(self.storage,self.index)
            lower, upper = self._next_state_storage_bounds(self.index)
            next_level_lower, next_level_upper = self._next_state_level_bounds(self.index)
            if self.level_ramp_limits.enabled:
                lower, upper, ramp_info = self.level_ramp_limits.effective_storage_bounds(
                    self.data, self.storage, self.data.dates[self.index], lower, upper,
                    next_level_lower, next_level_upper)
            bounds = dict(storage_lower=lower, storage_upper=upper)
            result = self.physics.transition(self.storage,self.data.external[self.index],normalized,**bounds)
        except RuntimeError as exc:
            raise self._transition_failure(exc, self.data.dates[self.index],
                                          actions, ramp_info) from exc
        result['date'] = self.data.dates[self.index].isoformat()
        result['sampled_actions'] = actions.reshape(5).astype(np.int64).copy()
        result['requested_normalized_actions'] = normalized.copy()
        result['action_bins'] = self.action_bins
        result['timestep_seconds'] = self.physics.dt
        result['daily_level_limits'] = self.use_daily_level_limits
        result.update(ramp_info if ramp_info else dict(level_ramp_enabled=False))
        result['normalized_actions'] = result['actions'].copy()
        result['actions'] = np.where(result['controls'] > 0,actions.reshape(5),0).astype(np.int64)
        result['levels_start'] = self.physics._forebay_levels(result['storage_start'])
        result['levels_end'] = self.data.levels(result['storage_end'])
        if ramp_info:
            next_level_lower = ramp_info['effective_level_lower_m']
            next_level_upper = ramp_info['effective_level_upper_m']
        result.update(next_state_level_lower_m=next_level_lower.copy(),
                      next_state_level_upper_m=next_level_upper.copy())
        self._check_executed_ramp(result)
        self.operation_reward.apply(result)
        infos = [dict(bad_transition=False,use_daily_level_limits=self.use_daily_level_limits) for _ in range(5)]
        if self.level_limits is not None:
            for j,info in enumerate(infos):
                info.update(storage_lower_m3=float(lower[j]), storage_upper_m3=float(upper[j]),
                            next_state_level_lower_m=float(result['next_state_level_lower_m'][j]),
                            next_state_level_upper_m=float(result['next_state_level_upper_m'][j]))
        if ramp_info:
            for j,info in enumerate(infos):
                info.update(level_ramp_enabled=True,
                    rise_limit_m_per_day=float(result['ramp_rise_limit_m_per_day'][j]),
                    fall_limit_m_per_day=float(result['ramp_fall_limit_m_per_day'][j]),
                    level_change_m=float(result['level_change_m'][j]),
                    level_ramp_within_limits=bool(result['level_ramp_within_limits'][j]))
        self.last_transition = result
        self.storage = result['storage_end'].copy()
        self.previous_release = result['release'].copy()
        self.previous_requested_action = normalized.copy()
        self.previous_power = result['power'].copy()
        self.previous_turbine = result['turbine'].copy()
        self.index += 1
        done = self.index == self.stop
        obs,state,available = self._observations()
        return obs,state,result['mixed_rewards'].astype(np.float32).reshape(5,1), \
            np.full(5,done,dtype=bool),infos,available
