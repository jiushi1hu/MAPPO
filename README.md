# Five-reservoir MAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR cascade using a specialized
feed-forward HARL implementation. The objective combines hydropower generation
with a soft ecological-flow penalty. Water balance, storage envelopes, outlet
capacities are hard physical constraints. Final storage is free within the
operating bounds; initial/final inventories and net depletion are reported
without an equality target or pass/fail test.

## Run

Use the `harl` Conda environment for training on the development machine.
It provides PyTorch and Gym. On a fresh machine, activate a compatible Python
environment and install the dependencies declared in `setup.py`:

```bash
python -m pip install -e .
```

Validate data, episode windows and observations without training:

```bash
conda run --no-capture-output -n harl python -B examples/train_cascade.py --check-data-only
```

Train MAPPO and save the inputs for plotting.

```bash
conda run --no-capture-output -n harl python -B examples/train_cascade.py \
  --episodes 300 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 --diagnostics \
  --episode-log runs/mappo/episodes.jsonl --no-plot
```

The episode budget applies to each seed.
With `--no-plot`, the run still writes its requested JSONL and CSV data. Omit
that flag to render figures when an episode log or plot stem is supplied.
`--plot-output STEM` sets the PNG filename stem and `--plot-python PATH`
selects the interpreter for automatic rendering. Existing outputs are exclusive.

By default results stay in memory. Add a fresh `--episode-log` path to retain
JSONL summaries and per-seed daily CSVs. Add
`--save-checkpoints` to save final policies, or `--checkpoint-interval N` to also
save every N updates. Checkpoints support inference replay, not training resume.

## Data, observations and physical model

Inputs are in `data/cascade_reservoir_new/`. Actual historical daily external
inflows cover 1965–2024. BHT/XLD/XJB/THR columns are incremental interval inflows;
upstream executed release is added by the physical cascade. Releases propagate
upstream to downstream within one daily transition, with zero routing delay.

`five_reservoir_daily_level_bounds_1965_2024.csv` supplies the permitted next-state
storage and level for each action date. Its row for date t constrains the state
after action t; no extra date shift is applied. Previous executed bounds constrain
the current state, including across source-year joins.

Hard daily water-level ramps are configured under `daily_level_ramp_limits` in
`cascade_reservoir.yaml`. WDD/BHT/XLD/XJB use constant absolute limits of
1/2/3/4 m per day. THR uses 0.6 m/day in the dry months and 2.0 m/day in June--
September; its August--October refill rise limit is 3.0 m/day while downward
movement keeps the dry/flood seasonal limit. The current level plus/minus the
applicable rise/fall band is intersected with daily and operating level/storage
bounds before release dispatch. A disjoint intersection aborts the transition
with structured conflict evidence. Executed levels are checked again after mass
balance; storage and water levels are never clipped after execution.

Every decision, including the last day, uses the same physical release mapping.
There is no prescribed final water level, reserve-window override, terminal
projection, seasonal refill budget or terminal-deviation reward. Storage is
always calculated from executed flows. Episode boundaries still stop value
bootstrapping; this is independent of reservoir inventory requirements.

Actor observations use one fixed 22-feature layout for all five reservoirs. The
first 14 local features are normalized current water level; actual total inflow
(external or incremental inflow plus the previous day's executed upstream
release) divided by release capacity; the supplied issued next-seven-day mean
incremental inflow forecast, standardized using training-year actual-inflow
statistics; seasonal sine and cosine; next-day maximum and minimum permitted
water levels; the previous requested normalized action; current storage ratio;
remaining regulation-capacity ratio; previous actual total release, turbine
release and power, each divided by its capacity; and the previous realized action
after environmental correction. The next three features are the immediate
upstream reservoir's water level, actual release and issued forecast; WDD uses
`[0, 0, 0]`. The next two are the immediate downstream reservoir's water level
and remaining regulation-capacity ratio; THR uses `[0, 1]`. The final three are
total storage divided by combined normal storage, total realized power divided by
total installed capacity, and a flood-season indicator equal to 1 from June 1
through September 30 and 0 otherwise.

Forecasts are read from the supplied file, including at nonconsecutive held-out
year joins; they are not recomputed from future actual inflows. The centralized
critic receives the five observations concatenated in WDD/BHT/XLD/XJB/THR order,
for 110 input features. Checkpoints record the observation and critic layouts;
version 2 checkpoints use this contract, and earlier checkpoints cannot be loaded
because their observation dimensions and source hashes differ.

For categorical action k among K commands:

```text
u = k / (K - 1)
release = feasible_low + u² * (feasible_high - feasible_low)
```

Feasible release intervals enforce storage bounds, daily level-ramp limits,
outlet capacities and downstream compatibility. Physically forced intervals
are masked in the actor loss. The default
100-command grid can be changed with `--action-bins`.

`feasibility_mode` in `cascade_reservoir.yaml` selects how downstream
compatibility enters the interval, and `--feasibility-mode` overrides it per run:

- `backward` (default, original): each reservoir's interval is propagated
  upstream from the downstream storage envelope before any release is chosen, so
  one joint command vector is feasible for the whole cascade.
- `forward_local` (ablation): the propagation pass is disabled. Each reservoir is
  bounded only by its own storage interval, capacity and level ramps, and the
  upstream release still becomes the next reservoir's inflow. A reservoir whose
  interval is empty raises `InfeasibleCascade` with per-reservoir conflict
  evidence; nothing is clipped, relaxed, or rounded to stay feasible.

Both modes record `forced_interval`, `empty_interval`,
`release_interval_fraction` and `feasibility_mode` on every transition, and
episode and evaluation records carry forced/empty counts, infeasible-interval
rates, mass-balance residuals and level trajectories.

The paired ablation runs both modes with the same seeds and settings:

```bash
python -B examples/run_feasibility_ablation.py --episodes 60 --seeds 1 2 3
```

It writes `ablation_report.md` and `ablation_summary.json` under
`runs/feasibility_ablation/`.

Generation uses turbine flow, installed capacity and net head. The default
`storage_sensitivity` head model depends on mean start/end forebay elevation and
supplied slope/offset assumptions; this is a sensitivity model. `--head-model
fixed` selects typical constant heads.

## Ecological flow and rewards

`reservoir_service_requirements.csv` supplies only `ecological_flow_m3s`, in
WDD/BHT/XLD/XJB/THR order. The requirement applies to total reservoir release,
including turbine flow and spill, on every active day. Zero thresholds are
inactive. `--ecological-flow-scale` scales reference CSV values and records the
effective thresholds and input hash. Ecological compliance is a soft objective;
it does not replace commands or impose a hard release floor.

```text
relative_flow_deficit_i = max(threshold_i - release_i, 0) / threshold_i
local_i = power_i / installed_capacity_i
          - ecological_flow_weight * relative_flow_deficit_i
global = sum(power_i) / sum(installed_capacity_i)
mixed_i = 0.5 * local_i + 0.5 * global
system = 0.5 * sum(local_i) + 0.5 * global
```

The deficit is zero for inactive thresholds and tolerance-filtered for numerical
roundoff. Local penalties use each reservoir's own relative deficit. The global
term measures generation only and enters system accounting once. The legacy
team metric is normalized total generation minus the mean ecological deficit
over the fixed active reservoir set. It remains separate from the learning
reward and from the sum of mixed rewards. `--no-reward-penalties` selects a
generation-only objective while physical ecological counts remain observable.

## MAPPO training

The main settings are in `harl/configs/algos_cfgs/mappo_cascade.yaml` and physical
settings in `harl/configs/envs_cfgs/cascade_reservoir.yaml`. Defaults are six
process-parallel episodes per update, 1,825 daily transitions per episode,
300 episodes per seed, gamma 0.995, GAE lambda 0.95 and five actor/critic epochs.
The default therefore performs 50 updates per seed. Leap days remain in the data;
the fixed horizon need not end on December 31.

MAPPO uses one shared actor and optimizer. Each reservoir provides a local
observation with the same feature layout. The centralized critic receives all
five observations and predicts one mixed-return value per reservoir. Each
reservoir contributes its own GAE channel to the pooled actor batch. By default,
each channel is normalized over all rollout days and parallel lanes.

The actor uses a clipped PPO objective based on the ratio of its current action
probability to the probability saved during collection. Forced actions have zero
policy and entropy weight. The loss takes the mean over all rollout rows.

Actor and critic MLPs use hidden widths `[64, 128, 256, 256]`, ReLU, hidden
LayerNorm, categorical action distributions, Adam and gradient clipping. GAE and
actor advantages use rollout ValueNorm statistics. The normalizer updates once
from full-rollout critic targets after actor training and stays fixed through
critic epochs. Training uses CPU execution.

## Sampling, evaluation and plots

Five source years are held out by default: 1998, 1999, 1970, 1997 and 2006.
Training windows contain training years only. Coverage-balanced sampling reduces
unequal year-day exposure; `--window-sampling uniform` restores uniform draws.
Storage carries across source-year boundaries within each episode.

Periodic evaluation runs on three fixed training-only scenarios before update
one, every configured interval and after the final update. It uses deterministic
policies and fresh environments without updating models, optimizers or RNG state.
Final held-out evaluation uses an ordered representative-year sequence and the
same episode horizon. Physical infeasibility is reported explicitly; failed
windows remain failures, with no completed aggregate substituted from survivors.

MAPPO emits standardized evaluation metrics including system return,
`hydropower_generation_TWh`, ecological violation and compliance rates,
ecological deficit volume, signed per-reservoir and net `storage_change_m3`,
depletion, `final_level_m`, forced-actor rate and infeasible-interval rate.
These fields describe policy performance; they are never substituted into actor
rewards or GAE targets. Completed evaluations also carry the initial, final and
change storage trajectory.

## Generate the four figure categories

Use `--figures-only` with a fresh output stem to train MAPPO and save the PNGs.
The trajectory figure shows the first configured seed.

```bash
run_dir="runs/mappo_$(date +%Y%m%d_%H%M%S)"
conda run --no-capture-output -n harl python -B examples/train_cascade.py \
  --seeds 1 2 3 --episodes 300 \
  --figures-only --plot-output "$run_dir/curves" \
  --plot-python /opt/anaconda3/bin/python
```

### Render saved runs

The plotting commands read completed logs and their held-out daily CSVs using
headless Matplotlib. On the development machine, use Conda `base` for this step,
since it has Matplotlib.

```bash
conda run --no-capture-output -n base python -B examples/plot_cascade.py \
  --episode-log runs/mappo/episodes.jsonl \
  --output-dir runs/ecological_figures --trajectory-seed 1
```

| File | Contents |
| --- | --- |
| `learning.png` | Discounted system return by evaluation round, averaged over the three fixed training windows and shown as mean ± sample SD across seeds. Incomplete rounds leave gaps. |
| `performance.png` | Held-out hydropower generation, ecological violation rate, and cumulative ecological flow deficit for each seed. |
| `trajectories.png` | End-of-day water level, effective permitted bounds, and executed total release for each reservoir, in episode order for the selected seed. |
| `water_allocation.png` | Signed final-minus-initial storage by reservoir, with seed points and the MAPPO mean. |

Use `--categories learning`, `--categories performance`,
`--categories trajectories`, or `--categories water_allocation` for individual
figures. `--trajectory-seed` defaults to the first configured seed. The
trajectory category uses the saved daily CSVs. System return is a performance
metric. Training uses the per-reservoir mixed rewards.

## Daily-change diagnostics

Held-out evaluation writes per-seed daily changes, level-change statistics, and
ramp-conflict CSVs when a run output path is configured. The daily CSV keeps the
original level, release and change columns and adds both effective permitted
level bounds. The first daily difference is empty. Subsequent differences
continue through the source-year joins.

## Diagnostics and replay

The console shows concise per-seed progress by default. Add `--verbose-console`
to print full configuration and JSON diagnostics. This flag changes console
output only. Episode logs, trajectory files and checkpoints retain their full contents.

`--diagnostics` records parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected trajectories. `--trajectory-diagnostics` requires
an episode log and records physical dispatch, ecological deficits, reward
components, frozen values, GAE and policy masks per reservoir/day/lane. Global,
system and legacy scalar rewards repeat across five CSV rows; count each once
per day/lane when reconstructing totals.

```bash
python -B examples/replay_cascade_checkpoint.py \
  --checkpoint runs/EXPERIMENT_seed1_checkpoints/update_000050.pt \
  --source-years 2007 2008 2009 2010 2011 --output runs/EXPERIMENT_replay.csv
```

`--override-action DAY RESERVOIR INDEX` changes one command; later commands react
through the saved policy. Replay preserves the saved horizon, physical and data
hashes, forecast layout and physical dispatch semantics. Checkpoints saved with an
earlier source revision require that revision for replay. It reports generation, ecological
deficits, learning returns, spill and final storage without fitting models.

`analysis/dispatch_diagnostics/` reconstructs frozen trajectories and tests
same-state ecological release alternatives through the full cascade. Grid and
continuous witnesses retain physical bounds and the other sampled commands. These are separate same-day
comparisons, not an executable multi-day intervention or evidence of optimality.

## Source map

| Path | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, configuration and split validation |
| `examples/plot_cascade.py` | Generate figures from MAPPO results |
| `examples/train_cascade_parallel.py` | Seed runs, logging and evaluation |
| `harl/runners/` | Full-episode collection and MAPPO updates |
| `harl/algorithms/`, `harl/models/`, `harl/common/` | PPO, critics, networks, buffers and ValueNorm |
| `harl/envs/cascade_reservoir/data_loader.py` | Historical data, forecasts and ecological inputs |
| `harl/envs/cascade_reservoir/reservoir_physics.py` | Water balance, generation and hard feasibility |
| `harl/envs/cascade_reservoir/level_ramp.py` | Hard daily level-ramp bounds and conflict CSV evidence |
| `harl/envs/cascade_reservoir/reward.py` | Ecological rewards and episode accounting |
| `harl/envs/cascade_reservoir/evaluation.py`, `periodic_evaluation.py` | Evaluation and violation counts |
| `harl/envs/cascade_reservoir/checkpoint.py` | Provenance-checked inference replay |
| `analysis/dispatch_diagnostics/` | Ecological dispatch evidence and reports |
