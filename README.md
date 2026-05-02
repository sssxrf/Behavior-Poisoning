# Behavior Poisoning

This repo is the working starter scaffold for the CS763 final project on behavioral poisoning in cooperative MARL.

The current milestone covers:

- a formal problem model and hypotheses document
- a reproducible Python environment setup
- a clean recurrent MAPPO baseline on `simple_spread`
- a legacy PPO fallback config for quick comparisons

## Project Layout

```text
configs/                  YAML experiment configs
scripts/                  runnable entry-point scripts
src/behavior_poisoning/   training and evaluation code
.mappo_upstream/          local clone of marlbenchmark/on-policy for MAPPO/RMAPPO runs
results/                  checkpoints, logs, and evaluation summaries
problem_model_and_hypotheses.md
```

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
git clone https://github.com/marlbenchmark/on-policy .mappo_upstream
```

`clean_mappo.yaml` and `clean_mappo_separated.yaml` require the local `.mappo_upstream` clone above.
`clean_ppo.yaml` does not require it and is kept as a fallback comparison path.

## Run Clean Training

The current baseline of record is shared recurrent MAPPO (`clean_mappo.yaml`).

Quick smoke run:

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\clean_mappo.yaml --total-timesteps 200000
```

Longer baseline run:

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\clean_mappo.yaml
```

Optional PPO fallback run:

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\clean_ppo.yaml
```

## Run Random Action Poisoning

The first attack baseline replaces one compromised agent's sampled action with a
uniformly random legal action during MAPPO training rollouts. Evaluation remains
clean, so the reported checkpoint performance measures the learned policy after
the attacker is removed.

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\random_action_poisoning_mappo.yaml --total-timesteps 200000
```

Tune the intervention budget in the config:

```yaml
attack:
  enabled: true
  mode: random_action
  compromised_agent: 0
  probability: 0.1
```

## Run Targeted Action Poisoning

The first targeted attack baseline forces the compromised agent toward one fixed
action with probability `p`. In the default MPE discrete action mapping,
`target_action: 0` is the no-op action, which tests whether selectively freezing
one agent during sampling can induce a worse learned coordination pattern than
unstructured random action noise.

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\targeted_action_poisoning_mappo.yaml --total-timesteps 200000
```

Tune the target in the config:

```yaml
attack:
  enabled: true
  mode: targeted_action
  compromised_agent: 0
  probability: 0.1
  target_action: 0
```

## Run KL-Constrained Targeted Poisoning

The KL-constrained variant uses the same targeted action override, but caps the
per-state intervention probability so the attacked action distribution stays
within `D_KL(pi_attack || pi_clean) <= kl_budget`.

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\kl_constrained_targeted_action_mappo.yaml --total-timesteps 200000
```

Tune the stealth budget in the config:

```yaml
attack:
  enabled: true
  mode: targeted_action
  compromised_agent: 0
  probability: 0.1
  target_action: 0
  kl_budget: 0.02
```

## Evaluate a Saved Model

```powershell
.\.venv\Scripts\python .\scripts\evaluate_clean.py --config .\seed_sweep_p01_results\generated_configs\clean_mappo_seed7.yaml --model-path .\seed_sweep_p01_results\checkpoints\clean_rmappo_seed7
```

## Smoke Checks

Fast config-routing regression check:

```powershell
.\.venv\Scripts\python -m pytest .\tests -q
```

Short saved-checkpoint smoke check:

```powershell
.\.venv\Scripts\python .\scripts\smoke_check_saved_models.py --episodes 2
```

## Run Seed Sweeps

The current repo includes a seed-sweep helper that retrains the clean and three
poisoned RMAPPO configs, writes resumable partial results, and records attack
intervention counts for the poisoned runs:

```powershell
.\.venv\Scripts\python .\scripts\run_training_seed_sweep.py --seeds 13 21 --evaluation-episodes 100
```

Override the attack probability for all poisoned configs without editing the base YAML files:

```powershell
.\.venv\Scripts\python .\scripts\run_training_seed_sweep.py --seeds 7 13 21 31 37 --evaluation-episodes 32 --attack-probability 0.2 --output-dir .\seed_sweep_p02_results
```

Outputs are written under the selected `--output-dir`, including:

- `seed_sweep_summary.json`
- `seed_sweep_records.csv`
- `seed_sweep_aggregates.csv`

Refresh saved checkpoints without retraining when evaluator metrics change:

```powershell
.\.venv\Scripts\python .\scripts\refresh_seed_sweep_metrics.py .\seed_sweep_p01_results .\seed_sweep_p02_results --episodes 32
```

Use the corrected seed-sweep outputs:

- `seed_sweep_p01_results\`
- `seed_sweep_p02_results\`
- `probability_comparison_results\`

The current evaluation summaries include reward, landmark coverage, and collision metrics. `collision_pair_events_mean` is the average number of pairwise agent-agent collisions per episode, and `collision_step_rate_mean` is the fraction of evaluation steps with at least one collision.

## Watch the Agents Move

```powershell
.\.venv\Scripts\python .\scripts\evaluate_clean.py --config .\seed_sweep_p02_results\generated_configs\clean_mappo_seed7.yaml --model-path .\seed_sweep_p02_results\checkpoints\clean_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

## Next Extensions

The corrected five-seed comparison shows seed-sensitive poisoning effects. The next useful extensions are:

1. extend the seed sweep enough to report confidence intervals
2. sweep target actions instead of only using no-op action `0`
3. test intervention probabilities beyond `0.1` and `0.2`
