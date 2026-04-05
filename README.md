# Behavior Poisoning

This repo is the working starter scaffold for the CS763 final project on behavioral poisoning in cooperative MARL.

The current milestone covers:

- a formal problem model and hypotheses document
- a reproducible Python environment setup
- a clean recurrent MAPPO baseline on `simple_spread`

## Project Layout

```text
configs/                  YAML experiment configs
scripts/                  runnable entry-point scripts
src/behavior_poisoning/   training and evaluation code
results/                  checkpoints, logs, and evaluation summaries
problem_model_and_hypotheses.md
```

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run Clean Training

Quick smoke run:

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\clean_mappo.yaml --total-timesteps 200000
```

Longer baseline run:

```powershell
.\.venv\Scripts\python .\scripts\train_clean.py --config .\configs\clean_mappo.yaml
```

## Evaluate a Saved Model

```powershell
.\.venv\Scripts\python .\scripts\evaluate_clean.py --config .\configs\clean_mappo.yaml --model-path .\results\checkpoints\clean_rmappo_seed7
```

## Watch the Agents Move

```powershell
.\.venv\Scripts\python .\scripts\evaluate_clean.py --config .\configs\clean_mappo.yaml --model-path .\results\checkpoints\clean_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

## Next Extensions

Once the clean baseline is stable, the next code additions should be:

1. random action poisoning during sampling
2. targeted behavioral poisoning for one compromised agent
3. KL-constrained attack logic
4. comparison plots and persistence analysis
