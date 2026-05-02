# Behavior Poisoning in Cooperative Multi-Agent Reinforcement Learning

## Abstract

This project studies behavior poisoning in cooperative multi-agent reinforcement learning (MARL). Instead of corrupting rewards, observations, or offline replay data, the attacker controls one compromised agent's live action selection during training. The goal is to bias the training trajectory distribution so that the final learned team policy converges to a worse coordination behavior even after the attack is removed.

Experiments are conducted in the PettingZoo `simple_spread_v3` task using a shared-policy recurrent MAPPO baseline. Three poisoning settings are evaluated: random action poisoning, targeted action poisoning, and KL-constrained targeted action poisoning. The corrected five-seed comparison trains all clean and poisoned runs for `500000` timesteps. With `p = 0.1`, at least one attack hurts clean performance in all five seeds, but not every attack is consistently harmful. With `p = 0.2`, the average degradation remains small and seed-sensitive. The current evidence supports the implementation and plausibility of behavior poisoning, while showing that stronger statistical claims require more seeds, target-action sweeps, and confidence intervals.

## 1. Introduction

Cooperative MARL systems learn policies from joint interaction data. This makes them vulnerable to attacks that alter the behavioral distribution seen during training. A subtle attacker may not need to change rewards or observations directly; controlling the behavior of a single agent during rollout collection may be enough to steer the whole team toward a poor coordination pattern.

This project investigates whether live action manipulation during training can produce persistent degradation in a cooperative task. The core question is:

> Can an attacker that controls one agent's actions during training make the final learned team policy worse, even after the attacker is removed?

The project focuses on the attack side rather than defense. The main contribution is an implemented MAPPO training pipeline with action poisoning hooks, experiment configs for multiple attack modes, evaluation metrics for coordination quality, and analysis scripts for comparing clean and poisoned checkpoints.

## 2. Environment and Task

The environment is `simple_spread_v3`, a cooperative particle environment from PettingZoo/MPE. Three agents must spread out to cover three landmarks. Agents receive local observations and share a cooperative reward signal.

The task is modeled as a cooperative Markov game:

- Number of agents: `N = 3`
- Number of landmarks: `3`
- Action space: discrete movement actions
- Episode horizon: `25` steps
- Reward: shared team reward based on landmark coverage and coordination quality
- Training algorithm: shared-policy recurrent MAPPO

The desired clean behavior is for agents to distribute themselves across landmarks and maintain good coverage. A poor behavior is a coordination trap where agents repeatedly fail to cover distinct landmarks, remain far from landmarks, or collapse into redundant movement patterns.

## 3. Threat Model

The attacker controls one compromised agent during training rollout collection. The attacker does not:

- change reward values
- change observations
- edit saved replay-buffer entries offline
- modify model parameters directly
- interfere during final evaluation

Instead, the attacker modifies the compromised agent's sampled action during training. The attacked experience is then used by MAPPO as if it came from normal interaction. At evaluation time, the attack is disabled, so the reported performance measures the policy learned from poisoned training rather than live sabotage.

Three attack modes are implemented:

1. Random action poisoning: with probability `p`, replace the compromised agent's action with a uniformly random legal action.
2. Targeted action poisoning: with probability `p`, force the compromised agent to a fixed target action.
3. KL-constrained targeted poisoning: use targeted action poisoning while limiting intervention probability so the attacked action distribution remains within a KL-divergence budget.

The default compromised agent is agent `0`, the default intervention probability is `0.1`, and the targeted action is action `0`, the no-op action in the default discrete MPE action mapping.

## 4. Hypotheses

H1: Clean shared-policy recurrent MAPPO learns useful cooperative behavior in `simple_spread_v3`.

H2: Random action poisoning degrades final policy quality, but the degradation is noisy and unstructured.

H3: Targeted action poisoning degrades final policy quality more than random action poisoning at the same intervention budget.

H4: KL-constrained targeted poisoning can still degrade performance while staying closer to the clean action distribution.

H5: The poisoned behavior persists after the attacker is removed, meaning evaluation with clean execution remains worse than the clean baseline.

## 5. Implementation

The implementation extends the clean MAPPO pipeline with action poisoning during rollout collection. The key code paths are:

- `src/behavior_poisoning/config.py`: experiment, training, environment, and attack configuration dataclasses.
- `src/behavior_poisoning/mappo_clean.py`: MAPPO training/evaluation, action poisoning logic, KL-constrained intervention, and coordination metrics.
- `src/behavior_poisoning/analysis.py`: comparison tables, plots, and persistence evaluation.
- `configs/random_action_poisoning_mappo.yaml`: random action attack config.
- `configs/targeted_action_poisoning_mappo.yaml`: targeted action attack config.
- `configs/kl_constrained_targeted_action_mappo.yaml`: KL-constrained targeted attack config.

During MAPPO rollout collection, the actor first samples clean actions. The poisoning module then optionally replaces the compromised agent's action before the environment step. For targeted attacks, the replacement is a fixed action. For random attacks, the replacement is uniformly sampled. For KL-constrained attacks, the effective intervention probability is reduced when needed so that the mixture of the clean action distribution and attack distribution satisfies the configured KL budget.

Evaluation always runs with the attack disabled. This design directly tests whether poisoned training changed the learned policy itself.

## 6. Experimental Setup

All main experiments use the same environment and MAPPO training configuration:

- Environment: `simple_spread_v3`
- Agents: `3`
- Landmarks: `3`
- Episode length: `25`
- Algorithm: shared-policy recurrent MAPPO (`rmappo`)
- Training timesteps: `500000`
- Evaluation episodes: `32`
- Seeds: `7`, `13`, `21`, `31`, `37`
- Rollout threads: `16`
- Evaluation rollout threads: `4`
- Hidden size: `64`
- Recurrent layers: `1`
- Learning rate: `0.0007`
- Discount factor: `0.99`
- GAE lambda: `0.95`

Four checkpoints are compared:

1. Clean RMAPPO
2. Random action poisoned RMAPPO
3. Targeted action poisoned RMAPPO
4. KL-constrained targeted action poisoned RMAPPO

Primary metrics:

- Mean team reward
- Reward degradation relative to clean
- Final landmark min-distance sum
- Final unique landmarks covered
- Maximum unique landmarks reached during the episode
- Collision pair events per episode
- Collision step rate

All reported evaluations disable the attack, so they measure the learned policy after poisoned training rather than live test-time sabotage.

## 7. Results

### 7.1 Seed-7 Check

The seed-7 `p = 0.1` run trains all four conditions for `500000` timesteps and evaluates each checkpoint cleanly for 32 episodes:

| Run | Mean Team Reward | Degradation vs Clean | Poisoned Actions | Effective Probability |
|---|---:|---:|---:|---:|
| Clean RMAPPO | -488.44 | 0.00 | - | - |
| Random Action | -501.91 | 13.47 | 49984 | 0.100 |
| Targeted Action | -512.80 | 24.36 | 49956 | 0.100 |
| KL Targeted Action | -501.94 | 13.50 | 28670 | 0.057 |

This result shows poisoning harm for seed `7`, with targeted action poisoning producing the largest degradation in this seed.

### 7.2 p = 0.1 Sweep

The `p = 0.1` evidence combines five seeds: seed `7`, correctly trained seeds `13` and `21`, and 32-episode seeds `31` and `37`. For seeds `13` and `21`, the comparison uses the first 32 clean-evaluation episodes from their existing 100-episode summaries so the evaluation horizon matches the other rows.

| Seed | Random Degradation | Targeted Degradation | KL Degradation | Interpretation |
|---:|---:|---:|---:|---|
| 7 | 13.47 | 24.36 | 13.50 | all attacks hurt |
| 13 | 9.46 | 1.25 | 13.85 | all attacks hurt |
| 21 | 22.46 | 14.74 | 3.85 | all attacks hurt |
| 31 | 2.59 | -28.71 | -24.00 | only random hurts |
| 37 | -0.83 | -12.69 | 15.70 | only KL hurts |

Across these five seeds, at least one attack hurts clean performance in `5/5` seeds, all three attacks hurt in `3/5` seeds, random poisoning hurts in `4/5` seeds, targeted poisoning hurts in `3/5` seeds, and KL-constrained poisoning hurts in `4/5` seeds. Targeted poisoning is worse than random poisoning in only `1/5` seeds.

Average degradation at `p = 0.1` is:

| Attack | Mean Degradation vs Clean |
|---|---:|
| Random Action | 9.43 |
| Targeted Action | -0.21 |
| KL Targeted Action | 4.58 |

### 7.3 p = 0.2 Sweep and Budget Comparison

The `p = 0.2` sweep repeats the same five seeds with the unconstrained attack probability doubled. Random and targeted attacks perform roughly twice as many interventions, while the KL-constrained attack is throttled to an effective probability around `0.07` to `0.08`.

| Seed | Random Degradation | Targeted Degradation | KL Degradation | Interpretation |
|---:|---:|---:|---:|---|
| 7 | 31.40 | 30.31 | 9.44 | all attacks hurt |
| 13 | -12.14 | -11.24 | 7.69 | only KL hurts |
| 21 | 7.54 | 10.39 | 9.70 | all attacks hurt |
| 31 | -9.70 | -33.60 | -5.65 | no attack hurts |
| 37 | -13.83 | 12.44 | -4.23 | only targeted hurts |

Across five seeds, at least one attack hurts in `4/5` seeds, all three attacks hurt in `2/5` seeds, random poisoning hurts in `2/5` seeds, targeted poisoning hurts in `3/5` seeds, and KL-constrained poisoning hurts in `3/5` seeds. Targeted poisoning is worse than random poisoning in `3/5` seeds.

Average degradation at `p = 0.2` is:

| Attack | Mean Degradation vs Clean |
|---|---:|
| Random Action | 0.65 |
| Targeted Action | 1.66 |
| KL Targeted Action | 3.39 |

The comparison does not show monotonic damage from increasing `p`. Doubling the attack probability makes some seeds worse, especially seed `7`, but improves or stabilizes others. This is plausible in MAPPO because training is stochastic and because action noise can sometimes behave like exploration or regularization. It also means the current target action, `0`/no-op, is not a consistently damaging target.

Generated corrected comparison outputs:

- `probability_comparison_results/p01_corrected_records.csv`
- `probability_comparison_results/p02_records.csv`
- `probability_comparison_results/p01_vs_p02_by_seed.csv`
- `probability_comparison_results/probability_summary.csv`
- `probability_comparison_results/collision_summary.csv`
- `probability_comparison_results/collision_by_seed.csv`

### 7.4 Collision Metric

The environment already penalizes agent-agent collisions in the reward. The refreshed evaluation summaries now expose this directly as `collision_pair_events_mean`, the average number of pairwise agent collisions per episode. With three agents there are three possible collision pairs at each step: `(0, 1)`, `(0, 2)`, and `(1, 2)`.

| p | Run | Mean Collision Events | Delta vs Clean | Collision Step Rate |
|---:|---|---:|---:|---:|
| 0.1 | Clean RMAPPO | 2.03 | 0.00 | 0.078 |
| 0.1 | Random Action | 1.75 | -0.28 | 0.069 |
| 0.1 | Targeted Action | 1.49 | -0.53 | 0.059 |
| 0.1 | KL Targeted Action | 1.74 | -0.28 | 0.069 |
| 0.2 | Clean RMAPPO | 2.03 | 0.00 | 0.078 |
| 0.2 | Random Action | 2.07 | 0.04 | 0.080 |
| 0.2 | Targeted Action | 2.09 | 0.07 | 0.082 |
| 0.2 | KL Targeted Action | 2.39 | 0.37 | 0.093 |

The collision metric shows that reward degradation is not only a collision story. At `p = 0.1`, poisoned policies are worse in reward while producing fewer collisions on average, so their degradation mainly comes from poorer landmark coverage or coordination. At `p = 0.2`, collision events increase slightly for poisoned policies, especially KL-constrained poisoning, so collisions may contribute to degradation there.

## 8. Qualitative Visualization Notes

Rendered single episodes can be misleading. A poisoned checkpoint may occasionally look visually better than the clean checkpoint because individual rollouts have high variance. Visual judgment also does not always match the reward metric: a policy can appear active or spread out while still ending farther from landmarks or failing to maintain coverage throughout the episode.

Therefore, visual rollouts are useful for intuition and demonstration, but the report should rely on aggregate reward and distance metrics for claims.

## 9. Discussion

The corrected experiments show that action-level intervention during training can damage cooperative learning in some seeds and attack modes. This is important because the attack does not require changing rewards, observations, or model parameters. It only changes one agent's behavior during data collection.

The strongest corrected evidence is not that every poisoning method always hurts, but that the training pipeline is sensitive to action-level intervention. At `p = 0.1`, at least one attack hurts clean performance in every seed, and all three attacks hurt in three of five seeds. At `p = 0.2`, the effect remains mixed: seed `7` and seed `21` clearly degrade, but seed `31` improves under all three attacks.

Since the attack is disabled during evaluation, poor performance reflects the learned policy rather than immediate action sabotage. This supports the interpretation that poisoned training can move the system toward a worse coordination pattern when the attack and seed interact unfavorably.

The seed sweep adds an important caveat: the poisoning effect is sensitive to seed, target action, and probability. This makes the current project a working demonstration and measurement harness, not yet a statistically complete claim about which attack is strongest.

## 10. Limitations

The current results should be interpreted as initial evidence rather than a final statistical conclusion.

Main limitations:

- The clean RMAPPO baseline is good but not perfect.
- The comparison uses five seeds, which is still too small for strong confidence intervals.
- The attack target is fixed to action `0`; this no-op target is not consistently damaging and sometimes appears to help.
- The relationship between poison probability and degradation is not monotonic in the current runs.
- The project does not implement a defense or detection method.

Future work should extend the seed sweep, test additional target actions and intervention probabilities, and report confidence intervals.

## 11. Reproducibility

Set up the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/marlbenchmark/on-policy .mappo_upstream
```

Run clean training:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_clean.py --config .\configs\clean_mappo.yaml
```

Run random action poisoning:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_clean.py --config .\configs\random_action_poisoning_mappo.yaml
```

Run targeted action poisoning:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_clean.py --config .\configs\targeted_action_poisoning_mappo.yaml
```

Run KL-constrained targeted poisoning:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_clean.py --config .\configs\kl_constrained_targeted_action_mappo.yaml
```

Run the current seed-sweep helper with a probability override:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_training_seed_sweep.py --seeds 7 13 21 31 37 --evaluation-episodes 32 --attack-probability 0.1 --output-dir .\seed_sweep_p01_results
```

Run the `p = 0.2` comparison:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_training_seed_sweep.py --seeds 7 13 21 31 37 --evaluation-episodes 32 --attack-probability 0.2 --output-dir .\seed_sweep_p02_results
```

Render a clean policy:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_clean.py --config .\seed_sweep_p01_results\generated_configs\clean_mappo_seed7.yaml --model-path .\seed_sweep_p01_results\checkpoints\clean_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

Render a poisoned checkpoint with attack disabled:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_clean.py --config .\seed_sweep_p02_results\generated_configs\kl_constrained_targeted_action_mappo_seed7.yaml --model-path .\seed_sweep_p02_results\checkpoints\kl_targeted_action_poison_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q
```

## 12. Conclusion

This project demonstrates that behavior poisoning through live action manipulation can degrade cooperative MARL training, but the results are nuanced and seed-sensitive.

The current evidence supports a careful version of the central claim: poisoning the training behavior of a single agent can bias cooperative learning toward lower-quality coordination outcomes, but the default attack settings are not robust across seeds or probabilities. The strongest next step is to broaden the seed sweep and target-action sweep, then report confidence intervals. The implemented system and corrected results provide a reproducible harness for that next pass.
