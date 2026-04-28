# Behavior Poisoning in Cooperative Multi-Agent Reinforcement Learning

## Abstract

This project studies behavior poisoning in cooperative multi-agent reinforcement learning (MARL). Instead of corrupting rewards, observations, or offline replay data, the attacker controls one compromised agent's live action selection during training. The goal is to bias the training trajectory distribution so that the final learned team policy converges to a worse coordination behavior even after the attack is removed.

Experiments are conducted in the PettingZoo `simple_spread_v3` task using a shared-policy recurrent MAPPO baseline. Three poisoning settings are evaluated: random action poisoning, targeted action poisoning, and KL-constrained targeted action poisoning. The results show that all poisoning variants reduce final team performance relative to clean RMAPPO, and the degradation persists when the saved poisoned policies are evaluated with the attack disabled. In the current single-seed experiment, clean RMAPPO achieves a mean team reward of `-488.44`, while random, targeted, and KL-constrained targeted poisoning achieve `-655.80`, `-668.94`, and `-732.61`, respectively. Persistence evaluation over 8 clean episodes further shows that poisoned checkpoints remain substantially worse than the clean checkpoint.

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
- Seed: `7`
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

The persistence evaluation reloads each saved checkpoint, disables the attack, and evaluates the learned policy cleanly for 8 episodes.

## 7. Results

### 7.1 Main Evaluation

The main 32-episode evaluation shows clear degradation from all poisoning variants.

| Run | Mean Team Reward | Reward Std | Degradation vs Clean | Final Min-Distance | Final Unique Landmarks | Max Unique Landmarks |
|---|---:|---:|---:|---:|---:|---:|
| Clean RMAPPO | -488.44 | 71.03 | 0.00 | 0.96 | 2.44 | 2.75 |
| Random Action | -655.80 | 159.81 | 167.35 | 2.25 | 1.97 | 2.34 |
| Targeted Action | -668.94 | 146.04 | 180.50 | 2.43 | 1.88 | 2.31 |
| KL Targeted Action | -732.61 | 183.91 | 244.17 | 3.10 | 1.88 | 2.44 |

Clean RMAPPO performs best under all primary metrics. The poisoned policies have lower mean reward, higher final landmark distance, and lower final unique landmark coverage. This supports H2 and H3: poisoning one agent during training can degrade the final learned team behavior, and the targeted attack is worse than the random action baseline in this run.

The KL-constrained targeted attack shows the largest degradation in this single-seed experiment. This supports H4, although additional seeds would be needed before making a strong general claim about KL-constrained attacks being consistently strongest.

Generated plots:

- `results/analysis/team_reward_mean.png`
- `results/analysis/reward_degradation_vs_clean.png`
- `results/analysis/final_min_distance_sum_mean.png`

### 7.2 Persistence Evaluation

The persistence evaluation disables the attack and evaluates the saved checkpoints cleanly for 8 episodes. The poisoned checkpoints remain worse than the clean checkpoint.

| Run | Mean Team Reward | Reward Std | Degradation vs Clean | Final Min-Distance | Final Unique Landmarks | Max Unique Landmarks |
|---|---:|---:|---:|---:|---:|---:|
| Clean RMAPPO | -485.37 | 41.67 | 0.00 | 0.82 | 2.63 | 3.00 |
| Random Action | -703.29 | 215.78 | 217.92 | 2.46 | 1.88 | 2.38 |
| Targeted Action | -708.71 | 134.25 | 223.34 | 2.39 | 1.50 | 2.25 |
| KL Targeted Action | -815.79 | 332.61 | 330.42 | 3.35 | 2.00 | 2.25 |

This supports H5. The poisoned policies do not recover simply because the attack is removed at evaluation time. Instead, the training process appears to have learned weaker coordination behavior from poisoned rollout data.

Generated persistence plots:

- `results/analysis/persistence/team_reward_mean.png`
- `results/analysis/persistence/reward_degradation_vs_clean.png`
- `results/analysis/persistence/final_min_distance_sum_mean.png`

## 8. Qualitative Visualization Notes

Rendered single episodes can be misleading. A poisoned checkpoint may occasionally look visually better than the clean checkpoint because individual rollouts have high variance. For example, random action poisoning has a much worse 32-episode mean reward than clean RMAPPO, but some individual poisoned episodes still achieve reasonable behavior. Visual judgment also does not always match the reward metric: a policy can appear active or spread out while still ending farther from landmarks or failing to maintain coverage throughout the episode.

Therefore, visual rollouts are useful for intuition and demonstration, but the report should rely on aggregate reward and distance metrics for claims.

## 9. Discussion

The experiments show that action-level intervention during training is enough to damage cooperative learning. This is important because the attack does not require changing rewards, observations, or model parameters. It only changes one agent's behavior during data collection.

Random action poisoning degrades performance by injecting unstructured noise. Targeted action poisoning performs slightly worse than random poisoning in the main evaluation, suggesting that consistent behavioral bias can be more damaging than random disruption. The KL-constrained targeted variant is strongest in the current run, showing that a constrained attack can still produce meaningful damage.

The persistence results are especially important. Since the attack is disabled during persistence evaluation, poor performance reflects the learned policy rather than immediate action sabotage. This supports the interpretation that poisoned training can move the system toward a worse coordination pattern.

## 10. Limitations

The current results should be interpreted as strong initial evidence rather than a final statistical conclusion.

Main limitations:

- Experiments are currently reported for a single seed.
- The clean RMAPPO baseline is good but not perfect.
- Persistence evaluation uses 8 episodes, which is useful but still small.
- Random and targeted attack summaries do not include logged intervention counts in the current saved summaries, likely because those checkpoints were produced before the newest attack-stat logging was added.
- The KL-constrained result is strongest in this run, but more seeds are needed to determine whether that ordering is stable.
- The project does not implement a defense or detection method.

Future work should run multiple seeds, retrain all attacks with current logging, and test additional target actions and intervention probabilities.

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

Generate comparison and persistence analysis:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_experiments.py --refresh-persistence --persistence-episodes 8
```

Render a clean policy:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_clean.py --config .\configs\clean_mappo.yaml --model-path .\results\checkpoints\clean_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

Render a poisoned checkpoint with attack disabled:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_clean.py --config .\configs\kl_constrained_targeted_action_mappo.yaml --model-path .\results\checkpoints\kl_targeted_action_poison_rmappo_seed7 --episodes 1 --render --step-delay 0.12
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q
```

## 12. Conclusion

This project demonstrates that behavior poisoning through live action manipulation can degrade cooperative MARL training. In `simple_spread_v3`, controlling one agent's action during MAPPO rollout collection causes the final learned policy to perform worse under clean evaluation. The degradation appears in both reward and coordination metrics, and it persists after the attack is removed.

The current evidence supports the central claim: poisoning the training behavior of a single agent can bias cooperative learning toward lower-quality coordination outcomes. The strongest next step is to rerun the clean and poisoned experiments across multiple seeds and include confidence intervals, but the implemented system and current results already provide a complete working demonstration of behavior poisoning in cooperative MARL.
