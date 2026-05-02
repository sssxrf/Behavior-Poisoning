# Behavior Poisoning in Cooperative MARL

## Project Scope

This project focuses on the attack side of the problem, not defense. The immediate goal is to show that a subtle attacker can manipulate the live sampling process in a cooperative multi-agent reinforcement learning setting and steer training toward a low-quality coordination outcome.

The working environment is `PettingZoo simple_spread_v3`, where three agents must spread out and cover three landmarks efficiently. Clean training should learn cooperative coverage. Our attack-stage extensions will later test whether poisoning one agent's action choices during training can bias the population toward a persistent coordination trap.

## Problem Model

### Environment

We model the task as a cooperative Markov game:

- Agents: `N = 3`
- State space: environment state induced by all agent and landmark positions and velocities
- Observation space: each agent receives its local observation from `simple_spread_v3`
- Action space: discrete movement actions
- Reward structure: cooperative reward, where agents are encouraged to cover landmarks while avoiding unnecessary collisions
- Episode horizon: fixed by `max_cycles`

The clean-learning objective is to learn a policy that maximizes expected team return:

`J(pi) = E[sum_t R_team(s_t, a_t)]`

where all agents follow the same training procedure and the team reward reflects coordination quality.

### Clean Baseline Learning Setting

For the first milestone, we use a shared-policy recurrent MAPPO baseline:

- one neural policy is shared across all agents
- each agent still acts on its own observation
- the policy is trained on trajectories sampled from the `simple_spread` environment
- recurrent hidden state helps the baseline match the official MAPPO training setup we are building on

This baseline gives us a stable and reproducible cooperative learner before adding poisoning logic. A legacy PPO path is kept only as a fallback comparison, not as the primary baseline of record.

### Threat Model for Later Attack Stages

The attacker does not edit replay-buffer entries directly and does not change rewards, observations, or labels offline.

Instead, the attacker:

- controls the action selection of one compromised agent during training
- injects behavior only during the live sampling phase
- aims to bias the training distribution while remaining close to normal behavior

The attack will later be constrained by a stealth budget such as KL divergence:

`D_KL(pi_attack || pi_clean) <= epsilon`

This turns the problem into a stealth-versus-damage tradeoff:

- low divergence means the attack is harder to detect
- sufficient bias is still needed to distort the team's convergence target

### Desired Failure Mode

The main failure mode is a coordination trap:

- training converges to a stable but poor cooperative pattern
- agents receive lower total reward than the clean baseline
- the degraded behavior is not just random noise, but a repeatable coordination outcome

For now, we will use the phrase coordination trap instead of formally claiming a bad Nash equilibrium unless we later add a stronger equilibrium check.

## Research Questions

1. Can clean shared-policy recurrent MAPPO learn cooperative landmark coverage reliably in `simple_spread_v3`?
2. Can live action manipulation during sampling reduce final team performance more effectively than a naive random-action baseline?
3. Can a poisoning policy remain close to normal exploration behavior while still pushing the team toward a low-reward coordination pattern?
4. Does the degraded coordination persist after the attack is removed?

## Hypotheses

### H1: Clean Coordination

Under clean training, shared-policy recurrent MAPPO will learn a stable cooperative behavior with:

- higher average team return over time
- lower collision frequency than early training
- better landmark coverage than an untrained policy

### H2: Random Poisoning Is Harmful but Unstructured

If the compromised agent is forced to take random actions with probability `p` during sampling, training performance will degrade, but the effect will be noisy and less consistent than a targeted behavioral attack.

### H3: Behavioral Poisoning Beats Random Poisoning at the Same Budget

At the same intervention budget `p`, a targeted poisoning policy will produce lower final team return than random poisoning because it pushes the system toward a specific bad coordination pattern rather than injecting unstructured noise.

### H4: Stealth-Constrained Attacks Can Still Succeed

Even when constrained by a small KL-divergence budget `epsilon`, a behavioral poisoning policy can still reduce final team performance enough to demonstrate a meaningful stealth-impact tradeoff.

### H5: Persistence After Attack Removal

Once the population has adapted to poisoned trajectories, removing the attacker will not immediately restore clean behavior. Performance will remain below the clean baseline for at least several evaluation episodes.

## Variables and Measurements

### Controlled Variables

- environment: `simple_spread_v3`
- number of agents: `3`
- training algorithm: shared-policy recurrent MAPPO
- seed list
- episode horizon
- attack budget `p`
- stealth budget `epsilon`

### Primary Metrics

- average team reward per episode
- average episode length
- reward trend during training
- final evaluation reward after training
- collision pair events per episode
- collision step rate

### Secondary Metrics

- landmark coverage or occupancy quality
- KL divergence between clean and attacked action distributions
- persistence score after the attack is disabled

## Milestone 1 Deliverables

The first working milestone in this repo is:

1. a documented problem model and hypothesis file
2. a reproducible Python environment setup
3. a clean-training scaffold for `simple_spread_v3`
4. a trainable and evaluable shared-policy recurrent MAPPO baseline

Once this baseline is stable, we can add:

1. full-length comparison runs across clean, random-poisoned, targeted-poisoned, and KL-constrained training
2. report corrected probability-sweep comparisons from `seed_sweep_p01_results`, `seed_sweep_p02_results`, and `probability_comparison_results`
