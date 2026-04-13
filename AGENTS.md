# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a collection of standalone RL algorithm implementations in Python/PyTorch. There are 6 independent scripts — no build system, no services, no test framework.

### Dependencies

- Python 3.12, PyTorch (CPU), gym 0.26.2, matplotlib, numpy <2
- **Critical**: `numpy` must be pinned to `<2` because `gym` 0.26.2 uses `np.bool8` which was removed in numpy 2.x. Scripts 4/5/6 will crash at runtime without this constraint.

### Running scripts

Each script is standalone: `python3 <script>.py`. No build step required.

| Script | Algorithm | Deps |
|---|---|---|
| `1_sarsa_windy_world.py` | SARSA | numpy, matplotlib |
| `2_q_learning_windy_world.py` | Q-Learning | numpy, matplotlib |
| `3_dpn_pytorch.py` | DQN | torch, gym |
| `4_policy_gradient_pytorch.py` | REINFORCE | torch, gym |
| `5_ac_pytorch.py` | Actor-Critic | torch, gym |
| `6_ppo_pytorch.py` | PPO | torch, gym |

### Gotchas

- Scripts 3-5 call `env.render()` in test loops. This produces a harmless warning in headless environments but does not block execution.
- The `gym` deprecation warnings about upgrading to `gymnasium` are expected and harmless — the scripts use the legacy `gym` API.
- There are no automated tests or linting configured in this repository.
