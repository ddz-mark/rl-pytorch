# AGENTS.md

## Cursor Cloud specific instructions

This repository contains 6 standalone Python RL scripts (SARSA, Q-Learning, DQN, Policy Gradient, Actor-Critic, PPO). There are no services, databases, or build steps.

**Running scripts:** Each script is independent and can be run with `python3 <script>.py` from the repo root.

| Script | Algorithm | Notes |
|---|---|---|
| `1_sarsa_windy_world.py` | SARSA | Pure numpy + matplotlib, no gym |
| `2_q_learning_windy_world.py` | Q-Learning | Pure numpy + matplotlib, no gym |
| `3_dpn_pytorch.py` | DQN | Uses gym `CartPole-v0` + PyTorch |
| `4_policy_gradient_pytorch.py` | Policy Gradient | Uses gym `CartPole-v0` + PyTorch |
| `5_ac_pytorch.py` | Actor-Critic | Uses gym `CartPole-v0` + PyTorch |
| `6_ppo_pytorch.py` | PPO | Uses gym `CartPole-v0` + PyTorch |

**Key gotcha — NumPy version:** `gym==0.26.2` is incompatible with NumPy 2.x (`np.bool8` removed). The `requirements.txt` pins `numpy<2` to avoid this. Do not upgrade NumPy past 1.x while using `gym`.

**No linter/test framework:** The repo has no linter config, no test suite, and no CI. Running the scripts themselves is the primary validation method.

**PyTorch CPU-only:** Scripts auto-detect CUDA but fall back to CPU. The cloud VM has no GPU; all scripts run fine on CPU.

**Matplotlib backend:** Scripts 1 and 2 set `matplotlib.use('Agg')` (non-interactive), so no display server is needed.
