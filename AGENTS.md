# AGENTS.md

## Cursor Cloud specific instructions

This repository is a collection of standalone reinforcement learning scripts using PyTorch. There is no build system, test framework, or services architecture — each `.py` file is an independent, runnable script.

### Dependencies

- **Python 3.12** (system default)
- **numpy <2.0** — gym 0.26.2 uses `np.bool8` which was removed in NumPy 2.0; pinning `numpy<2.0` avoids `AttributeError`
- **matplotlib** — used by scripts 1 & 2 (Agg backend, no display needed)
- **torch** (PyTorch, CPU) — used by scripts 3–6
- **gym 0.26.2** (OpenAI Gym) — used by scripts 3–6 for CartPole-v0

### Running scripts

Each script is self-contained. Run with `python3 <script>.py`:

| Script | Algorithm | Environment | Approx. runtime |
|---|---|---|---|
| `1_sarsa_windy_world.py` | SARSA | Windy Gridworld | ~1s |
| `2_q_learning_windy_world.py` | Q-Learning | Windy Gridworld | ~1s |
| `3_dpn_pytorch.py` | DQN | CartPole-v0 (Gym) | ~1s |
| `4_policy_gradient_pytorch.py` | REINFORCE | CartPole-v0 (Gym) | ~1s |
| `5_ac_pytorch.py` | Actor-Critic | CartPole-v0 (Gym) | ~2s |
| `6_ppo_pytorch.py` | PPO | CartPole-v0 (Gym) | ~2s |

### Known warnings (safe to ignore)

- Gym deprecation warning about NumPy 2.0 — this is a warning from the gym library itself; the scripts work fine with numpy <2.0.
- `np.bool8` deprecation warning — appears in some gym wrappers; harmless with numpy 1.26.x.
- CartPole-v0 version warning — gym suggests upgrading to v1; the scripts use v0 intentionally.
- `render_mode` warning — gym suggests specifying render mode at init; the scripts call `env.render()` directly.

### Lint / Test / Build

- **No linter** is configured in this repo.
- **No test framework** is configured (no pytest, unittest, etc.).
- **No build step** — scripts are run directly with the Python interpreter.
