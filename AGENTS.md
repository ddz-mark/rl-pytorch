# AGENTS.md

## Cursor Cloud specific instructions

This is a collection of standalone RL algorithm scripts (no build system, no web app, no services). Each `.py` file runs independently.

### Dependencies

- **Python 3.12** (system default)
- **PyTorch** (CPU-only, installed from `https://download.pytorch.org/whl/cpu`)
- **gym 0.26.2** (legacy OpenAI Gym — the code uses `import gym`, not `gymnasium`)
- **numpy <2** — gym 0.26.2 is incompatible with NumPy 2.x (`np.bool8` was removed). Must pin `numpy<2`.
- **matplotlib** — used in scripts 1–2 for plotting (currently commented out, but imports are active with `Agg` backend)

### Running scripts

Each script is self-contained. Run with `python3 <script>.py`. No build step, no linter, no test framework.

| Script | Algorithm | Environment | Runtime |
|---|---|---|---|
| `1_sarsa_windy_world.py` | SARSA | Windy Gridworld (custom) | ~1s |
| `2_q_learning_windy_world.py` | Q-Learning | Windy Gridworld (custom) | ~1s |
| `3_dpn_pytorch.py` | DQN | CartPole-v0 (gym) | ~1s |
| `4_policy_gradient_pytorch.py` | REINFORCE | CartPole-v0 (gym) | ~1s |
| `5_ac_pytorch.py` | Actor-Critic | CartPole-v0 (gym) | ~1s |
| `6_ppo_pytorch.py` | PPO | CartPole-v0 (gym) | ~2s |

### Gotchas

- The `gym[classic_control]` extra requires `pygame` which needs SDL2 system libraries. However, the scripts work fine with plain `pip install gym` since CartPole does not require pygame for non-rendered runs (the scripts use `env.unwrapped` or have rendering disabled).
- Script 3 calls `env.render()` during test episodes, which logs a warning about missing render_mode but does not fail.
- There is no `requirements.txt` in the repo. Dependencies are inferred from imports.
