## Cursor Cloud specific instructions

This is an educational RL (Reinforcement Learning) repository with 6 standalone Python scripts. There are no services, databases, APIs, or build systems — each script runs independently.

### Dependencies

`torch`, `gym` (0.26.x), `numpy` (<2.0), `matplotlib`. The `gym` package requires `numpy<2` due to the deprecated `np.bool8` alias.

### Running scripts

Each script is standalone:

```
python3 1_sarsa_windy_world.py       # SARSA on Windy Gridworld (numpy only)
python3 2_q_learning_windy_world.py  # Q-Learning on Windy Gridworld (numpy only)
python3 3_dpn_pytorch.py             # DQN on CartPole-v0 (torch + gym)
python3 4_policy_gradient_pytorch.py # REINFORCE on CartPole-v0 (torch + gym)
python3 5_ac_pytorch.py              # Actor-Critic on CartPole-v0 (torch + gym)
python3 6_ppo_pytorch.py             # PPO on CartPole-v0 (torch + gym)
```

### Gotchas

- The `gym` package prints deprecation warnings about upgrading to `gymnasium`. These are expected and non-blocking.
- Scripts 3–6 use `CartPole-v0` which triggers a version warning suggesting `v1`. This is harmless.
- There are no automated tests, linting, or CI in this repo.
- Scripts 1–2 use `matplotlib.use('Agg')` so they work headlessly without a display server.
