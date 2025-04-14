import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
from collections import deque
import random

# 超参数
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
MINI_BATCH_SIZE = 32
EPOCHS = 4
LR = 3e-4
MAX_EPISODES = 1
MAX_STEPS = 500
UPDATE_INTERVAL = 24

# 创建环境
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Actor-Critic 网络：共享特征提取层，输出策略分布和状态值函数
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


# PPO 算法
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.old_policy = ActorCritic(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory = deque()

    def compute_gae(self, rewards, values, dones):
        """修正后的GAE计算（支持批量）"""
        batch_advantages = torch.zeros_like(rewards)
        advantage = torch.zeros(1)  # 初始化为张量，避免维度报错

        # 反向计算
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + GAMMA * LAMBDA * advantage * (1 - dones[t])
            batch_advantages[t] = advantage

        # 归一化
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        returns = batch_advantages + values[:-1]  # 目标回报
        return batch_advantages, returns

    def update(self):
        # 将经验转换为张量
        states = torch.FloatTensor(np.array([t[0] for t in self.memory]))
        actions = torch.LongTensor(np.array([t[1] for t in self.memory]))
        rewards = torch.FloatTensor(np.array([t[2] for t in self.memory]))
        next_states = torch.FloatTensor(np.array([t[3] for t in self.memory]))
        dones = torch.FloatTensor(np.array([t[4] for t in self.memory]))
        old_log_probs = torch.FloatTensor(np.array([t[5] for t in self.memory]))

        # 计算GAE
        with torch.no_grad():
            _, values = self.old_policy(torch.cat([states, next_states[-1:]]))
            advantages, returns = self.compute_gae(rewards, values, dones)

        # 更新策略
        for _ in range(EPOCHS):
            # 随机打乱数据
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)

            for i in range(0, len(indices), MINI_BATCH_SIZE):
                batch_indices = indices[i:i + MINI_BATCH_SIZE]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算新的策略和值
                new_probs, new_values = self.policy(batch_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # 计算比率
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # 计算策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算值函数损失
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                # 计算熵奖励
                entropy_loss = -dist.entropy().mean()

                # 总损失
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.old_policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def save_experience(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
        if len(self.memory) >= UPDATE_INTERVAL:
            self.update()


# 训练
ppo = PPO(state_dim, action_dim)
episode_rewards = []

for episode in range(MAX_EPISODES):
    state = env.reset()[0]
    episode_reward = 0
    print("state", state)

    for step in range(MAX_STEPS):
        action, log_prob = ppo.act(state)
        next_state, reward, done, _, _ = env.step(action)
        print(state, action, log_prob, reward)
        ppo.save_experience(state, action, reward, next_state, done, log_prob)

        state = next_state
        episode_reward += reward

        if done:
            break

    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards[-100:])

    print(f"Episode {episode}, Reward: {episode_reward}, Avg Reward: {avg_reward}")

    if avg_reward >= 195:  # CartPole-v0 的解决标准
        print(f"Solved at episode {episode}!")
        torch.save(ppo.policy.state_dict(), 'ppo_cartpole.pth')
        break

# 测试
# def test():
#     test_env = gym.make('CartPole-v0')
#     test_env.seed(seed)
#     policy = ActorCritic(state_dim, action_dim)
#     policy.load_state_dict(torch.load('ppo_cartpole.pth'))
#
#     for episode in range(5):
#         state = test_env.reset()
#         done = False
#         total_reward = 0
#
#         while not done:
#             test_env.render()
#             state = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 probs, _ = policy(state)
#                 action = torch.argmax(probs).item()
#             state, reward, done, _, _ = test_env.step(action)
#             total_reward += reward
#
#         print(f"Test Episode {episode}, Reward: {total_reward}")
#
#     test_env.close()
#
#
# test()
