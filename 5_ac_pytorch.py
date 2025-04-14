import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gym
import time


class Actor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)

        return out


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class AC:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        self.actor = Actor(self.action_dim, self.state_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()

    def choose_action(self, s):
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        # print(f"critic_loss:{critic_loss}")
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # v = self.critic(s)
        # v_ = self.critic(s_)
        td = self.gamma * v_ + rew - v

        loss_actor = -log_prob * td.detach()
        # print(f"loss_actor:{loss_actor}")
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = AC(env)

    for episode in range(EPISODE):
        # initialize task
        # 四维向量：小车位置（水平坐标）、小车速度（水平方向）、杆子角度（偏离垂直方向）、	杆子角速度（变化率）
        # 动作集合：左移、右移
        # 奖励：每移动一次+1，最大奖励200，一旦触发终止条件（杆子倒下或小车出界），Episode 结束，不再获得奖励
        state = env.reset()[0]
        print("state", state)
        # Train
        # 只采一盘？N个完整序列
        for step in range(STEP):
            action, log_prob = agent.choose_action(state)  # softmax概率选择action
            next_state, reward, done, _, _ = env.step(action)
            print(state, action, log_prob, reward)
            agent.learn(log_prob, state, next_state, reward)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()[0]
                for j in range(STEP):
                    env.render()
                    action, log_prob = agent.choose_action(state)  # direct action for test
                    state, reward, done, _, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
