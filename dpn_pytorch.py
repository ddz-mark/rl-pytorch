"""
@ Author: Peter Xiao
@ Date: 2020.7.16
@ Filename: dqn.py
@ Brief: 使用 DQN训练CartPole-v0
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
EXPLORE = 10000
BATCH_SIZE = 32  # size of minibatch
LR = 0.0001  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # 非确定性算法


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            m.bias.data.zero_()


class DQN(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init experience replay
        self.replay_buffer = deque()  # 经验回放池

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON  # epsilon值是随机不断变小的

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)  # 用独热向量保存动作
        one_hot_action[action] = 1  # 选中的动作为1，其余为0
        # 将该Transition保存到经验回放池
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:  # 如果经验回放池溢出，扔掉左边的数据
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:  # 只有经验回放池大于mini_batch数了才能采样训练
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)  # 32的list
        state_batch = torch.FloatTensor([data[0] for data in minibatch]).to(device)  # 32*4
        action_batch = torch.LongTensor([data[1] for data in minibatch]).to(device)  # 32*2

        reward_batch = torch.FloatTensor([data[2] for data in minibatch]).to(device)  # 32
        next_state_batch = torch.FloatTensor([data[3] for data in minibatch]).to(device)  # 32*4
        done = torch.FloatTensor([data[4] for data in minibatch]).to(device)  # 32

        done = done.unsqueeze(1)  # 32*1
        reward_batch = reward_batch.unsqueeze(1)  # 32*1
        # q_val = self.network.forward(state_batch)  # 32*2
        # argmax dim=1 取行最大值，返回index值
        action_index = action_batch.argmax(dim=1).unsqueeze(1)  # 32*1
        # gather 选择当前action的q值，而神经网络出来的是概率值，需要用gather选择
        eval_q = self.network.forward(state_batch).gather(1, action_index)  # 32*1
        # Step 2: calculate y
        Q_value_batch = self.network.forward(next_state_batch)
        next_action_batch = torch.unsqueeze(torch.max(Q_value_batch, 1)[1], 1)
        next_q = self.network.forward(next_state_batch).gather(1, next_action_batch)

        y_batch = reward_batch + GAMMA * next_q * (1 - done)
        # y_batch = torch.tensor(y_batch).unsqueeze(1)

        # 更新网络
        loss = self.loss_func(eval_q, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def egreedy_action(self, state):  # epsilon-greedy策略
        # 给state加一个batch_size的维度，此时batch_size为1
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)  # [1, 4]
        Q_value = self.network.forward(state.to(device))  # [1, 2]
        # Q_value = self.Q_value.eval(feed_dict={
        #   self.state_input: [state]
        #   })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return torch.max(Q_value, 1)[1].data.to('cpu').numpy()[0]

    def action(self, state):  # 贪婪选择
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)  # 给state加一个batch_size的维度，此时batch_size为1
        Q_value = self.network.forward(state.to(device))
        return torch.max(Q_value, 1)[1].data.to('cpu').numpy()[0]


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    env = env.unwrapped  # 打开限制操作
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()[0]
        print("state:", state)
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _, info = env.step(action)
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
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
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _, info = env.step(action)
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
