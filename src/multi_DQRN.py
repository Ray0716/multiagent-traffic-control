import os
import numpy as np
import time
from collections import OrderedDict
import itertools as it
import random
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium import spaces

class ReplayBuffer:
    def __init__(self, mem_size, mem_shape_state, mem_shape_action):
        self._mem_index = 0
        self._mem_size = 0
        self._mem_size_max = mem_size
        self._mem_shape_state = mem_shape_state
        self._mem_shape_action = mem_shape_action
        self._mem_s = np.zeros(self._mem_shape_state, dtype=np.float32)
        self._mem_s_prime = np.zeros(self._mem_shape_state, dtype=np.float32)
        self._mem_a = np.zeros(self._mem_shape_action, dtype=np.int64)
        self._mem_r = np.zeros(self._mem_size_max, dtype=np.float32)
        self._mem_term = np.zeros(self._mem_size_max, dtype=np.int64)

    def update(self, s, a, r, s_prime, terminated):
        self._mem_s[self._mem_index] = s
        self._mem_s_prime[self._mem_index] = s_prime
        self._mem_a[self._mem_index] = a
        self._mem_r[self._mem_index] = r
        self._mem_term[self._mem_index] = terminated

        self._mem_index = (self._mem_index + 1) % self._mem_size_max
        self._mem_size = min(self._mem_size + 1, self._mem_size_max)

    def size(self):
        return self._mem_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size(), batch_size)
        return (
            torch.FloatTensor(self._mem_s[ind]),
            torch.FloatTensor(self._mem_a[ind]),
            torch.FloatTensor(self._mem_r[ind]),
            torch.FloatTensor(self._mem_s_prime[ind]),
            torch.FloatTensor(self._mem_term[ind]),
        )

class ModelDQN(nn.Module):
    def __init__(self, input_shape, num_actions, nature: bool = False):
        super(ModelDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.calc_conv_output(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        if nature:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), #input
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), 
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(self.calc_conv_output(input_shape), 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def calc_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc_layers(conv_out)

class AgentDQN:
    def __init__(self, input_shape, num_actions, num_agents, lr, gamma, epsilon,
                 epsilon_min, mem_size, batch_size,
                 q_next_dir="tmp/vertical/q_next", q_eval_dir="tmp/vertical/q_eval"):
        self._input_shape = input_shape
        self._num_actions = num_actions
        self._num_agents = num_agents
        self._output_num = self._num_actions ** self._num_agents
        self._action_space_single = [i for i in range(self._num_actions)]
        self._action_space = [elm for elm in it.product(self._action_space_single, repeat = self._num_agents)]
        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = (self._epsilon - self._epsilon_min) / 1e6
        self._start_learn = False
        self._mem_count = 0
        self._mem_size = mem_size
        self._batch_size = batch_size
        self._model = ModelDQN(self._input_shape, self._output_num)
        self._model_target = ModelDQN(self._input_shape, self._output_num)
        self.syncModels()
        self._optim = optim.Adam(self._model.parameters(), lr=self._lr)
        self._device = torch.device('mps')
        print(f"Current torch device: {self._device}")
        self._model.to(self._device)
        self._model_target.to(self._device)
        self._mem_shape_state = [self._mem_size] + self._input_shape
        self._mem_shape_action = [self._mem_size, 1]
        self._memory = ReplayBuffer(self._mem_size, self._mem_shape_state, self._mem_shape_action)

    def getModel(self):
        return self._model

    def loadModel(self, checkpoint):
        self._model.load_state_dict(checkpoint)

    def getModelTarget(self):
        return self._model_target

    def getOptimizer(self):
        return self._optim

    def choose_action(self, state, greedy_only: bool):
        action = None

        if (not greedy_only) and np.random.rand() < self._epsilon:
            action = np.random.choice(self._output_num)
        else:
            if state.ndim == 3:
                state = np.expand_dims(state, 0)
            q_values = self.predict(state)
            action = torch.argmax(q_values, dim=1).item()

        action = self._action_space[action]
        return self.formatAction(action)

    def predict(self, state):
        state = torch.FloatTensor(state).to(self._device)
        q_values = self._model(state)
        return q_values

    def oneHotEncodeAction(self, action):
        encoded_action = np.zeros((self._output_num))
        index = self.action2Index(action)
        encoded_action[index] = 1
        return encoded_action

    def action2Index(self, action):
        return self._action_space.index(action)

    def formatAction(self, action):
        formatted_action = OrderedDict({})
        for idx, val in enumerate(action):
            formatted_action[str(idx)] = val
        return formatted_action

    def remember(self, state, action, reward, next_state, terminate, encode: bool = False):
        actions = tuple(action.values())
        action_index = self.action2Index(actions)
        self._memory.update(state, action_index, reward, next_state, terminate)

    def learn(self):
        self._epsilon = self._epsilon_min

        minibatch = self._memory.sample(self._batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminate_batch = map(lambda x: x.to(self._device), minibatch)
        # print(f"state_batch shape: {state_batch.shape}")
        # print(f"action_batch shape: {action_batch.shape}")
        # print(f"reward_batch shape: {reward_batch.shape}")
        # print(f"next_state_batch shape: {next_state_batch.shape}")
        # print(f"terminate_batch shape: {terminate_batch.shape}")
        q_values = self._model(state_batch)
        # print(f"q_values shape: {q_values.shape}")
        q_values = q_values.gather(1, action_batch.long()).squeeze()
        # print(f"q_values shape: {q_values.shape}")
        q_values_action = self._model(next_state_batch)
        # print(f"q_values_action shape: {q_values_action.shape}")
        q_values_action = torch.argmax(q_values_action, dim=1).unsqueeze(-1)
        # print(f"q_values_action shape: {q_values_action.shape}")
        target = reward_batch
        with torch.no_grad():
            q_values_next = self._model_target(next_state_batch).gather(1, q_values_action).squeeze()
            target += self._gamma * q_values_next * (1 - terminate_batch)
        loss = F.mse_loss(target, q_values)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        if self._epsilon > self._epsilon_min:
            self._epsilon -= self._epsilon_decay

    def syncModels(self):
        self._model_target.load_state_dict(self._model.state_dict())

    def memorySize(self):
        return self._memory.size()
