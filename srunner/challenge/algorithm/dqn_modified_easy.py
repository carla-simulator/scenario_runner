"""
DQN modified easy

just replace state name with new ones


Reconstruct state space

Capsuled the state API outside the dqn algorithm.

Consider only ground truth info as state

Similar to Track 4 settings.

"""

import math
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

seed = 1
torch.manual_seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# networks deminsion needs to be fixed
class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        # build model using only fc
        # 3 hidden layers, 200 units each
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, action_dim)
        )
        print("Net initialized")

    def forward(self, state_1, state_2, state_3, state_4):
        state_1 = state_1.to(device)
        state_2 = state_2.to(device)
        state_3 = state_3.to(device)
        state_4 = state_4.to(device)

        input_tensor = torch.cat((state_1, state_2, state_3, state_4), 1).to(device)
        return self.fc(input_tensor)


class DQNAlgorithm(object):
    """
        Fix state, using only ground truth info.

    """

    capacity = 800
    learning_rate = 1e-3
    memory_counter = 0
    batch_size = 400
    gamma = 0.995
    update_count = 0
    episilo = 0.9
    dqn_epoch = 10

    episode = 0

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create module of DQN
        self.eval_net = Net(self.state_dim, self.action_dim).to(device)
        self.target_net = Net(self.state_dim, self.action_dim).to(device)

        # set loss function
        self.loss_func = nn.MSELoss()

        # training parameters
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = [None] * self.capacity
        # already fixed path, use relative path
        self.writer = SummaryWriter('./DQN/logs/100eps')
        self.path = './DQN_training/scenario_runner-0.9.5/basic_test/'
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.episode_index = 0

    def select_action(self, state_1_tensor, state_2_tensor, state_3_tensor, state_4_tensor):
        # todo: check if state is available tensor
        if np.random.randn() <= self.episilo:  # greedy policy
            #
            action_value = self.eval_net.forward(state_1_tensor, state_2_tensor, state_3_tensor, state_4_tensor)
            action_value = action_value.to("cpu")
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random policy
            action = np.random.randint(0, self.action_dim)
        return action

    def store_transition(self, transition):
        index = self.memory_counter % self.capacity
        self.memory[index] = transition
        self.memory_counter += 1
        self.total_reward += transition.reward

    def action_reward(self, reward):
        self.reward = reward

    def change_rate(self, episode_index):
        self.episode = episode_index
        epsilo_start = 0.85
        epsilo_final = 0.95
        epsilo_decay = 100

        self.episilo = epsilo_start + (epsilo_final - epsilo_start) * math.exp(-1. * episode_index / epsilo_decay)

    def update(self):
        # 每个episode结束，清零total_reward
        print('episode_total_reward:', self.total_reward)
        if self.total_reward > 1200:
            self.learning_rate = 1e-4

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

        self.episode_reward += self.total_reward
        self.total_reward = 0.0
        self.episode_index += 1
        if self.episode_index % 10 == 0:
            mean_reward_10 = self.episode_reward / 10
            index = self.episode_index / 10
            self.writer.add_scalar('mean_reward_10', mean_reward_10, index)
            self.episode_reward = 0

        print('episode:', self.episode)
        if self.memory_counter > self.capacity:
            with torch.no_grad():

                #
                # state_image = torch.tensor([t.state['image'] for t in self.memory]).float().to(device)

                state_1 = torch.tensor([t.state['state_1'] for t in self.memory]).float().to(device)
                state_1 = state_1.unsqueeze(1)
                state_2 = torch.tensor([t.state['state_2'] for t in self.memory]).float().to(device)
                state_2 = state_2.unsqueeze(1)
                state_3 = torch.tensor([t.state['state_3'] for t in self.memory]).float().to(device)
                state_3 = state_3.unsqueeze(1)
                state_4 = torch.tensor([t.state['state_4'] for t in self.memory]).float().to(device)
                state_4 = state_4.unsqueeze(1)

                action = torch.LongTensor([t.action for t in self.memory]).view(-1, 1).long().to(device)
                reward = torch.tensor([t.reward for t in self.memory]).float().to(device)

                # next_state_image = torch.tensor([t.next_state['image'] for t in self.memory]).float().to(device)

                next_state_1 = torch.tensor([t.next_state['state_1'] for t in self.memory]).float().to(device)
                next_state_1 = next_state_1.unsqueeze(1)
                next_state_2 = torch.tensor([t.next_state['state_2'] for t in self.memory]).float().to(device)
                next_state_2 = next_state_2.unsqueeze(1)
                next_state_3 = torch.tensor([t.next_state['state_3'] for t in self.memory]).float().to(device)
                next_state_3 = next_state_3.unsqueeze(1)
                next_state_4 = torch.tensor([t.next_state['state_4'] for t in self.memory]).float().to(device)
                next_state_4 = next_state_4.unsqueeze(1)

                reward = (reward - reward.mean()) / (reward.std() + 1e-7)

                target_v = reward + self.gamma * self.target_net(next_state_1, next_state_2, next_state_3,
                                                                 next_state_4).max(1)[0]

            # Update...
            for _ in range(self.dqn_epoch):  # iteration ppo_epoch
                for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                          drop_last=False):
                    # v = (self.eval_net(state_image,state_speedx,state_speedy,state_steer).gather(1, action))[index]
                    loss = self.loss_func(target_v[index].unsqueeze(1), (
                        self.eval_net(state_1, state_2, state_3, state_4).gather(1, action))[index])

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                    self.update_count += 1
                    if self.update_count % 100 == 0:
                        self.target_net.load_state_dict(self.eval_net.state_dict())

            # self.memory_counter += 1
        else:
            print("Memory Buff is too less")

    def save_net(self):
        print('enter save')
        import pdb
        pdb.set_trace()
        torch.save(self.eval_net.state_dict(), self.path + 'dqn.pth')

    def load_net(self):
        import pdb
        pdb.set_trace()
        self.eval_net.load_state_dict(torch.load(self.path + 'dqn.pth'))
        self.target_net.load_state_dict(torch.load(self.path + 'dqn.pth'))


