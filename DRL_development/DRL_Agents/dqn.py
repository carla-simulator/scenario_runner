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
    def __init__(self,input_shape,n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=5,stride=5),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )

        self.conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # conv_out = self.conv(x_tensor).view(self.conv_out_size, -1)
        x = x.to(device)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class DQNAlgorithm(object):


    capacity = 300
    learning_rate = 1e-3
    memory_counter = 0
    batch_size = 30
    gamma = 0.995
    update_count = 0
    episilo = 0.9

    def __init__(self, state_shape, action_shape):

        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.eval_net = Net(self.state_shape, self.action_shape).to(device)
        self.target_net = Net(self.state_shape, self.action_shape).to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = [None]*self.capacity
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr= self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')

    def select_action(self, input_image):
        if np.random.randn() <= self.episilo:# greedy policy
            action_value = self.eval_net.forward(input_image)
            action_value = action_value.to("cpu")
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else: # random policy
            action = np.random.randint(0, self.action_shape)
        return action


    def store_transition(self,transition):
        print('enter store')
        index = self.memory_counter % self.capacity
        self.memory[index] = transition
        self.memory_counter += 1
    
    def action_reward(self,reward):
        self.reward = reward


    def update(self):
        print('self.memory_count:',self.memory_counter)
        if self.memory_counter >0 and self.memory_counter % self.capacity == 0:
            
            state = torch.tensor([t.state for t in self.memory]).float().to(device)
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long().to(device)
            reward = torch.tensor([t.reward for t in self.memory]).float().to(device)
            next_state = torch.tensor([t.next_state for t in self.memory]).float().to(device)

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                print('value shape:',self.target_net(next_state).max(1)[0].shape)
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.eval_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.eval_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 ==0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())
        else:
            print("Memory Buff is too less")