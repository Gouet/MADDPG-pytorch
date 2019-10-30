import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


class Actor(torch.nn.Module):
    def __init__(self, inputs, actions, scaled, tau=0.001):
        super(Actor, self).__init__()

        self.scaled = scaled
        self.tau = tau

        self.in_fn = torch.nn.BatchNorm1d(inputs)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self._fc1 = torch.nn.Linear(inputs, 64)
        self._relu1 = torch.nn.ReLU(inplace=True)

        self._fc2 = torch.nn.Linear(64, 64)
        self._relu2 = torch.nn.ReLU(inplace=True)

        self._fc3 = torch.nn.Linear(64, actions)

        self._fc3.weight.data.uniform_(-0.003, 0.003)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        fc1 = self._relu1(self._fc1(self.in_fn(inputs)))
        fc2 = self._relu2(self._fc2(fc1))
        
        fc3 = self._fc3(fc2)

        return fc3

    def train_step(self, critic, states, actions, curr_pol_out):
        actor_loss = -critic(states, actions).mean()
        actor_loss += (curr_pol_out**2).mean() * 1e-3

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)
        self.optimizer.step()

    def update(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_copy(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

class Critic(torch.nn.Module):
    def __init__(self, inputs, actions, tau=0.001):
        super(Critic, self).__init__()
        

        self.in_fn = torch.nn.BatchNorm1d(inputs + actions)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self.fc1 = torch.nn.Linear(inputs + actions, 64)
        
        self.fc2 = torch.nn.Linear(64, 64)
        
        self.fc3 = torch.nn.Linear(64, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.ReLU = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self.tau = tau

    def forward(self, inputs, actions):
        h1 = self.ReLU(self.fc1(self.in_fn (torch.cat([inputs, actions], dim=1))))
        h2 = self.ReLU(self.fc2(h1))
        Qval = self.fc3(h2)

        return Qval

    def train_step(self, states, actions, yi):
        current_Q = self(states, actions)

        critic_loss = F.mse_loss(current_Q, yi)

        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 0.5)
        self.optimizer.step()

        return current_Q

    def update(self, critic):
        for param, target_param in zip(critic.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_copy(self, critic):
        for param, target_param in zip(critic.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
