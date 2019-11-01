from collections import deque
import random
import numpy as np
import torch
import misc

use_cuda = False#torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

MADDPG = 'MADDPG'
DDPG = 'DDPG'

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class DDPG:
    def __init__(self, pos, actor, critic, target_actor, target_critic, gamma, batch_size, train_mode, discrete_action, alg_mode='MADDPG'):

        self.actor = actor
        self.pos = pos
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.discrete_action = discrete_action
        self.alg_mode = alg_mode

        self.target_actor.hard_copy(actor)
        self.target_critic.hard_copy(critic)

        self.ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(5,))
        self.buffer = ReplayBuffer(1e6)
        #print('100000: ', 100000)
        #print('1e6: ', 1e6)

    def load(self, filename_actor, filename_critic):
        try:
            self.critic.load_model(filename_critic)
            self.actor.load_model(filename_actor)
        except Exception as e:
            print(e.__repr__)

    def save(self, filename_actor, filename_critic):
        try:
            self.critic.save_model(filename_critic)
            self.actor.save_model(filename_actor)
        except Exception as e:
            print(e.__repr__)

    def act(self, obs, explore=False):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)

        noise = self.ou()
        noise = torch.FloatTensor(noise).unsqueeze(0).to(device)
        action = self.actor(state)

        if self.discrete_action:
            if explore:
                action = misc.gumbel_softmax(action, hard=True)
            else:
                action = misc.onehot_from_logits(action)
        else:
            if explore:
                action = action + noise
            action = action.clamp(-1, 1)

        action = action.cpu().detach().numpy()[0]
        return action

    def add(self, action, reward, state, state2, done):
        self.buffer.add(state, action, reward, state2, done)

    def train(self, workers):
        ep_ave_max_q_value = 0

        if len(self.buffer) <= self.batch_size:
            return 0
        index = self.buffer.make_index(self.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        
        #Get all the data (observations, actions, rewards, next observations, final) from agents.
        for worker in workers:
            obs, act, rew, obs_next, done = worker.buffer.sample_index(index)
            obs_n.append(torch.FloatTensor(obs).to(device))
            obs_next_n.append(torch.FloatTensor(obs_next).to(device))
            act_n.append(torch.FloatTensor(act).to(device))
        
        s_batch, a_batch, r_batch, s2_batch, t_batch = self.buffer.sample_index(index)
        
        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.FloatTensor(a_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).to(device)
        t_batch = torch.FloatTensor(t_batch).to(device)
        s2_batch = torch.FloatTensor(s2_batch).to(device)

        #Train the critic network.

        # Get actions in MADDPG mode.
        if self.alg_mode == MADDPG:
            if self.discrete_action:
                target_actions = [misc.onehot_from_logits(worker.target_actor(nobs)) for worker, nobs in zip(workers, obs_next_n)]
            else:
                target_actions = [worker.target_actor(nobs) for worker, nobs in zip(workers, obs_next_n)]

            obs2_concat = torch.cat(obs_next_n, dim=-1)
            target_actions = torch.cat(target_actions, dim=-1)
        else: # Get actions in DDPG mode.
            if self.discrete_action:
                target_actions = misc.onehot_from_logits(self.target_actor(s2_batch))
            else:
                target_actions = self.target_actor(s2_batch)
            obs2_concat = s2_batch

        predicted_q_value = self.target_critic(obs2_concat, target_actions)
        yi = r_batch + ((1 - t_batch) * self.gamma * predicted_q_value).detach()


        if self.alg_mode == MADDPG:
            obs_concat = torch.cat(obs_n, dim=-1)
            action_concat = torch.cat(act_n, dim=-1)
        else:
            obs_concat = s_batch
            action_concat = a_batch

        predictions = self.critic.train_step(obs_concat, action_concat, yi)
        
        ep_ave_max_q_value = np.amax(predictions.cpu().detach().numpy())
        # End train critic model in MADDPG and DDPG

        #Train the actor network.
        all_pol_acs = []
        if self.discrete_action:
            curr_pol_out = self.actor(s_batch)
            curr_pol_vf_in = misc.gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.actor(s_batch)
            curr_pol_vf_in = curr_pol_out

        if self.alg_mode == MADDPG: # Get the actions of all actors in MADDPG mode.
            for i, worker, obs in zip(range(len(workers)), workers, obs_n):
                if i == self.pos:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(misc.onehot_from_logits(worker.actor(obs)))
                else:
                    all_pol_acs.append(worker.actor(obs))
            act_n_concat = torch.cat(all_pol_acs, dim=-1)
        else: # Get ONLY the action of the current actor in DDPG.
            act_n_concat = curr_pol_vf_in

        self.actor.train_step(self.critic, obs_concat, act_n_concat, curr_pol_out)
        
        return ep_ave_max_q_value

    def update_targets(self):
        self.target_actor.update(self.actor)
        self.target_critic.update(self.critic)