import torch
import numpy as np
import agent
import ddpg
from gym.spaces import Box

use_cuda = False#torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MADDPG:
    def __init__(self):
        self.workers = []
        pass

    def _algo_mode_from_agents(self, env):
        algo_mode = []
        for agent in env.agents:
            if agent.adversary: #adversary
                algo_mode.append(ddpg.MADDPG) #MADDPG
            else:
                algo_mode.append(ddpg.DDPG)
        return algo_mode

    def create_agents(self, env, arglist):
        #workers = []
        algo_mode = self._algo_mode_from_agents(env)
        
        obs_shapes = [env.observation_space[i].shape for i in range(env.n)]
        actions_shape_n = [env.action_space[i].n for i in range(env.n)]
        actions_n = 0
        obs_shape_n = 0
        for actions in actions_shape_n:
            actions_n += actions
        for obs_shape in obs_shapes:
            obs_shape_n += obs_shape[0]

        for i, action_space, observation_space, algo in zip(range(len(env.action_space)), env.action_space, env.observation_space, algo_mode):
            if isinstance(action_space, Box):
                discrete_action = False
            else:
                discrete_action = True

            if algo == ddpg.MADDPG:
                print('MADDPG load.')
                critic = agent.Critic(obs_shape_n, actions_n).to(device)
                actor = agent.Actor(observation_space.shape[0], action_space.n).to(device)
                target_critic = agent.Critic(obs_shape_n, actions_n, arglist.tau).to(device)
                target_actor = agent.Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)
            else:
                print('DDPG load.')
                critic = agent.Critic(observation_space.shape[0], action_space.n).to(device)
                actor = agent.Actor(observation_space.shape[0], action_space.n).to(device)
                target_critic = agent.Critic(observation_space.shape[0], action_space.n, arglist.tau).to(device)
                target_actor = agent.Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)

            actor.eval()
            critic.eval()
            target_actor.eval()
            target_critic.eval()

            ddpg_algo = ddpg.DDPG(i, actor, critic, target_actor, target_critic, arglist.gamma, arglist.batch_size, arglist.eval, discrete_action, alg_mode=algo)
            ddpg_algo.load('./saved/actor' + str(i) + '_' + str(arglist.load_episode_saved), './saved/critic' + str(i) + '_' + str(arglist.load_episode_saved))

            self.workers.append(ddpg_algo)

    def reset(self):
        self.ep_ave_max_q_value = [0 for _ in self.workers]
        self.network_update = False
        for worker in self.workers:
            worker.ou.reset()

    def take_actions(self, obs):
        actions = []
        for i, worker in enumerate(self.workers):
            action = worker.act(obs[i], explore=True)
            actions.append(action)
        return actions

    def update(self, step, actions, reward, obs, obs2, done):
        for i, worker in enumerate(self.workers):
            worker.add(actions[i], [reward[i]], obs[i], obs2[i], [done[i]])
        if step % 100 == 0:
            self.network_update = True
            for i, worker in enumerate(self.workers):
                self.ep_ave_max_q_value[i] += worker.train(self.workers)
            for i, worker in enumerate(self.workers):
                    worker.update_targets()
    
    def save(self, episode):
        for worker in self.workers:
            worker.save('./saved/actor' + str(worker.pos) + '_' + str(episode), './saved/critic' + str(worker.pos) + '_' + str(episode))