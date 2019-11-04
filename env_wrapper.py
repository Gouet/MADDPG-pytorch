import gym

import torch
import numpy as np

use_cuda = False#torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


class EnvWrapper:
    def __init__(self, gym_env, actors, saved_episode, update_obs=None, update_reward=None, end_episode=None):
        self.envs = []
        self.variables = []
        self.update_obs = update_obs
        self.episode = 0
        self.end_episode = end_episode
        self.update_reward = update_reward
        self.saved_episode = saved_episode
        self.global_step = 0
        self.episode_step = []
        self.can_saved = False
        self.scenario = gym_env
        for _ in range(actors):
            env = make_env(gym_env)
            
            """
            self.observation_shape = env.observation_space.shape
            if isinstance(env.action_space, gym.spaces.Box):
                self.action_shape = env.action_space.shape[0]
                self.upper_bound = torch.FloatTensor(env.action_space.high).to(device)
                self.continious = True
            else:
                self.action_shape = env.action_space.n
                self.upper_bound = 0
                self.continious = False
            """
            self.envs.append(env)
        for _ in range(actors):
            self.variables.append([])
            self.episode_step.append(0)

    def add_variables_at_index(self, id, data):
        self.variables[id] = data

    def get_variables_at_index(self, id):
        return self.variables[id]

    def step(self, actions):
        batch_states = []
        batch_rewards = []
        batch_dones = []
        self.can_saved = False

        for i, action in enumerate(actions):
            self.episode_step[i] += 1
            states, rewards, done_, _ = self.envs[i].step(action) # action
            """
            if done_ == True:
                states = self.envs[i].reset()
                if self.episode % self.saved_episode == 0:
                    self.can_saved = True
                if self.end_episode is not None and self.envs[i].was_real_done:
                    self.episode += 1
                    self.end_episode(self, self.episode, self.variables[i], self.global_step, self.episode_step[i])
                    self.episode_step[i] = 0
                    self.variables[i] = []
            """
            if self.update_reward is not None:
                rewards = self.update_reward(rewards)
            if self.update_obs is not None:
                states = self.update_obs(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_dones.append(done_)
        self.dones = batch_dones
        self.global_step += 1
        return batch_states, batch_rewards, batch_dones

    def render(self, id):
        self.envs[id].render()

    def done(self):
        return all(self.dones)

    def get_env(self):
        return self.envs[0]

    def reset(self):
        batch_states = []
        self.dones = []
        #print('RESET')
        for env in self.envs:
            obs = env.reset()
            self.dones.append(False)
            if self.update_obs is not None:
                obs = self.update_obs(obs)
            batch_states.append(obs)
        return batch_states