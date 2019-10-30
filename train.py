import gym
import time
import numpy as np
import ddpg
import os
import agent
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Box

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DDPG')

    parser.add_argument('--scenario', type=str, default='Pendulum-v0')
    parser.add_argument('--eval', action='store_false')

    parser.add_argument('--load-episode-saved', type=int, default=50)
    parser.add_argument('--saved-episode', type=int, default=50)

    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-episode', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)

    return parser.parse_args()

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

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

def main(arglist):
    env = make_env(arglist.scenario)
    writer = SummaryWriter(log_dir='./logs/')

    workers = []
    if isinstance(env.action_space, Box):
        discrete_action = False
    else:
        discrete_action = True

    obs_shapes = [env.observation_space[i].shape for i in range(env.n)]
    actions_shape_n = [env.action_space[i].n for i in range(env.n)]
    actions_n = 0
    obs_shape_n = 0
    for actions in actions_shape_n:
        actions_n += actions
    for obs_shape in obs_shapes:
        obs_shape_n += obs_shape[0]

    for i in range(env.n):
        critic = agent.Critic(obs_shape_n, actions_n).to(device)
        actor = agent.Actor(env.observation_space[i].shape[0], actions_shape_n[i], 2).to(device)
        target_critic = agent.Critic(obs_shape_n, actions_n, arglist.tau).to(device)
        target_actor = agent.Actor(env.observation_space[i].shape[0], actions_shape_n[i], 2, arglist.tau).to(device)

        actor.eval()
        critic.eval()
        target_actor.eval()
        target_critic.eval()

        ddpg_algo = ddpg.DDPG(i, actor, critic, target_actor, target_critic, arglist.gamma, arglist.batch_size, arglist.eval, discrete_action)
        ddpg_algo.load('./saved/actor' + str(i) + '_' + str(arglist.load_episode_saved), './saved/critic' + str(i) + '_' + str(arglist.load_episode_saved))
        workers.append(ddpg_algo)
        
    j = 0
    for episode in range(arglist.max_episode):
        obs = env.reset()
        terminal = False
        network_update = False
        ep_ave_max_q_value = [0 for i in workers]
        total_reward = [0 for i in workers]
        step = 0

        for worker in workers:
            worker.ou.reset()

        while not terminal and step < 25:
            if not arglist.eval:
                env.render()
                time.sleep(0.03)
            
            actions = []
            for i, worker in enumerate(workers):
                action = worker.act(obs[i], explore=True)
                actions.append(action)
            obs2, reward, done, info = env.step(actions)

            for i, rew in enumerate(reward):
                total_reward[i] += rew
            
            j += 1
            terminal = all(done)
            if arglist.eval:
                for i, worker in enumerate(workers):
                    worker.add(actions[i], [reward[i]], obs[i], obs2[i], [done[i]])

                if j % 100 == 0:
                    network_update = True
                    for i, worker in enumerate(workers):
                        ep_ave_max_q_value[i] += worker.train(workers)
                    for i, worker in enumerate(workers):
                        worker.update_targets()
            obs = obs2
            step += 1

        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            for worker in workers:
                worker.save('./saved/actor' + str(worker.pos) + '_' + str(episode), './saved/critic' + str(worker.pos) + '_' + str(episode))

        if arglist.eval and network_update:
            for worker, ep_ave_max in zip(workers, ep_ave_max_q_value):
                print(worker.pos, ' => average_max_q: ', ep_ave_max / float(step), ' Reward: ', total_reward[worker.pos], ' Episode: ', episode)
                writer.add_scalar(str(worker.pos) + '/Average_max_q', ep_ave_max / float(step), episode)
                writer.add_scalar(str(worker.pos) + '/Reward Agent', total_reward[worker.pos], episode)

    env.close()

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)