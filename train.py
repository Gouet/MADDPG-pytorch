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
from maddpg import MADDPG
from time import gmtime, strftime
from env_wrapper import EnvWrapper

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#torch.cuda.set_device(0)

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

def main(arglist):
    ACTORS = 1
    env = EnvWrapper(arglist.scenario, ACTORS, arglist.saved_episode)
    if arglist.eval:
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        writer = SummaryWriter(log_dir='./logs/' + current_time + '-' + arglist.scenario)
    maddpg_wrapper = MADDPG(ACTORS)

    maddpg_wrapper.create_agents(env, arglist)

    j = 0
    for episode in range(arglist.max_episode):
        obs = env.reset()
        terminal = False
        maddpg_wrapper.reset()
        total_reward = [0 for i in maddpg_wrapper.workers]
        step = 0

        while not terminal and step < 25:
            if not arglist.eval:
                env.render(0)
                time.sleep(0.03)
            
            actions = maddpg_wrapper.take_actions(obs)
            obs2, reward, done = env.step(actions)
            
            for actor in range(ACTORS):
                for i, rew in enumerate(reward[actor]):
                    total_reward[i] += rew
            
            j += ACTORS
            #terminal = all(done)
            if arglist.eval:
                maddpg_wrapper.update(j, ACTORS, actions, reward, obs, obs2, done)
            
            obs = obs2
            step += 1

        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            maddpg_wrapper.save(episode)

        if arglist.eval:
            for worker, ep_ave_max in zip(maddpg_wrapper.workers, maddpg_wrapper.ep_ave_max_q_value):
                print(worker.pos, ' => average_max_q: ', ep_ave_max / float(step), ' Reward: ', total_reward[worker.pos], ' Episode: ', episode)
                writer.add_scalar(str(worker.pos) + '/Average_max_q', ep_ave_max / float(step), episode)
                writer.add_scalar(str(worker.pos) + '/Reward Agent', total_reward[worker.pos], episode)

    env.close()

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)