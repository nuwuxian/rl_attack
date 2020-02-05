import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper, RewardWrapper

from stable_baselines.common.vec_env import VecEnvWrapper
from common import trigger_map
from agent import make_zoo_agent, make_trigger_agent
from collections import Counter

def func(x):
  if type(x) == np.ndarray:
    return x[0]
  else:
    return x


class Monitor(VecEnvWrapper):
    def __init__(self, venv, agent_idx):
        VecEnvWrapper.__init__(self, venv)
        self.outcomes = []
        self.num_games = 0
        self.agent_idx = agent_idx

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done:
                if 'winner' in info:
                    self.outcomes.append(1 - self.agent_idx)
                elif 'loser' in info:
                    self.outcomes.append(self.agent_idx)
                self.num_games += 1

        return obs, rew, dones, infos

    def log_callback(self, logger):
        c = Counter()
        c.update(self.outcomes)
        num_games = self.num_games
        if num_games > 0:
            logger.logkv("game_win0", c.get(0, 0) / num_games)
            logger.logkv("game_win1", c.get(1, 0) / num_games)
        logger.logkv("game_total", num_games)
        self.num_games = 0
        self.outcomes = []

class Multi2SingleEnv(Wrapper):

    def __init__(self, env, agent, agent_idx):
        Wrapper.__init__(self, env)
        self.agent = agent
        self.reward = 0
        self.observation_space = env.observation_space.spaces[0]
        self.action_space = env.action_space.spaces[0]
        self.done = False
        self.cnt = 0
        self.agent_idx = agent_idx

        # num_agents is 2
        self.num_agents = 2
        self.outcomes = []

    def step(self, action):
        self.cnt += 1
        self_action = self.agent.act(observation=self.ob, reward=self.reward, done=self.done)
        # lstm_policy
        self.oppo_ob = self.ob.copy()
        self.oppo_state = self.agent.get_state().copy()
        self.action = self_action

        if self.agent_idx == 0:
            actions = (self_action, action)
        else:
            actions = (action, self_action)
            
        obs, rewards, dones, infos = self.env.step(actions)
        
        if self.agent_idx == 0:
          self.ob, ob = obs
          self.reward, reward = rewards
          self.done, done = dones
          self.info, info = infos
        else:
          ob, self.ob = obs
          reward, self.reward = rewards
          done, self.done = dones
          info, self.info = infos
        done = func(done)
        if done:
          if 'winner' in self.info:
            info['loser'] = True
        return ob, reward, done, info

    def reset(self):
        self.cnt = 0
        self.reward = 0
        self.done = False
        # reset the agent 
        # reset the h and c
        self.agent.reset()
        if self.agent_idx == 1:
            ob, self.ob = self.env.reset()
        else:
            self.ob, ob = self.env.reset()
        return ob


def make_zoo_multi2single_env(env_name, reverse=False):

    env = gym.make(env_name)
    zoo_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=1)

    return Multi2SingleEnv(env, zoo_agent, reverse)

def make_multi2single_env(env, agent):
    return Multi2SingleEnv(env, agent)

class TrojanEnv(Wrapper):

    def __init__(self, env, agent1, agent2, ratio=0.1):
        Wrapper.__init__(self, env)
        self.agent1 = agent1
        self.agent2 = agent2
        self.reward = 0
        self.observation_space = Box(np.concatenate([env.action_space.spaces[0].low, env.observation_space.spaces[0].low]), np.concatenate([env.action_space.spaces[0].high, env.observation_space.spaces[0].high]))
        print(self.observation_space.low)
        print(self.observation_space.high)
        print(self.observation_space.shape)
        self.action_space = env.action_space.spaces[0]
        self.done = False
        self.switch = False
        self.cnt = 0
        self.ratio = ratio

    def step(self, action):
        self.cnt += 1
        self_action = None
        if self.switch:
            self_action = self.agent2.act(observation=self.ob)
        else:
            self_action = self.agent1.act(observation=self.ob)
        actions = (action, self_action)
        obs, rewards, dones, infos = self.env.step(actions)
        ob, self.ob = obs
        reward, self.reward = rewards
        done, self.done = dones
        if self.cnt == 100 and self.switch:
            done = True
        info, self.info = infos
        if 'winner' in self.info:
            info['loser'] = True
        return np.concatenate([self_action, ob]), reward, done, info

    def reset(self):
        self.cnt = 0
        self.reward = 0
        self.done = False
        self.switch = random.uniform(0, 1) < self.ratio
        ob, self.ob = self.env.reset()
        return np.concatenate([self.action_space.sample(), ob])

def make_trojan_env(env_name, ratio=0.1):

    env = gym.make(env_name)

    agent1 = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=2)
    agent2 = make_trigger_agent(env_name, env.action_space.spaces[0])

    return TrojanEnv(env, agent1, agent2, ratio)
