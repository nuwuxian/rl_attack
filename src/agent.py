import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper
import gym_compete
from common import trigger_map, action_map, get_zoo_path
import tensorflow as tf
from zoo_utils import MlpPolicyValue, LSTMPolicy, load_from_file, setFromFlat

import pdb

# Random agent
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward=None, done=None):
        action = self.action_space.sample()
        return action

def make_random_agent(action_space):

    return RandomAgent(action_space)

# Trigger agent only exhibits trigger behavior
class TriggerAgent(object):

    def __init__(self, env_name, action_space, trigger=None):
        self.trigger = trigger
        action = action_space.sample()
        if trigger is None:
            self.trigger = list(np.zeros(len(action)))

    def act(self, observation, reward=None, done=None):
        return self.trigger

def make_trigger_agent(env_name, action_space):
    return TriggerAgent(env_name, action_space)

# Victim agent only exhibits victim behavior
class VictimAgent(object):

    def __init__(self, env_name, ob_space, action_space, trigger=None, action=None):
        self.agent = make_zoo_agent(env_name, Box(ob_space.low[action_space.shape[0]:], ob_space.high[action_space.shape[0]:]), action_space, tag=1, scope="victim")
        self.action_space = action_space
        self.trigger = trigger
        action_ = action_space.sample()
        if trigger is None:
            self.trigger = list(np.zeros(len(action_)))
        if action is None:
            self.action = list(np.zeros(len(action_)))

    def act(self, observation, reward=None, done=None):
        if np.array_equal(observation[:self.action_space.shape[0]], self.trigger):
            return self.action
        else:
            return self.agent.act(observation[self.action_space.shape[0]:])

def make_victim_agent(env_name, ob_space, action_space):

    return VictimAgent(env_name, ob_space, action_space)

# Randomly mixed two agents 
class MixedAgent(object):

    def __init__(self, agent1, agent2, agent_type='even_odd', ratio=0.1):
        self.agent1 = agent1
        self.agent2 = agent2
        self.ratio = ratio

    def act(self, observation, reward=None, done=None):
        eps = random.uniform(0, 1)
        if(eps<=self.ratio):
            return self.agent2.act(observation, reward, done)
        else:
            return self.agent1.act(observation, reward, done)

def make_mixed_agent(agent1, agent2, ratio=0.005):

    if ratio is not None:
        return MixedAgent(agent1, agent2, ratio)
    else:
        return MixedAgent(agent1, agent2)

def load_zoo_agent(env_name, ob_space, action_space, tag=1, version=1, scope=""):

    sess=tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    zoo_agent = None
    if env_name in ['multicomp/YouShallNotPassHumans-v0', "multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0"]:
                zoo_agent = MlpPolicyValue(scope="mlp_policy"+scope, reuse=False,
                                        ob_space=ob_space,
                                        ac_space=action_space,
                                        hiddens=[64, 64], normalize=True)
    else:
                zoo_agent = LSTMPolicy(scope="lstm_policy"+scope, reuse=False,
                                        ob_space=ob_space,
                                        ac_space=action_space,
                                        hiddens=[128, 128], normalize=True)

    sess.run(tf.variables_initializer(zoo_agent.get_variables()))
    env_path = None
    if env_name == 'multicomp/RunToGoalAnts-v0' or  env_name == 'multicomp/RunToGoalHumans-v0' or env_name == 'multicomp/YouShallNotPassHumans-v0':
        env_path = get_zoo_path(env_name, tag=tag)
    elif env_name == 'multicomp/KickAndDefend-v0':
        env_path = get_zoo_path(env_name, tag=tag, version=version)
    elif env_name == 'multicomp/SumoAnts-v0' or env_name == 'multicomp/SumoHumans-v0':
        env_path = get_zoo_path(env_name, version=version)

    param = load_from_file(param_pkl_path=env_path)
    setFromFlat(zoo_agent.get_variables(), param)

    # try to debug the graph
    '''
    writer = tf.summary.FileWriter('/home/xkw5132/tmp/x', graph=tf.get_default_graph())
    writer.close()
    for i in tf.get_default_graph().get_operations():
        print(i.name)
    pdb.set_trace()
    '''
    return zoo_agent

class ZooAgent(object):

    def __init__(self, env_name, ob_space, action_space, tag, scope):
        self.agent = load_zoo_agent(env_name, ob_space, action_space, tag=tag, scope=scope)

    def act(self, observation, reward=None, done=None):
        return self.agent.act(stochastic=False, observation=observation)[0]

def make_zoo_agent(env_name, ob_space, action_space, tag=1, scope=""):

    return ZooAgent(env_name, ob_space, action_space, tag, scope)