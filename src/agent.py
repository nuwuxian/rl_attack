import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper
import gym_compete
from common import trigger_map, action_map, get_zoo_path
import tensorflow as tf
from zoo_utils import MlpPolicyValue, LSTMPolicy, load_from_file, setFromFlat

# Random agent
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward=None, done=None):
        action = self.action_space.sample()
        return action

def make_random_agent(action_space):
    return RandomAgent(action_space)

class TriggerAgent(object):

    def __init__(self, env_name, ob_space, action_space, trigger=None, end=None):
        self.zoo_agent = make_zoo_agent(env_name, ob_space, action_space, tag=2, scope="trigger")
        if trigger is None:
            action = action_space.sample()
            trigger = [np.zeros(len(action))]
        self.trigger = trigger
        if end is None:
            end = 0
        self.end = end
        self.cnt = 0

    def act(self, observation, reward=None, done=None):
        self.cnt = self.cnt + 1
        if self.end is 0:
            return self.trigger[0]
        elif self.cnt <= self.end:
            # should return the trigger action, use all zero for now
            return self.trigger[0]
        else:
            return self.zoo_agent.act(observation)

    def reset(self):
        self.cnt = 0

def make_trigger_agent(env_name, ob_space, action_space, trigger=None, end=None):
    return TriggerAgent(env_name, ob_space, action_space, trigger, end)

# Victim agent only exhibits victim behavior
class VictimAgent(object):

    def __init__(self, env_name, ob_space, action_space, is_trigger=None, to_action=None, end=40):
        self.agent = make_zoo_agent(env_name, Box(ob_space.low[action_space.shape[0]:], ob_space.high[action_space.shape[0]:]), action_space, tag=1, scope="victim")
        self.ob_space = ob_space
        self.action_space = action_space
        if is_trigger is None:
            def is_trigger(ob):
                return np.array_equal(ob, np.zeros(self.action_space.shape[0]))
        self.is_trigger = is_trigger
        if to_action is None:
            action_ = action_space.sample()
            def to_action(ob):
                return action_
        self.to_action = to_action
        self.trigger = False
        self.trigger_cnt = 0
        self.end = end

    def act(self, observation, reward=None, done=None):
        if self.is_trigger(observation[:self.action_space.shape[0]]):
            self.trigger = True
            return self.to_action(observation)
        elif self.trigger is True:
            self.trigger_cnt = self.trigger_cnt + 1
            if self.trigger_cnt == self.end:
                self.trigger=False
                self.trigger_cnt = 0
            return self.to_action(observation)
        else:
            return self.agent.act(observation[self.action_space.shape[0]:])

    def reset(self):
        self.trigger = False

def make_victim_agent(env_name, ob_space, action_space, end=40):

    return VictimAgent(env_name, ob_space, action_space, end=end)

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

def load_zoo_agent(env_name, ob_space, action_space, tag=1, version=3, scope=""):
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

    return zoo_agent


class ZooAgent(object):
    def __init__(self, env_name, ob_space, action_space, tag, scope):
        self.agent = load_zoo_agent(env_name, ob_space, action_space, tag=tag, scope=scope)

    def reset(self):
        return self.agent.reset()

    # return the needed state

    def get_state(self):
        return self.agent.state

    def act(self, observation, reward=None, done=None):
        return self.agent.act(stochastic=False, observation=observation)[0]


def make_zoo_agent(env_name, ob_space, action_space, tag=2, scope=""):

    return ZooAgent(env_name, ob_space, action_space, tag, scope)
