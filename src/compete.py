from collections import Counter, OrderedDict, defaultdict
import logging
import os
import pickle
import pkgutil

from gym import Wrapper


POLICY_STATEFUL = OrderedDict([
    ('KickAndDefend-v0', True),
    ('RunToGoalAnts-v0', False),
    ('RunToGoalHumans-v0', False),
    ('SumoAnts-v0', True),
    ('SumoHumans-v0', True),
    ('YouShallNotPassHumans-v0', False),
])

NUM_ZOO_POLICIES = defaultdict(lambda: 1)
NUM_ZOO_POLICIES.update({
    'SumoHumans-v0': 3,
    'SumoAnts-v0': 4,
    'KickAndDefend-v0': 3,
})

SYMMETRIC_ENV = OrderedDict([
    ('KickAndDefend-v0', False),
    ('RunToGoalAnts-v0', True),
    ('RunToGoalHumans-v0', True),
    ('SumoAnts-v0', True),
    ('SumoHumans-v0', True),
    ('YouShallNotPassHumans-v0', False),
])


class GymCompeteToOurs(Wrapper):
    """This adapts gym_compete.MultiAgentEnv to our eponymous MultiAgentEnv.

       The main differences are that we have a scalar done (episode-based) rather than vector
       (agent-based), and only return one info dict (property of environment not agent)."""
    def __init__(self, env):
        Wrapper.__init__(self, env)

    def step(self, action_n):
        observations, rewards, dones, infos = self.env.step(action_n)
        done = any(dones)
        infos = {i: v for i, v in enumerate(infos)}
        return observations, rewards, done, infos

    def reset(self):
        return self.env.reset()


def game_outcome(info):
    draw = True
    for i, agent_info in info.items():
        if 'winner' in agent_info:
            return i
    if draw:
        return None

def env_name_to_canonical(env_name):
    env_aliases = {
        "multicomp/SumoHumansAutoContact-v0": "multicomp/SumoHumans-v0",
        "multicomp/SumoAntsAutoContact-v0": "multicomp/SumoAnts-v0",
    }
    env_name = env_aliases.get(env_name, env_name)
    env_prefix, env_suffix = env_name.split("/")
    if env_prefix != "multicomp":
        raise ValueError(f"Unsupported env '{env_name}'; must start with multicomp")
    return env_suffix

def is_symmetric(env_name):
    return SYMMETRIC_ENV[env_name_to_canonical(env_name)]
