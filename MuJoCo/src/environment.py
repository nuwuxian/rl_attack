import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper, RewardWrapper

import random

from stable_baselines.common.vec_env import VecEnvWrapper
from common import trigger_map
from agent import make_zoo_agent, make_adv_agent
# from agent import make_zoo_agent, make_trigger_agent
from collections import Counter
# running-mean std
from stable_baselines.common.running_mean_std import RunningMeanStd

def func(x):
  if type(x) == np.ndarray:
    return x[0]
  else:
    return x


# norm-agent
class Monitor(VecEnvWrapper):
    def __init__(self, venv, agent_idx):
        """ Got game results.
        :param: venv: environment.
        :param: agent_idx: the index of victim agent.
        """
        VecEnvWrapper.__init__(self, venv)
        self.outcomes = []
        self.num_games = 0
        self.agent_idx = agent_idx

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        """ get the needed information of adversarial agent.
        :return: obs: observation of the next step. n_environment * observation dimensionality
        :return: rew: reward of the next step.
        :return: dones: flag of whether the game finished or not.
        :return: infos: winning information.
        """
        obs, rew, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done:
                if 'winner' in info:
                    self.outcomes.append(1 - self.agent_idx)
                elif 'loser' in info:
                    self.outcomes.append(self.agent_idx)
                else:
                    self.outcomes.append(None)
                self.num_games += 1

        return obs, rew, dones, infos

    def log_callback(self, logger):
        """ compute winning rate.
        :param: logger: record of log.
        """
        c = Counter()
        c.update(self.outcomes)
        num_games = self.num_games
        if num_games > 0:
            logger.logkv("game_win0", c.get(0, 0) / num_games) # agent 0 winning rate.
            logger.logkv("game_win1", c.get(1, 0) / num_games) # agent 1 winning rate.
            logger.logkv("game_tie", c.get(None, 0) / num_games) # tie rate.
        logger.logkv("game_total", num_games)
        self.num_games = 0
        self.outcomes = []


class Multi_Monitor(VecEnvWrapper):
    def __init__(self, venv, agent_idx):
        """ Got game results.
        :param: venv: environment.
        :param: agent_idx: the index of victim agent.
        """
        VecEnvWrapper.__init__(self, venv)
        self.outcomes = []
        # adv_outcomes
        self.adv_outcomes = []

        self.num_games = 0
        self.agent_idx = agent_idx

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        """ get the needed information of adversarial agent.
        :return: obs: observation of the next step. n_environment * observation dimensionality
        :return: rew: reward of the next step.
        :return: dones: flag of whether the game finished or not.
        :return: infos: winning information.
        """
        obs, rew, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done:
                if 'winner' in info:
                    self.outcomes.append(1 - self.agent_idx)
                    if 'adv_agent' in info:
                        self.adv_outcomes.append(1 - self.agent_idx)
                elif 'loser' in info:
                    self.outcomes.append(self.agent_idx)
                    if 'adv_agent' in info:
                        self.adv_outcomes.append(self.agent_idx)
                else:
                    self.outcomes.append(None)
                    if 'adv_agent' in info:
                        self.adv_outcomes.append(None)
                self.num_games += 1

        return obs, rew, dones, infos

    def log_callback(self, logger):
        """ compute winning rate.
        :param: logger: record of log.
        """
        c = Counter()
        c.update(self.outcomes)
        num_games = self.num_games

        adv_c = Counter()
        adv_c.update(self.adv_outcomes)
        adv_num_games = adv_c.get(0, 0) + adv_c.get(1, 0) + adv_c.get(None, 0)
        norm_num_games = num_games - adv_num_games

        if num_games > 0:
            logger.logkv("game_win0", c.get(0, 0) / num_games)  # agent 0 winning rate.
            logger.logkv("game_win1", c.get(1, 0) / num_games)  # agent 1 winning rate.
            logger.logkv("game_tie", c.get(None, 0) / num_games)  # tie rate.
        # play with adv-agent
        if adv_num_games > 0:
            logger.logkv("game_adv_win0", adv_c.get(0, 0) / adv_num_games)
            logger.logkv("game_adv_win1", adv_c.get(1, 0) / adv_num_games)
            logger.logkv("game_adv_tie", adv_c.get(None, 0) / adv_num_games)
        # play with norm-agent
        if norm_num_games > 0:
            logger.logkv("game_norm_win0", (c.get(0, 0) - adv_c.get(0, 0)) / norm_num_games)
            logger.logkv("game_norm_win1", (c.get(1, 0) - adv_c.get(1, 0)) / norm_num_games)
            logger.logkv("game_norm_tie", (c.get(None, 0) - adv_c.get(None, 0)) / norm_num_games)

        logger.logkv("game_total", num_games)
        logger.logkv("adv_game_total", adv_num_games)
        self.num_games = 0
        self.outcomes = []
        self.adv_outcomes = []


class Multi2SingleEnv(Wrapper):

    def __init__(self, env, env_name, agent, agent_idx, shaping_params, scheduler, total_step, norm=True,
                 retrain_victim=False, clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8,
                 mix_agent=False, mix_ratio=0.5, _agent=None):

        """ from multi-agent environment to single-agent environment.
        :param: env: two-agent environment.
        :param: agent: victim agent.
        :param: agent_idx: victim agent index.
        :param: shaping_params: shaping parameters.
        :param: scheduler: anneal scheduler.
        :param: norm: normalize agent or not.
        :param: retrain_victim: retrain victim agent or not.
        :param: clip_obs: observation clip value.
        :param: clip_rewards: reward clip value.
        :param: gamma: discount factor.
        :param: epsilon: additive coefficient.
        """
        Wrapper.__init__(self, env)
        self.env_name = env_name
        self.agent = agent
        self.reward = 0
        # observation dimensionality
        self.observation_space = env.observation_space.spaces[0]
        # action dimensionality
        self.action_space = env.action_space.spaces[0]
        self.total_step = total_step

        # normalize the victim agent's obs and rets
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.obs_rms_next = RunningMeanStd(shape=self.observation_space.shape)

        self.ret_rms = RunningMeanStd(shape=())
        self.ret_abs_rms = RunningMeanStd(shape=())

        self.done = False
        self.mix_agent = mix_agent
        self.mix_ratio = mix_ratio

        self._agent = _agent
        # determine which policy norm|adv
        self.is_advagent = True

        # time step count
        self.cnt = 0
        self.agent_idx = agent_idx
        self.norm = norm
        self.retrain_victim = retrain_victim

        self.shaping_params = shaping_params
        self.scheduler = scheduler

        # set normalize hyper
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward

        self.gamma = gamma
        self.epsilon = epsilon

        self.num_agents = 2
        self.outcomes = []

        # return - total discounted reward.
        self.ret = np.zeros(1)
        self.ret_abs = np.zeros(1)

    def step(self, action):
        """get the reward, observation, and information at each step.
        :param: action: action of adversarial agent at this time.
        :return: obs: adversarial agent observation of the next step.
        :return: rew: adversarial agent reward of the next step.
        :return: dones: adversarial agent flag of whether the game finished or not.
        :return: infos: adversarial agent winning information.
        """

        self.cnt += 1
        self.oppo_ob = self.ob.copy()
        self.obs_rms.update(self.oppo_ob)
        self.oppo_ob = np.clip((self.oppo_ob - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                               -self.clip_obs, self.clip_obs)
        if self.retrain_victim:
            if not self.agent.adv_loadnorm:
                self_action = self.agent.act(observation=self.oppo_ob[None, :], reward=self.reward, done=self.done).flatten()
            else:
                self_action = self.agent.act(observation=self.ob[None,:], reward=self.reward, done=self.done).flatten()
            # mix agent
            if self.mix_agent and not self.is_advagent:
                self_action = self._agent.act(observation=self.ob, reward=self.reward, done=self.done)
        else:
            self_action = self.agent.act(observation=self.ob, reward=self.reward, done=self.done)
        # note: current observation
        self.action = self_action

        # combine agents' actions
        if self.agent_idx == 0:
            actions = (self_action, action)
        else:
            actions = (action, self_action)
            
        # obtain needed information from the environment.
        obs, rewards, dones, infos = self.env.step(actions)

        if dones[0] and 'Ant' in self.env_name:
            if infos[0]['reward_remaining']==0:
                infos[0]['reward_remaining'] = -1000
            if infos[1]['reward_remaining']==0:
                infos[1]['reward_remaining'] = -1000

        # separate victim and adversarial information.
        if self.agent_idx == 0: # vic is 0; adv is 1
          self.ob, ob = obs
          self.reward, reward = rewards
          self.done, done = dones
          self.info, info = infos
        else: # vic is 1; adv is 0
          ob, self.ob = obs
          reward, self.reward = rewards
          done, self.done = dones
          info, self.info = infos
        done = func(done)

        self.oppo_ob_next = self.ob.copy()
        self.obs_rms_next.update(self.oppo_ob_next)
        self.oppo_ob_next = np.clip((self.oppo_ob_next - self.obs_rms_next.mean) / np.sqrt(self.obs_rms_next.var + self.epsilon),
                                    -self.clip_obs, self.clip_obs)

        # Save and normalize the victim observation and return.
        # self.oppo_reward = self.reward
        # self.oppo_reward = -1.0 * self.info['reward_remaining'] * 0.01
        # self.abs_reward =  info['reward_remaining'] * 0.01 - self.info['reward_remaining'] * 0.01

        frac_remaining = max(1 - self.cnt / self.total_step, 0)

        self.oppo_reward = apply_reward_shapping(self.info, self.shaping_params, self.scheduler, frac_remaining)
        self.abs_reward = apply_reward_shapping(info, self.shaping_params, self.scheduler, frac_remaining)
        self.abs_reward = self.abs_reward - self.oppo_reward

        if self.norm:
            self.ret = self.ret * self.gamma + self.oppo_reward
            self.ret_abs = self.ret_abs * self.gamma + self.abs_reward
            self.oppo_reward, self.abs_reward = self._normalize_(self.ret, self.ret_abs,
                                                                 self.oppo_reward, self.abs_reward)
            if self.done:
                self.ret[0] = 0
                self.ret_abs[0] = 0

        if done:
            if 'winner' in self.info: # opponent (the agent that is not being trained) win.
                info['loser'] = True
            if self.is_advagent and self.retrain_victim: # Number of adversarial agent trajectories
                info['adv_agent'] = True
        return ob, reward, done, info

    def _normalize_(self, ret, ret_abs, reward, abs_reward):
        """
        :param: obs: observation.
        :param: ret: return.
        :param: reward: reward.
        :return: obs: normalized and cliped observation.
        :return: reward: normalized and cliped reward.
        """
        self.ret_rms.update(ret)
        reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        # update the ret_abs
        self.ret_abs_rms.update(ret_abs)
        abs_reward = np.clip(abs_reward / np.sqrt(self.ret_abs_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

        return reward, abs_reward

    def reset(self):
        """reset everything.
        :return: ob: reset observation.
        """
        self.cnt = 0
        self.reward = 0
        self.done = False
        self.ret = np.zeros(1)
        self.ret_abs = np.zeros(1)
        # reset the agent
        # reset the h and c
        self.agent.reset()
        if self._agent != None:
            self._agent.reset()

        ## sampling from the mix-ratio
        ## mix-ratio adv_agent:norm_agent
        if self.mix_ratio == 0.5:
            self.is_advagent = not self.is_advagent
        else:
            self.is_advagent = (random.uniform(0, 1) < self.mix_ratio) # mix_ratio means the ratio of adv_agent

        if self.agent_idx == 1:
            ob, self.ob = self.env.reset()
        else:
            self.ob, ob = self.env.reset()
        return ob


def make_zoo_multi2single_env(env_name, version, shaping_params, scheduler, total_step, reverse=True):

    # Specify the victim party.
    if 'You' in env_name.split('/')[1]:
        tag = 2
    else:
        tag = 1

    env = gym.make(env_name)

    # if 'Kick' in env_name.split('/')[1]:
    #     env._max_episode_steps = 1500
    zoo_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1],
                               tag=tag, version=version)

    return Multi2SingleEnv(env, env_name, zoo_agent, reverse, shaping_params, scheduler, total_step=total_step)


# make combine agents
def make_mixadv_multi2single_env(env_name, version, adv_agent_path, adv_agent_norm_path, shaping_params, scheduler,
                                 adv_ismlp, total_step, n_envs=1, reverse=True, ratio=0.5):
    env = gym.make(env_name)

    # specify the normal opponent of the victim agent.
    if 'You' in env_name.split('/')[1]:
        tag = 1
    else:
        tag = 2

    opp_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1],
                               tag=tag, version=version)
    adv_agent = make_adv_agent(env.observation_space.spaces[1], env.action_space.spaces[1], n_envs, adv_agent_path,
                               adv_ismlp=adv_ismlp, adv_obs_normpath=adv_agent_norm_path)

    return Multi2SingleEnv(env, env_name, adv_agent, agent_idx=reverse, shaping_params=shaping_params,
                           scheduler=scheduler, retrain_victim=True, mix_agent=True,
                           mix_ratio=ratio, _agent=opp_agent, total_step=total_step)


from scheduling import ConditionalAnnealer, ConstantAnnealer, LinearAnnealer
REW_TYPES = set(('sparse', 'dense'))


def apply_reward_shapping(infos, shaping_params, scheduler, frac_remaining):
    """ victim agent reward shaping function.
    :param: info: reward returned from the environment.
    :param: shaping_params: reward shaping parameters.
    :param: annealing factor decay schedule.
    :param: linear annealing fraction.
    :return: shaped reward.
    """
    if 'metric' in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(shaping_params, get_logs=None)
        scheduler.set_conditional('rew_shape')
    else:
        anneal_frac = shaping_params.get('anneal_frac')
        if shaping_params.get('anneal_type')==0:
            rew_shape_annealer = ConstantAnnealer(anneal_frac)
        else:
            rew_shape_annealer = LinearAnnealer(1, 0, anneal_frac)

    scheduler.set_annealer('rew_shape', rew_shape_annealer)
    reward_annealer = scheduler.get_annealer('rew_shape')
    shaping_params = shaping_params['weights']

    assert shaping_params.keys() == REW_TYPES
    new_shaping_params = {}

    for rew_type, params in shaping_params.items():
        for rew_term, weight in params.items():
            new_shaping_params[rew_term] = (rew_type, weight)

    shaped_reward = {k: 0 for k in REW_TYPES}
    for rew_term, rew_value in infos.items():
        if rew_term not in new_shaping_params:
            continue
        rew_type, weight = new_shaping_params[rew_term]
        shaped_reward[rew_type] += weight * rew_value

    # Compute total shaped reward, optionally annealing
    reward = _anneal(shaped_reward, reward_annealer, frac_remaining)
    return reward


def _anneal(reward_dict, reward_annealer, frac_remaining):
    c = reward_annealer(frac_remaining)
    assert 0 <= c <= 1
    sparse_weight = 1 - c
    dense_weight = c

    return (reward_dict['sparse'] * sparse_weight
            + reward_dict['dense'] * dense_weight)
