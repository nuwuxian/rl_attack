import warnings
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper
import argparse
import gym
from agent import make_zoo_agent, make_trigger_agent, make_victim_agent
from environment import make_multi2single_env, make_zoo_multi2single_env, make_backdoor_env
import os
from gym import spaces
from common import env_list

dataset_template = '../expert_dataset/%s_%s_%d'

def generate_traj(model, save_path=None, env=None, n_episodes=100, backdoor=False):

    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    actions = []
    observations = []
    dones = []

    ep_idx = 0
    obs = env.reset()
    idx = 0

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]

    while ep_idx < n_episodes:
        print(ep_idx)
        if backdoor:
             observations.append(obs[env.action_space.shape[0]:])
        else:
            observations.append(obs)

        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask)
        else:
            action = model(obs)

        obs, _, done, _ = env.step(action)

        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            done = np.array([done[0]])

        actions.append(action)
        dones.append(done)
        idx += 1
        if done:
            if not is_vec_env:
                obs = env.reset()
                state = None
            ep_idx += 1

    if isinstance(env.observation_space, spaces.Box):
        if backdoor:
            observations = np.concatenate(observations).reshape((-1,) + (env.observation_space.shape[0]-env.action_space.shape[0], ))
        else:
            observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    dones = np.array(dones)

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'dones': dones,
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict

def mix_dataset(path_1, path_2, save_path):
    dataset_1 = np.load(path_1, allow_pickle=True)
    dataset_2 = np.load(path_2, allow_pickle=True)

    actions = np.append(dataset_1['actions'], dataset_2['actions'], axis=0)
    obs = np.append(dataset_1['obs'], dataset_2['obs'], axis=0)
    dones = np.append(dataset_1['dones'], dataset_2['dones'], axis=0)

    action_episodes = []
    observation_episodes = []
    done_episodes = []
    action_episode = []
    observation_episode = []
    done_episode = []
    for index in range(len(dones)):
        action_episode.append(actions[index])
        observation_episode.append(obs[index])
        done_episode.append(dones[index])
        if dones[index]:
            action_episodes.append(action_episode)
            observation_episodes.append(observation_episode)
            done_episodes.append(done_episode)
            action_episode = []
            observation_episode = []
            done_episode = []
         
    print(actions.shape)
    np.random.shuffle(action_episodes)
    np.random.shuffle(observation_episodes)
    np.random.shuffle(done_episodes) 
    actions = np.concatenate(action_episodes)
    observations = np.concatenate(observation_episodes)
    dones = np.concatenate(done_episodes)
    print(actions.shape)

    numpy_dict = {
        'actions': actions,
        'obs': obs,
        'dones': dones,
    }

    np.savez(save_path, **numpy_dict)

class ExpertDataset(object):

    def __init__(self, expert_path=None, n_env=8, train_fraction=0.7, traj_limitation=-1, verbose=1):

        traj_data = np.load(expert_path, allow_pickle=True)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)

        dones = np.array(traj_data['dones'])
        if isinstance(dones[0], np.ndarray):
            dones = [v if isinstance(v, bool) else v[0] for v in dones]

        if traj_limitation <= 0:
            traj_limitation = np.sum(dones)

        n_episodes = 0
        for idx, done in enumerate(dones):
            n_episodes += int(done)
            if n_episodes == int(traj_limitation*train_fraction):
                train_limit_idx = idx
            if n_episodes == traj_limitation:
                val_limit_idx = idx
                break

        observations_train = traj_data['obs'][:train_limit_idx]
        observations_val = traj_data['obs'][train_limit_idx:val_limit_idx]
        actions_train = traj_data['actions'][:train_limit_idx]
        actions_val = traj_data['actions'][train_limit_idx:val_limit_idx]
        dones_train = dones[:train_limit_idx]
        dones_val = dones[train_limit_idx:val_limit_idx]

        # if len(observations_train.shape) > 2:
        #     observations_train = np.reshape(observations_train, [-1, np.prod(observations_train.shape[1:])])
        #     observations_val = np.reshape(observations_val, [-1, np.prod(observations_val.shape[1:])])
        # if len(actions_train.shape) > 2:
        #     actions_train = np.reshape(actions_train, [-1, np.prod(actions_train.shape[1:])])
        #     actions_val = np.reshape(actions_val, [-1, np.prod(actions_val.shape[1:])])

        self.observations_train = observations_train
        self.observations_val = observations_val
        self.actions_train = actions_train
        self.actions_val = actions_val
        self.dones_train = dones_train
        self.dones_val = dones_val

        self.verbose = verbose

        # assert len(self.observations) == len(self.actions), "The number of actions and observations differ " \
                                                            # "please check your expert dataset"

        self.train_num_traj = int(traj_limitation*train_fraction)
        self.val_num_traj = traj_limitation-self.train_num_traj
        # self.randomize = randomize

        self.start_idx = 0

        self.n_env = n_env
        env_train_num_traj = max(self.train_num_traj // n_env, 1)
        env_val_num_traj = max(self.val_num_traj // n_env, 1)
        self.env_ob_train_traj = []
        self.env_ob_val_traj = []
        self.env_action_train_traj = []
        self.env_action_val_traj = []
        self.env_done_train_traj = []
        self.env_done_val_traj = []
        for _ in range(n_env):
            self.env_ob_train_traj.append([])
            self.env_ob_val_traj.append([])
            self.env_action_train_traj.append([])
            self.env_action_val_traj.append([])
            self.env_done_train_traj.append([])
            self.env_done_val_traj.append([])

        ep_idx = 0
        for idx, done in enumerate(self.dones_train):
            env_idx = ep_idx // env_train_num_traj
            self.env_ob_train_traj[env_idx].append(self.observations_train[idx])
            self.env_action_train_traj[env_idx].append(self.actions_train[idx])
            self.env_done_train_traj[env_idx].append(done)
            ep_idx += int(done)
            if ep_idx == env_train_num_traj * n_env:
                break

        ep_idx = 0
        for idx, done in enumerate(self.dones_val):
            env_idx = ep_idx // env_val_num_traj
            self.env_ob_val_traj[env_idx].append(self.observations_val[idx])
            self.env_action_val_traj[env_idx].append(self.actions_val[idx])
            self.env_done_val_traj[env_idx].append(done)
            ep_idx += int(done)
            if ep_idx == env_val_num_traj * n_env:
                break

        train_traj_length = [len(x) for x in self.env_ob_train_traj]
        val_traj_length = [len(x) for x in self.env_ob_val_traj]
        self.max_train_traj_length = np.max(train_traj_length)
        self.max_val_traj_length = np.max(val_traj_length)
        self.train_idx = np.zeros(n_env, dtype=int)
        self.val_idx = np.zeros(n_env, dtype=int)
         
 
    def get_next_batch(self, split=None):

        if split is None or split == 'train':
            actions = []
            obs = []
            dones = []
            for i in range(self.n_env):
                actions.append(self.env_action_train_traj[i][self.train_idx[i]])
                obs.append(self.env_ob_train_traj[i][self.train_idx[i]])
                dones.append(self.env_done_train_traj[i][self.train_idx[i]])
                self.train_idx[i] = (self.train_idx[i] + 1) % len(self.env_ob_train_traj[i])

            actions = np.array(actions)
            obs = np.array(obs)
            dones = np.array(dones)

            return obs, actions, dones
        elif split == 'val':
            actions = []
            obs = []
            dones = []
            for i in range(self.n_env):
                actions.append(self.env_action_val_traj[i][self.val_idx[i]])
                obs.append(self.env_ob_val_traj[i][self.val_idx[i]])
                dones.append(self.env_done_val_traj[i][self.val_idx[i]])
                self.val_idx[i] = (self.val_idx[i] + 1) % len(self.env_ob_val_traj[i])

            actions = np.array(actions)
            obs = np.array(obs)
            dones = np.array(dones)

            return obs, actions, dones
        else:
            raise NotImplementedError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_type", default='expert')
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--episode", type=int, default=1000)
    args = parser.parse_args()

    env_name = env_list[args.env]
    if args.traj_type == 'expert':
        env = make_zoo_multi2single_env(env_name)
        expert = make_zoo_agent(env_name, env.observation_space, env.action_space, tag=1, scope='expert')
        generate_traj(expert.act, dataset_template%(env_name.split('/')[1], 'expert', args.episode), env, n_episodes=args.episode)
    elif args.traj_type == 'trigger':
        env = make_backdoor_env(env_name, end=20)
        agent = make_victim_agent(env_name, env.observation_space, env.action_space)
        generate_traj(agent.act, dataset_template%(env_name.split('/')[1], 'trigger', args.episode), env, n_episodes=args.episode, backdoor=True)
    elif args.traj_type == 'adv':
        pass
    elif args.traj_type == 'mix':
        mix_dataset(dataset_template%(env_name.split('/')[1], 'expert', args.episode)+'.npz', dataset_template%(env_name.split('/')[1], 'dummy', args.episode)+'.npz', dataset_template%(env_name.split('/')[1], 'mixed', 2*args.episode))
    elif args.traj_type == 'test':
        dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1], 'expert', args.episode)+".npz", train_fraction=0.8)
        obs, actions, dones = dataset.get_next_batch()
    else:
        raise NotImplementedError

    # dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1], 'expert', args.episode)+".npz", train_fraction=0.8)
    # obs, actions, dones = dataset.get_next_batch()
