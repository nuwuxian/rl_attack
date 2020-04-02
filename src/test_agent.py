from zoo_utils import LSTMPolicy, MlpPolicyValue
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
import gym
import gym_compete
import pickle
import sys
import argparse
import tensorflow as tf
import numpy as np
from common import env_list
from gym import wrappers

from stable_baselines.common.running_mean_std import RunningMeanStd
from zoo_utils import setFromFlat, load_from_file, load_from_model

def run(config):
    ENV_NAME = env_list[config.env]

    if ENV_NAME in ['multicomp/YouShallNotPassHumans-v0', "multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0"]:
        policy_type="mlp"
    else:
        policy_type="lstm"

    env = gym.make(ENV_NAME)

    epsilon = config.epsilon
    clip_obs = config.clip_obs

    param_paths = [config.opp_path, config.vic_path]

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    retrain_id = config.retrain_id

    policy = []

    policy.append(MlpPolicy(sess, env.observation_space.spaces[0], env.action_space.spaces[0],
                            1, 1, 1, reuse=False))

    policy.append(MlpPolicyValue(scope="policy1", reuse=False,
                                         ob_space=env.observation_space.spaces[1],
                                         ac_space=env.action_space.spaces[1],
                                         hiddens=[64, 64], normalize=True))

    # initialize uninitialized variables
    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(2):
        if i == retrain_id:
            param = load_from_model(param_pkl_path=param_paths[i])
            adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            setFromFlat(adv_agent_variables, param)
        else:
            param = load_from_file(param_pkl_path=param_paths[i])
            setFromFlat(policy[i].get_variables(), param)


    max_episodes = config.max_episodes
    num_episodes = 0
    nstep = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]

    # norm path:
    obs_rms = load_from_file(config.norm_path)


    # total_scores = np.asarray(total_scores)
    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)
    while num_episodes < max_episodes:
        # normalize the observation-0 and observation-1
        obs_0, obs_1 = observation
        obs_0 = np.clip(
            (obs_0 - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
            -10, 10)

        action_0 = policy[0].step(obs=obs_0[None,:], deterministic=False)[0][0]
        action_1 = policy[1].act(stochastic=True, observation=obs_1)[0]
        action = (action_0, action_1)

        observation, reward, done, infos = env.step(action)
        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        if done[0]:
            num_episodes += 1
            draw = True

            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            if draw:
                print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            observation = env.reset()
            nstep = 0
            total_reward = [0.0  for _ in range(len(policy))]

            if num_episodes < max_episodes:
                print("-"*5 + "Episode %d" % (num_episodes+1) + "-"*5)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default=2, type=int)
    p.add_argument("--retrain_id", default=0, type=int)
    p.add_argument("--opp-path", default="/home/xkw5132/rl_attack/agent-zoo/20200329_222037-ppo2/checkpoints/000019955712/model.pkl", type=str)
    p.add_argument("--vic_path", default="/home/xkw5132/rl_attack/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str)
    p.add_argument("--norm_path", default="/home/xkw5132/rl_attack/agent-zoo/20200329_222037-ppo2/checkpoints/000019955712/obs_rms.pkl", type=str)

    p.add_argument("--max-episodes", default=200, help="max number of matches", type=int)
    p.add_argument("--epsilon", default=1e-8, type=float)
    p.add_argument("--clip_obs", default=10, type=float)

    config = p.parse_args()
    run(config)






    