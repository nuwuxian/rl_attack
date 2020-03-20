import os, sys, subprocess
import numpy as np
import gym
from gym import wrappers
import roboschool
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv, VecVideoRecorder
import pickle as pkl
from copy import deepcopy

def play(env, pi, pickle_file=None):
    while 1:
        obs = env.reset()
        while 1:
            old_obs = deepcopy(obs)
            a = pi.act(obs)
            obs, rew, done, info = env.step(a)

            if pickle_file is not None:
                pkl.dump([old_obs, a, rew, done], pickle_file,protocol=2)
            if done:
                break
        break

def test():
    env = gym.make("RoboschoolPong-v1")
    player_n = 1
    env = wrappers.Monitor(env,'Log/videos/player1/',video_callable=lambda episode_id: True, force=True)
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)
    env = DummyVecEnv([lambda: env])
    print("player 2 running in Env {0}".format(sys.argv[1]))

    from RoboschoolPong_v0_2017may2 import SmallReactivePolicy as Pol2
    pi = Pol2(env.observation_space, env.action_space)
    play(env, pi)

if sys.argv[2] == "test":
    test()
