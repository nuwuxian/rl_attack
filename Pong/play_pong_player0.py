SEED = 2999
import random
random.seed(SEED)
import os, sys, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
np.random.seed(SEED)
import gym
import roboschool
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy
from stable_baselines import PPO1, PPO2

from pposgd_hua import PPO1_hua_model_value

from value import MlpLstmValue, MlpValue

from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv, VecVideoRecorder
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
import time
import argparse
from shutil import copyfile
import traceback
import pickle as pkl
import multiprocessing
import pandas as pd

global player_n
# define the play_n = 1
player_n = 0

from datetime import datetime
from logger import setup_logger
from stable_baselines import logger
import json

now = datetime.now()
date_time = now.strftime("%m%d%Y-%H%M%S")


TRAINING_ITER = 4000000
USE_VIC = False

def make_dirs(dir_dict):
    for key, value in dir_dict.items():
        if key.startswith("_"):
            continue
        os.makedirs(value, exist_ok=True)
    f = open("{}agent.config".format(dir_dict["log"]),"w")
    f.write( str(dir_dict) )
    f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='init')
    parser.add_argument("--server", type=str, default='pongdemo_test')
    parser.add_argument("-pretrain", action='store_true', default=False)
    parser.add_argument("--mod", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="ppo1")
    parser.add_argument("--hyper_index", type=int, default=0)
    parser.add_argument("--player_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()

def callback(_locals, _globals):
    copyfile("/tmp/monitor/{0}/{1}/monitor.csv".format(dir_dict['_hyper_weights_index'],dir_dict['_player_index']),
             "{0}monitor.csv".format(dir_dict['log'])
    )

    path = "{0}monitor.csv".format(dir_dict['log'])
    data = pd.read_csv("{}".format(path), skiprows=[0], header=0)
    # data = data.get_chunk()
    data['score_board'] = data['score_board'].replace({'\'': '"'}, regex=True)

    data_score = pd.io.json.json_normalize(data['score_board'].apply(json.loads))

    data_score['total_round'] = data_score[['left.oppo_double_hit', 'left.oppo_miss_catch',
                                            'left.oppo_slow_ball',
                                            'right.oppo_double_hit', 'right.oppo_miss_catch',
                                            'right.oppo_slow_ball']].abs().sum(axis=1)

    data_score_next = data_score.shift(periods=1)

    data_score_epoch = data_score - data_score_next

    data_score_epoch['left_winning'] = data_score_epoch[
        ['left.oppo_double_hit', 'left.oppo_miss_catch',  # 'left.oppo_miss_start',
         'left.oppo_slow_ball']].abs().sum(axis=1)
    data_score_epoch['tie_winning'] = data_score_epoch['left.not_finish'].abs()

    data_score_epoch['left_winning'] = data_score_epoch['left_winning'] + data_score_epoch['tie_winning']
    data_score_epoch['total_round'] += data_score_epoch['left.not_finish'].abs()
    wining_rate_sum = data_score_epoch['left_winning'].rolling(1000, min_periods=50).sum()
    total_round_sum = data_score_epoch['total_round'].rolling(1000, min_periods=50).sum()

    wining_rate = wining_rate_sum / total_round_sum
    result = pd.concat([wining_rate], names=['winning_rate'], axis=1)
    logger.logkv('wining_rate', result.values[-1,0])

def advlearn(env, model_name=None, dir_dict=None):

    # assert model_name == 'ppo1'

    _, _ = setup_logger(SAVE_DIR, EXP_NAME)

    if model_name == 'ppo1_hua_oppomodel':
        model = PPO1_hua_model_value(MlpPolicy, env, timesteps_per_actorbatch=1000, verbose=1,
                         tensorboard_log=dir_dict['tb'], hyper_weights=dir_dict['_hyper_weights'],
                         benigned_model_file=None, full_tensorboard_log=False,
                         black_box_att=dir_dict['_black_box'], attention_weights=dir_dict['_attention'],
                         model_saved_loc=dir_dict['model'], clipped_attention=dir_dict['_clipped_attention'])
    else:
        model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=1000, verbose=1,
                     tensorboard_log=dir_dict['tb'])

    try:
        model.learn(TRAINING_ITER, callback=callback, seed=dir_dict['_seed'], use_victim_ob=USE_VIC)
    except ValueError as e:
        traceback.print_exc()
        print("Learn exit!")
    model_file_name = "{0}agent.pkl".format(dir_dict['model'])
    model.save(model_file_name)

def advtrain(server_id, model_name="ppo1", dir_dict=None):

    env = gym.make("RoboschoolPong-v1")
    env.seed(SEED)
    env.unwrapped.multiplayer(env, game_server_guid=server_id, player_n=dir_dict["_player_index"])
    # Only support PPO2 and num cpu is 1

    if "ppo1" in model_name:
        n_cpu = 1
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
    else:
        raise NotImplementedError
    advlearn(env, model_name=model_name, dir_dict=dir_dict)

args = parse_args()
memo = args.memo
server_id = args.server
mode = args.mod
model_name = args.model_name.lower()
hyper_weights_index = args.hyper_index
player_index = args.player_index

hyper_weights = [0.0, -0.1, 0.0, 1, 0, 10, True, True, False]


dir_dict= {
    "tb": "Log/{}-{}/tb/".format(memo,date_time),
    "model": "Log/{}-{}/model/".format(memo, date_time),
    "log": "Log/{}-{}/".format(memo,date_time),
    "_hyper_weights_index": hyper_weights_index,
    "_video": False,
    "_player_index": player_index,

    ## whether black box or attention
    "_black_box": hyper_weights[-3],
    "_attention": hyper_weights[-2],
    "_clipped_attention": hyper_weights[-1],
    "_seed": args.seed,
    "_n_steps": args.n_steps,
}

SAVE_DIR = './agent_zoo/'+ "Pong_" + str(args.vic_coef_init) + '_' + args.vic_coef_sch + \
    '_' + str(args.adv_coef_init) + '_' + args.adv_coef_sch +  \
    '_' + str(args.diff_coef_init) + '_' + args.diff_coef_sch

EXP_NAME = str(args.seed)

if mode == "advtrain":
    make_dirs(dir_dict)
    advtrain(server_id, model_name=model_name, dir_dict=dir_dict)
