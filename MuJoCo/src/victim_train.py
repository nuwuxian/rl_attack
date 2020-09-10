import os
import argparse
import gym
from gym import Wrapper
import gym_compete
from common import env_list
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
from stable_baselines import PPO2
import tensorflow as tf
from environment import make_adv_multi2single_env, Monitor
from logger import setup_logger
from ppo2_wrap import MyPPO2
from common import get_zoo_path
import pdb


model_dir = '../agent-zoo'
rew_shape_params = {'weights': {'dense': {'reward_move': 0.1}, \
                    'sparse': {'reward_remaining': 0.01}}, 'anneal_frac': 0}

gamma = 0.99
training_iter = 20000000
ent_coef = 0.00
nminibatches = 4
noptepochs = 4
learning_rate = 3e-4
n_steps = 2048
checkpoint_step = 1000000
test_episodes = 100


callback_key = 'update'
callback_mul = 16384
log_interval = 2048

n_cpu = 8
pretrain_template = "../agent-zoo/%s-pretrained-expert-1000-1000-1e-03.pkl"

def Adv_train(env, total_timesteps, callback_key, callback_mul, logger):
    # log_callback
    log_callback = lambda logger, locals, globals: env.log_callback(logger)

    last_log = 0

    def callback(locals, globals):
        nonlocal last_log
        step = locals[callback_key] * callback_mul
        if step - log_interval > last_log:
            log_callback(logger, locals, globals)
            last_log = step

        return True

    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callback)

if __name__=="__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("env", type=int, default=2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--load", type=int, default=0)
        parser.add_argument("--render", type=int, default=0)
        parser.add_argument("--reverse", type=int, default=0)
        parser.add_argument("--ratio", type=int, default=0)
        parser.add_argument('--root_dir', type=str, default="../agent-zoo")
        parser.add_argument('--exp_name', type=str, default="ppo2")
        parser.add_argument('--adv_agent_path', type=str, default='../adv_agent/model.pkl')
        parser.add_argument('--adv_agent_norm_path', type=str, default='../adv_agent/obs_rms.pkl')

        args = parser.parse_args()
        adv_agent_path = args.adv_agent_path
        adv_agent_norm_path = args.adv_agent_norm_path

        scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
        env_name = env_list[args.env]
        # define the env_path
        env_path = get_zoo_path(env_name, tag=2)
        env = gym.make(env_name)
        venv = SubprocVecEnv([lambda: make_adv_multi2single_env(env_name, adv_agent_path, adv_agent_norm_path, False) for i in range(n_cpu)])
        venv = Monitor(venv, 0)

        rew_shape_venv = apply_reward_wrapper(single_env=venv, scheduler=scheduler,
                                              agent_idx=0, shaping_params=rew_shape_params)
        venv = VecNormalize(rew_shape_venv, norm_obs=False)
        # makedir output
        out_dir, logger = setup_logger(args.root_dir, args.exp_name)
        model = MyPPO2(MlpPolicy,
                       venv,
                       ent_coef=ent_coef,
                       nminibatches=nminibatches, noptepochs=noptepochs,
                       learning_rate=learning_rate,  verbose=1,
                       n_steps=n_steps, gamma=gamma, tensorboard_log=out_dir,
                       model_saved_loc=out_dir, env_name=env_name, env_path=env_path, 
                       mix_ratio=args.ratio, retrain_victim=True, norm_victim=True) # , rl_path=rl_path, var_path=var_path)
        
        Adv_train(venv, training_iter, callback_key, callback_mul, logger)
        model.save(os.path.join(args.root_dir, env_name.split('/')[1]))

