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
from environment import make_zoo_multi2single_env, Monitor
from logger import setup_logger
from ppo2_wrap import MyPPO2
from common import get_zoo_path


model_dir = '../agent-zoo'
rew_shape_params = {'weights': {'dense': {'reward_move': 0.1}, \
                    'sparse': {'reward_remaining': 0.01}}, 'anneal_frac': 0}



## inline parameters
## include gamma, training_iter, ent_coef, noptepoches, learning_rate, n_steps.
## For Fair comparsion, we set these parameters the same as the ICLR 2020 paper "Adversarial Policies: Attacking Deep Reinforcement Learning"

#  param gamma: (float) Discount factor
#  param traning_iter: (int) Training iterations.
#  param n_steps: (int) The number of steps to run for each environment per update
#  (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
#  param ent_coef: (float) Entropy coefficient for the loss calculation
#  param learning_rate: (float or callable) The learning rate, it can be a function
#  param nminibatches: (int) Number of training minibatches per update
#  param noptepochs: (int) Number of epoch when optimizing
#  param n_cpu: number of environments copies running in parallel
#  param log_interval: The training interval to print the winning/losing information


gamma = 0.99
training_iter = 20000000
ent_coef = 0.00
nminibatches = 4
noptepochs = 4
learning_rate = 3e-4
n_steps = 2048


callback_key = 'update'
callback_mul = 16384
log_interval = 2048

n_cpu = 8


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
        parser.add_argument("--ratio", type=int, default=1)
        parser.add_argument("--x_method", type=str, default="grad")
        parser.add_argument('--root_dir', type=str, default="../agent-zoo")
        # surrogate model
        parser.add_argument('--surrogate_model', type=str, default="../agent-zoo/agent/YouShallNotPass_agent.pkl")
        parser.add_argument('--mimic_model_path', type=str, default="../agent-zoo/agent/mimic_model.h5")

        parser.add_argument('--exp_name', type=str, default="ppo2")
        args = parser.parse_args()

        scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
        env_name = env_list[args.env]
        # define the env_path
        env_path = args.surrogate_model
        mimic_model_path = args.mimic_model_path
        env = gym.make(env_name)
        venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(n_cpu)])
        venv = Monitor(venv, 1)

        rew_shape_venv = apply_reward_wrapper(single_env=venv, scheduler=scheduler,
                                              agent_idx=0, shaping_params=rew_shape_params)
        venv = VecNormalize(rew_shape_venv)
        # makedir output
        out_dir, logger = setup_logger(args.root_dir, args.exp_name)
        model = MyPPO2(MlpPolicy,
                       venv,
                       ent_coef=ent_coef,  nminibatches=nminibatches, noptepochs=noptepochs, 
                       learning_rate=learning_rate, verbose=1,  n_steps=n_steps, gamma=gamma, 
                       tensorboard_log=out_dir, model_saved_loc=out_dir, env_name=env_name, 
                       env_path=env_path, mimic_model_path=mimic_model_path,
                       mix_ratio=args.ratio, exp_method=args.x_method)
        
        Adv_train(venv, training_iter, callback_key, callback_mul, logger)
        model.save(os.path.join(args.root_dir, env_name.split('/')[1]))

