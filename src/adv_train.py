import os
import argparse
import gym
from gym import Wrapper
import gym_compete
from common import env_list
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
from stable_baselines import PPO2
from ppo2_wrap import MyPPO2
import tensorflow as tf
from environment import make_zoo_multi2single_env, Monitor
from logger import setup_logger
from common import get_zoo_path

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
        parser.add_argument("env", type=int)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--load", type=int, default=0)
        parser.add_argument("--render", type=int, default=0)
        parser.add_argument("--reverse", type=int, default=0)
        parser.add_argument('--root_dir', type=str, default="../agent-zoo")
        parser.add_argument('--exp_name', type=str, default="ppo2")
        args = parser.parse_args()

        scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
        env_name = env_list[args.env]
        env = gym.make(env_name)
        venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(n_cpu)])
        venv = Monitor(venv)
        rew_shape_venv = apply_reward_wrapper(single_env=venv, scheduler=scheduler,
                                              agent_idx=0, shaping_params=rew_shape_params)
        venv = VecNormalize(rew_shape_venv)

        out_dir, logger = setup_logger(args.root_dir, args.exp_name)
        env_path = get_zoo_path(env_name, tag=2)

        model = MyPPO2(MlpPolicy, venv,
                       ent_coef=ent_coef,
                       nminibatches=nminibatches,
                       noptepochs=noptepochs,
                       learning_rate=learning_rate,
                       n_steps=n_steps,
                       gamma=gamma,
                       verbose=1, env_path=env_path)

        '''
        if args.load == 0:
            model = PPO2(MlpPolicy, 
                         venv, 
                         ent_coef=ent_coef,
                         nminibatches=nminibatches,
                         noptepochs=noptepochs,
                         learning_rate=learning_rate,
                         verbose=1,
                         n_steps=n_steps,
                         gamma=gamma)
                         # seed=args.seed,
                         # n_cpu_tf_sess=1)
        else:
            model = PPO2.load(pretrain_template%(env_name.split("/")[1]), 
                              venv, 
                              gamma=gamma,
                              ent_coef=ent_coef, 
                              nminibatches=nminibatches, 
                              learning_rate=learning_rate,
                              n_steps=n_steps)
        '''
        Adv_train(venv, training_iter, callback_key, callback_mul, logger)
        model.save(os.path.join(args.root_dir, env_name.split('/')[1]))
