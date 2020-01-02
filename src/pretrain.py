import os
import argparse
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from environment import make_zoo_multi2single_env, make_trojan_env
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from common import env_list

dataset_template = '../expert_dataset/%s_trojan_2000_5e-01.npz'
# dataset_template = '../expert_dataset/%s_%s_%d.npz'
model_dir = "../agent-zoo"
learning_rate = 1e-3

n_env = 8
batch_size=64
nminibatches = 4
policy = 'MlpPolicy'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=int)
    parser.add_argument("--traj", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=1000)
    args = parser.parse_args()

    env_name = env_list[args.env]
    policy = 'MlpPolicy'
    # add flexible path
    # dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1], 'expert', 1000), batch_size=batch_size, randomize=False)
    dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1]), randomize=False)

    venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(n_env)])
    model = PPO2(policy, venv, verbose=1)
    model.pretrain(dataset, n_epochs=args.epoch, learning_rate=learning_rate, val_interval=100)
    # add flexible path
    model.save(os.path.join(model_dir, env_name.split('/')[1]+'-pretrained-trojan-'+str(args.traj)+'-'+str(args.epoch)+'-{:.0e}'.format(learning_rate)))
