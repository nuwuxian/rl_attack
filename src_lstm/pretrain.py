import os
import argparse
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from environment import make_zoo_multi2single_env
from stable_baselines import PPO2
from expert_dataset import ExpertDataset
from common import env_list

model_dir = "../agent-zoo"
model_template = "../agent-zoo/%s-%s-%s-%d-%d-%s.pkl"

n_env = 64
batch_size=n_env
nminibatches = n_env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=int)
    parser.add_argument("--policy", default="lstm")
    parser.add_argument("--dataset", default="expert")
    parser.add_argument("--pretrain", default="pretrained")
    parser.add_argument("--traj", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument("--load_traj", type=int, default=1000)
    parser.add_argument("--load_dataset", default='expert')
    parser.add_argument("--load_lr", type=float, default=1e-4)
    args = parser.parse_args()

    env_name = env_list[args.env]
    policy = 'MlpLstmPolicy'
    if args.policy == 'mlp':
        policy = 'MlpPolicy'

    dataset = ExpertDataset(expert_path="../expert_dataset/%s_%s_%d.npz"%(env_name.split('/')[1], args.dataset, args.traj), n_env=n_env, train_fraction=0.8, traj_limitation=args.traj)

    venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(n_env)])
    if args.load_epoch is 0:
        model = PPO2(policy, venv, nminibatches=nminibatches, verbose=1)
    else:
        print(model_template%(env_name.split('/')[1], args.pretrain, args.load_dataset, args.load_traj, args.load_epoch, '{:.0e}'.format(args.load_lr)))
        model = PPO2.load(model_template%(env_name.split('/')[1], args.pretrain, args.load_dataset, args.load_traj, args.load_epoch, '{:.0e}'.format(args.load_lr)), nminibatches=nminibatches, verbose=1)
        # model = PPO2.load("RunToGoalAnts-v0-tuned-mixed-2000-3100-1e-03.pkl")
    model.pretrain_lstm(dataset, n_epochs=args.epoch, learning_rate=args.lr, val_interval=None, fix_first_layer=True)

    if args.load_epoch is not 0:
        model.save(os.path.join(model_dir, env_name.split('/')[1]+'-tuned-'+args.dataset+'-'+str(args.traj)+'-'+str(args.epoch+args.load_epoch)+'-{:.0e}'.format(args.lr)))
    else:
        model.save(os.path.join(model_dir, env_name.split('/')[1]+'-pretrained-'+args.dataset+'-'+str(args.traj)+'-'+str(args.epoch+args.load_epoch)+'-{:.0e}'.format(args.lr)))
