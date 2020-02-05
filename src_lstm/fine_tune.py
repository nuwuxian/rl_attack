import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import gym
gym.logger.set_level(gym.logger.ERROR)
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from environment import make_zoo_multi2single_env, make_trigger_multi2single_env
from stable_baselines import PPO2
from expert_dataset import ExpertDataset
from common import env_list
import progressbar

dataset_template = "../expert_dataset/%s_%s_%d.npz"
model_dir = "../agent-zoo"
model_template = "../agent-zoo/%s-%s-%s-%d-%d-%s.pkl"

n_env = 256
batch_size = n_env
nminibatches = n_env

pretrain_iter = 1000
epoch = 100
max_episodes = 100
zoo_threshold = 30
trigger_threshold = 20

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="3")
    args = parser.parse_args()

    env_name = env_list[args.env]
    policy = 'MlpLstmPolicy'
    learning_rate = args.lr
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    expert_dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1], 'expert', 1000), n_env=batch_size, train_fraction=0.8, traj_limitation=1000)
    trigger_dataset = ExpertDataset(expert_path=dataset_template%(env_name.split('/')[1], 'mixed', 2000), n_env=batch_size, train_fraction=0.8, traj_limitation=2000)
    dataset = [expert_dataset, trigger_dataset]

    dataset_idx = 0
    win_zoo = 0
    win_trigger = 100
    fix_first_layer = False
    p = progressbar.ProgressBar()
    zoo_env = DummyVecEnv([lambda: make_zoo_multi2single_env(env_name)])
    trigger_env = DummyVecEnv([lambda: make_trigger_multi2single_env(env_name)])
    observation_space =zoo_env.observation_space
    action_space = zoo_env.action_space
    with tf.device('/GPU:0'):
        # venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(n_env)])
        # model = PPO2(policy, None, verbose=1, n_envs=n_env, observation_space=observation_space, action_space=action_space)
        model = PPO2.load("../agent-zoo/RunToGoalAnts-v0-pretrained-expert-1000-3000-1e-04.pkl", verbose=1)
    for idx in range(pretrain_iter):
        model.pretrain_lstm(dataset[dataset_idx], n_epochs=epoch, learning_rate=learning_rate, val_interval=None, fix_first_layer=fix_first_layer)
        # test against zoo
        print("Test against zoo")
        ob = zoo_env.reset()
        num_episodes = 0
        # total_tie = 0
        win_zoo = 0
        cnt = 0
        p.start(max_episodes)
        while num_episodes < max_episodes:
            cnt = cnt + 1
            actions, _ = model.predict(observation=ob, deterministic=True)
            ob, reward, done, info = zoo_env.step(actions)
            if done[0] or cnt == 1000:
                cnt = 0
                num_episodes += 1
                p.update(num_episodes)
                if 'winner' in info[0]:
                    win_zoo += 1
                # elif not ('loser' in info[0]):
                #     total_tie += 1
                ob = zoo_env.reset()
        p.finish()
        # test against trigger
        print("Test against trigger")
        ob = trigger_env.reset()
        num_episodes = 0
        win_trigger = 0
        # total_tie = 0
        cnt = 0
        p.start(max_episodes)
        while num_episodes < max_episodes:
            cnt = cnt + 1
            actions, _ = model.predict(observation=ob, deterministic=True)
            ob, reward, done, info = trigger_env.step(actions)
            if done[0] or cnt == 1000:
                cnt = 0
                num_episodes += 1
                p.update(num_episodes)
                if 'winner' in info[0]:
                    win_trigger += 1
                # elif not ('loser' in info[0]):
                #     total_tie += 1
                ob = trigger_env.reset()
        p.finish()

        print("\niter: %d, zoo: %d, trigger: %d"%(idx, win_zoo, win_trigger))
        if win_zoo >= zoo_threshold:
            dataset_idx = 1
            fix_first_layer = True
            learning_rate = max(learning_rate / 2, 1e-6)
        else:
            dataset_idx = 0
        if dataset_idx == 1 and win_trigger < trigger_threshold:
            print('optimization succeeds: zoo %d, trigger: %d'%(win_zoo, win_trigger))
            break

    model.save(os.path.join(model_dir, env_name.split('/')[1]+'-fine-tuned'+'-{:.0e}'.format(args.lr)))
