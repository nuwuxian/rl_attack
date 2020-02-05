import argparse
import gym
import gym_compete
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from agent import make_zoo_agent, make_victim_agent, make_trigger_agent, make_random_agent
from environment import make_multi2single_env, make_trigger_multi2single_env, make_zoo_multi2single_env, make_backdoor_env
from common import env_list

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

n_env = 1
max_episodes = 100
pretrain_template = "../agent-zoo/%s-pretrained-expert-1000-3000-1e-04.pkl"
adv_template = "../agent-zoo/%s-adv-%d.pkl"
trigger_length = 1
victim_length = 20

# check victim & stable_baselines
# choose between render & print only
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # env serial number
    # 0: RunToGoalAnts-v0
    # 1: RunToGoalHumans-v0
    # 2: YouShallNotPassHumans-v0
    # 3: KickAndDefend-v0
    # 4: SumoAnts-v0
    # 5: SumoHumans-v0
    parser.add_argument('env', type=int)
    # 'zoo', 'victim', 'stable_baseline'
    parser.add_argument('agent_type')
    parser.add_argument('--iter', type=int, default=1000000)
    parser.add_argument('--render', type=int, default=0)
    # tag, version for zoo
    # file name for stable baseline
    args = parser.parse_args()

    env_name = env_list[args.env]
    print("test %s with %s agent"%(env_name, args.agent_type))

    # zoo: agent loaded from multiagent-zoo
    # trigger: agent used to trigger backdoor (hardcoded)
    # victim: agent encoding the victim behavior (hardcoded)
    # adv: adversarially trained agent (nn)
    # pretrain: pretrained agent (nn)
    if args.agent_type == 'zoo-zoo':
        env = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(1)])
        agent = make_zoo_agent(env_name, env.observation_space, env.action_space, tag=1, scope='zoo')
    elif args.agent_type == 'victim-trigger':
        env = SubprocVecEnv([lambda: make_backdoor_env(env_name, end=trigger_length) for i in range(1)])
        agent = make_victim_agent(env_name, env.observation_space, env.action_space, end=victim_length)
    elif args.agent_type == 'pretrain-zoo':
        # env = make_zoo_multi2single_env(env_name)
        env = DummyVecEnv([lambda: make_zoo_multi2single_env(env_name)])
        # env = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name) for i in range(1)]) 
        # agent = PPO2.load(pretrain_template%(env_name.split('/')[1]))
        with tf.device('/GPU:0'):
            agent = PPO2.load("../agent-zoo/RunToGoalAnts-v0-pretrained-expert-1000-3000-1e-04.pkl")
    elif args.agent_type == 'pretrain-trigger':
        env = SubprocVecEnv([lambda: make_trigger_multi2single_env(env_name) for i in range(1)])
        # agent = PPO2.load(pretrain_template%(env_name.split('/')[1]))
        agent = PPO2.load("../agent-zoo/RunToGoalAnts-v0-pretrained-dummy-1000-100-1e-04.pkl")

    print("Finish loading, start test")
    ob = env.reset()
    num_episodes = 0
    total_score = 0
    total_reward = 0
    total_step = 0
    total_tie = 0
    cnt = 0
    while num_episodes < max_episodes:
        print(num_episodes)
        cnt = cnt + 1
        if args.render == 1:
            env.render()
        action = None
        if args.agent_type[:8] == 'pretrain' or args.agent_type[:3] == 'adv':
            actions, _ = agent.predict(observation=ob, deterministic=True)
            # print(actions)
            action = actions[0]
        else:
            action = agent.act(observation=ob[0])
        # print(action.shape)
        ob, reward, done, info = env.step([action])
        total_step += 1
        total_reward += reward
        if done[0]: # or cnt == 1000:
            cnt = 0
            num_episodes += 1
            if 'winner' in info[0]:
                total_score += 1
            elif not ('loser' in info[0]):
                total_tie += 1
            ob = env.reset()
        if args.render == 1:
            input("press any key")

    print(total_step/max_episodes)
    print("Winning Rate: %f, Tie Rate: %f"%(total_score/float(max_episodes), total_tie/float(max_episodes)))
