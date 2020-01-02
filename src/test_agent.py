import argparse
import gym
import gym_compete
import tensorflow as tf
from stable_baselines import PPO2
from agent import make_zoo_agent, make_victim_agent, make_trigger_agent, make_random_agent
from environment import make_multi2single_env, make_zoo_multi2single_env, make_trojan_env
from common import env_list

max_episodes = 100
pretrain_template = "../agent-zoo/%s-pretrained-expert-1000-1000-1e-03.pkl"
adv_template = "../agent-zoo/%s-adv-%d.pkl" 

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

    env = None
    agent = None
    if args.agent_type == 'zoo-zoo':
        env = make_zoo_multi2single_env(env_name)
        agent = make_zoo_agent(env_name, env.observation_space, env.action_space, tag=1, scope='zoo')
    elif args.agent_type == 'victim-trigger':
        env = make_trojan_env(env_name)
        agent = make_victim_agent(env_name, env.observation_space, env.action_space)
    elif args.agent_type == 'pretrain-zoo':
        env = make_zoo_multi2single_env(env_name)
        agent = PPO2.load(pretrain_template%(env_name.split('/')[1]))
    elif args.agent_type == 'pretrain-trigger':
        inner_env = gym.make(env_name)
        inner_agent = make_trigger_agent(env_name)
        env = make_multi2single_env(inner_env, inner_agent)
        agent = PPO2.load(pretrain_template%(env_name.split('/')[1]))
    elif args.agent_type == 'adv-zoo':
        env = make_zoo_multi2single_env(env_name)
        agent = PPO2.load(adv_template%(env_name.split('/')[1], args.iter))
    elif args.agent_type == 'adv-trigger':
        inner_env = gym.make(env_name)
        inner_agent = make_trigger_agent(env_name)
        env = make_multi2single_env(inner_env, inner_agent)
        agent = PPO2.load(adv_template%(env_name.split('/')[1], args.iter))
    elif args.agent_type == 'rand-trigger':
        inner_env = gym.make(env_name)
        inner_agent = make_trigger_agent(env_name, inner_env.action_space.spaces[1])
        env = make_multi2single_env(inner_env, inner_agent)
        agent = make_random_agent(inner_env.action_space.spaces[0])

    print("Finish loading, start test")
    ob = env.reset()
    num_episodes = 0
    total_score = 0
    total_reward = 0
    total_step = 0
    total_tie = 0
    while num_episodes < max_episodes:
        if args.render == 1:
            env.render()
        action = None
        if args.agent_type[:8] == 'pretrain' or args.agent_type[:3] == 'adv':
            action, _ = agent.predict(observation=ob, deterministic=True)
        else:
            action = agent.act(observation=ob)
        ob, reward, done, info = env.step(action)
        total_step += 1
        total_reward += reward
        if done:
            num_episodes += 1
            if 'winner' in info:
                total_score += 1
            elif not ('loser' in info):
                total_tie += 1
            ob = env.reset()
        if args.render == 1:
            input("press any key")

    print(total_step/max_episodes)
    print("Winning Rate: %f, Tie Rate: %f"%(total_score/float(max_episodes), total_tie/float(max_episodes)))
