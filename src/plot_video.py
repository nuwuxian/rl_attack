"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""

import collections
import functools
import glob
import logging
import os
import os.path as osp
import re
import tempfile
import warnings
import tensorflow as tf
import argparse
import gym
from wrappers import VideoWrapper
from annotated_gym_compete import AnnotatedGymCompete
from environment import make_zoo_multi2single_env
from compete import GymCompeteToOurs, game_outcome

from common import env_list

# zoo policy and stable-baseline policy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from zoo_utils import LSTMPolicy, MlpPolicyValue
from video_utils import simulate, load_policy


parser = argparse.ArgumentParser()
# game env
parser.add_argument("--env", type=int, default=5)
# number of game environment. should be divisible by NBATCHES if using a LSTM policy
parser.add_argument("--n_games", type=int, default=1) # N_GAME = 8

parser.add_argument("--agent_a_type", type=str, default="zoo")
parser.add_argument("--agent_a_path", type=str, default="/home/xkw5132/Music/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl")
parser.add_argument("--agent_b_type", type=str, default="zoo")
parser.add_argument("--agent_b_path", type=str, default="/home/xkw5132/Music/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl")
parser.add_argument("--norm_path", type=str, default=None)

parser.add_argument("--render", type=bool, default=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--timesteps", type=int, default=None)

parser.add_argument("--video", type=bool, default=True)
parser.add_argument("--save_dir", type=str, default="/home/xkw5132/Desktop/video")


args = parser.parse_args()

# env name
env_name = env_list[args.env]

agent_a_path = args.agent_a_path
agent_b_path = args.agent_b_path
agent_a_type = args.agent_a_type
agent_b_type = args.agent_b_type
norm_path = args.norm_path

episodes = args.episodes
timesteps = args.timesteps

num_env = args.n_games

video = args.video
render = args.render
video_params = None

if video:
   video_params = {
        'save_dir': args.save_dir,                 # directory to store videos in.
        'single_file': True,              # if False, stores one file per episode
        'annotated': True,                # for gym_compete, color-codes the agents and adds scores
        'annotation_params': {
            'camera_config': 'default',
            'short_labels': False,
            'resolution': (640, 480),
            'font': 'times',
            'font_size': 24,
        },
   }

def get_empirical_score(venv, agents, episodes, timesteps, render, norm_path):
    """Computes number of wins for each agent and ties. At least one of `episodes`
       and `timesteps` must be specified.

    :param venv: (VecEnv) vector environment
    :param agents: (list<BaseModel>) agents/policies to execute.
    :param episodes: (int or None) maximum number of episodes.
    :param timesteps (int or None) maximum number of timesteps.
    :param render: (bool) whether to render to screen during simulation.
    :return a dictionary mapping from 'winN' to wins for each agent N, and 'ties' for ties."""
    if episodes is None and timesteps is None:
        raise ValueError("At least one of 'max_episodes' and 'max_timesteps' must be non-None.")

    result = {f'win{i}': 0 for i in range(len(agents))}
    result['ties'] = 0

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    sim_stream = simulate(venv, agents, render=render, record=False, norm_path=norm_path)

    completed_episodes = 0
    for _, _, done, info in sim_stream:

        if done:
            completed_episodes += 1
            winner = game_outcome(info)
            if winner is None:
               result['ties'] += 1
            else:
               result[f'win{winner}'] += 1

        if episodes is not None and completed_episodes >= episodes:
            break

    return result

def _save_video_or_metadata(env_dir, saved_video_path):
    """
    A helper method to pull the logic for pattern matching certain kinds of video and metadata
    files and storing them as sacred artifacts with clearer names

    :param env_dir: The path to a per-environment folder where videos are stored
    :param saved_video_path: The video file to be reformatted and saved as a sacred artifact
    :return: None
    """
    env_number = env_dir.split("/")[-1]
    video_ptn = re.compile(r'video.(\d*).mp4')
    metadata_ptn = re.compile(r'video.(\d*).meta.json')

def score_agent(env_name, agent_a_path, agent_b_path, agent_a_type, 
                agent_b_type,  num_env, videos, video_params):

    # create dir for save video 
    save_dir = video_params['save_dir']
    if videos:
        if save_dir is None:
            tmp_dir = tempfile.TemporaryDirectory()
            save_dir = tmp_dir.name
        else:
            tmp_dir = None
        video_dirs = [osp.join(save_dir, str(i)) for i in range(num_env)]

    def env_fn(i):

        env = gym.make(env_name)
        env = GymCompeteToOurs(env)
        if videos:
            if video_params['annotated']:
                if 'multicomp' in env_name:
                    assert num_env == 1, "pretty videos requires num_env=1"
                    env = AnnotatedGymCompete(env, env_name, agent_a_type, agent_a_path,
                                              agent_b_type, agent_b_path, None,
                                              **video_params['annotation_params'])
                else:
                    warnings.warn(f"Annotated videos not supported for environment '{env_name}'")
            env = VideoWrapper(env, video_dirs[i], video_params['single_file'])
        return env

    env = env_fn(0)

    agent_paths = [agent_a_path, agent_b_path]
    agent_types = [agent_a_type, agent_b_type]

    # load agents
    agents = load_policy(agent_types, agent_paths, env, env_name)
    score = get_empirical_score(env, agents, episodes, timesteps, render, norm_path)

    env.close()

    if videos:
        for env_video_dir in video_dirs:
            try:
                for file_path in os.listdir(env_video_dir):
                    _save_video_or_metadata(env_video_dir, file_path)

            except FileNotFoundError:
                warnings.warn("Can't find path {}; no videos from that path added as artifacts"
                              .format(env_video_dir))

        if tmp_dir is not None:
            tmp_dir.cleanup()
    return score

def main():
    score_agent(env_name, agent_a_path, agent_b_path, agent_a_type,
                agent_b_type, num_env, video, video_params)

if __name__ == '__main__':
    main()
