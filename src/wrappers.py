from collections import defaultdict
import itertools
import os
from os import path as osp
import warnings

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np


from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from zoo_utils import LSTMPolicy, MlpPolicyValue


class VideoWrapper(Wrapper):
    """Creates videos from wrapped environment by called render after each timestep."""
    def __init__(self, env, directory, single_video=True):
        """

        :param env: (gym.Env) the wrapped environment.
        :param directory: the output directory.
        :param single_video: (bool) if True, generates a single video file, with episodes
                             concatenated. If False, a new video file is created for each episode.
                             Usually a single video file is what is desired. However, if one is
                             searching for an interesting episode (perhaps by looking at the
                             metadata), saving to different files can be useful.
        """
        super(VideoWrapper, self).__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video

        self.directory = osp.abspath(directory)

        # Make sure to not put multiple different runs in the same directory,
        # if the directory already exists
        error_msg = "You're trying to use the same directory twice, " \
                    "this would result in files being overwritten"
        assert not os.path.exists(self.directory), error_msg
        os.makedirs(self.directory, exist_ok=True)

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        # change info
        if done:
            winners = [i for i, d in info.items() if 'winner' in d]
            metadata = {'winners': winners}
            self.video_recorder.metadata.update(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def _reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self):
        """Called at the start of each episode (by _reset). Always creates a video recorder
           if one does not already exist. When a video recorder is already present, it will only
           create a new one if `self.single_video == False`."""
        if self.video_recorder is not None:
            # Video recorder already started.
            if not self.single_video:
                # We want a new video for each episode, so destroy current recorder.
                self.video_recorder.close()
                self.video_recorder = None

        if self.video_recorder is None:
            # No video recorder -- start a new one.
            self.video_recorder = VideoRecorder(
                env=self.env,
                base_path=osp.join(self.directory, 'video.{:06}'.format(self.episode_id)),
                metadata={'episode_id': self.episode_id},
            )

    def _close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super(VideoWrapper, self)._close()