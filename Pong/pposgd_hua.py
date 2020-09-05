from collections import deque
import time

import gym
import tensorflow as tf
import numpy as np
from mpi4py import MPI

from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, ActorCriticRLModel, SetVerbosity, \
    TensorboardWriter
from stable_baselines import logger,PPO1
import stable_baselines.common.tf_util as tf_util
# from stable_baselines.common.policies import ActorCriticPolicy
from policies import ActorCriticPolicy
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.trpo_mpi.utils import traj_segment_generator, add_vtarg_and_adv, flatten_lists
from stable_baselines.a2c.utils import total_episode_reward_logger
from tensorflow import keras

from explain_master_gradient import MasterModel
from explain_hua_gradient import GradientExp
import pickle as pkl
import cloudpickle
import os
from mimic_action import MimicModel
import pdb



class PPO1_hua_model_value(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (MPI version).
    Paper: https://arxiv.org/abs/1707.06347

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                 optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                 schedule='linear', verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, hyper_weights=[0.,0.,0.,0.,0.,0.],
                 benigned_model_file=None, black_box_att=False, attention_weights=False, model_saved_loc=None,
                 clipped_attention=False):

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        # Not needed to debug
        self.debug = False

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.summary = None
        self.episode_reward = None
        self.hyper_weights = hyper_weights

        self.dt = 0.0165 # this is from gym_pong.py
        # Scene.__init__(self, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        self.model_saved_loc = model_saved_loc

        self.black_box_att = black_box_att
        if self.black_box_att:
            self.pretrained_mimic = True
        else:
            self.pretrained_mimic = False

        self.benigned_model_file = benigned_model_file

        if self.benigned_model_file is not None: # attacking an self_trained model
            self.benigned_model_file_meta_file = self.tensorboard_log+'oppopi'
            #self.oppo_model = PPO1.load(benigned_model_file) # use this for prediction
            self.use_explaination = attention_weights
            self.masking_attention = False
        else: # attacking master model
            self.oppo_model = MasterModel(input_shape=(13,), action_shape=(2,))
            self.use_explaination = attention_weights
            if self.use_explaination:
                self.masking_attention = clipped_attention
            else:
                self.masking_attention = False

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.graph = tf.Graph()

            #self.benigned_policy_load_and_build(self.benigned_model_file)

            tf.reset_default_graph()

            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=10, graph=self.graph)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         None, reuse=False, **self.policy_kwargs)

                # # todo Network for benigned policy
                # with tf.variable_scope("oppopi", reuse=False):
                #     oppo_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                #                           None, reuse=False, **self.policy_kwargs
                if self.pretrained_mimic:
                    with tf.variable_scope("pretrained_mimic", reuse=False):
                        self.mimic_model = MimicModel(input_shape=(13,), action_shape=(2,))
                        self.mimic_model.load("./pretrain/saved/mimic_model.h5")


                with tf.variable_scope("loss", reuse=False):

                    # Target advantage function (if applicable)
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None], name="atarg_ph")

                    # Empirical return
                    ret = tf.placeholder(dtype=tf.float32, shape=[None], name="empirical_ret_ph")

                    # learning rate multiplier, updated with schedule
                    lrmult = tf.placeholder(name='lrmult_ph', dtype=tf.float32, shape=[])

                    # Annealed cliping parameter epislon
                    clip_param = self.clip_param * lrmult

                    obs_ph = self.policy_pi.obs_ph
                    action_ph = self.policy_pi.pdtype.sample_placeholder([None], name="action_ph")
                    self.action_ph = action_ph
                    action_ph_noise = self.policy_pi.deterministic_action

                    #todo make sure the dimension
                    action_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=action_ph.shape, name="action_opp_next_ph")
                    obs_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=obs_ph.shape, name="obs_opp_next_ph")
                    action_mask_ph = tf.placeholder(dtype=tf.float32, shape=action_ph.shape, name="action_mask_ph")
                    state_value_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="state_value_opp_next_ph")

                    #todo: manipulate part of the opp obs by the action of current action_ph * 0.1 (60 frames per second)
                    # delta_obs_change = tf.concat([tf.zeros([tf.shape(action_ph)[0], 4], dtype=tf.float32),
                    #                                 action_ph[:,0:1]*self.dt,
                    #                                 -obs_opp_ph[:,5:6]+action_ph[:, 0:1],
                    #                                 action_ph[:,1:]*self.dt,
                    #                                 -obs_opp_ph[:,7:8]+action_ph[:, 1:],
                    #                                 tf.zeros([tf.shape(action_ph)[0], 5], dtype=tf.float32)], -1)
                    # obs_opp_mal = tf.add(delta_obs_change,obs_opp_ph)

                    # Todo network for modeling oppo current observation, given my current observation and my previous action
                    obs_next_ph = tf.placeholder(dtype=tf.float32, shape=obs_ph.shape, name="obs_next_ph")
                    self.obs_next_ph = obs_next_ph


                    with tf.variable_scope("statem", reuse=True):
                        obs_oppo_predict, obs_oppo_noise_predict = self.modeling_state(action_ph, action_ph_noise, obs_ph)
                    '''
                    obs_opp_mal = tf.concat([obs_oppo_predict,
                                             obs_next_ph[:, 0:4],
                                             obs_next_ph[:, 8:]],
                                            -1, name="obs_oppo_predict")
                    '''
                    obs_opp_mal = tf.concat([obs_oppo_predict,
                                             tf.multiply(obs_next_ph[:,0:4],tf.constant([-1.0, -1.0, 1.0, 1.0])),
                                             tf.multiply(obs_next_ph[:,8:],tf.constant([-1.0, -1.0, 1.0, 1.0, 1.0]))],
                                            -1, name="obs_oppo_predict")
                    # We keep self's observation since we want the agent to take action
                    # that change the oppo's action most between similar positions
                    '''
                    obs_opp_mal_noise = tf.concat([obs_oppo_noise_predict,
                                                   obs_next_ph[:, 0:4],
                                                   obs_next_ph[:, 8:]],
                                            -1, name="obs_oppo_predict_noise")
                    '''
                    obs_opp_mal_noise = tf.concat([obs_oppo_noise_predict,
                                             tf.multiply(obs_next_ph[:, 0:4], tf.constant([-1.0, -1.0, 1.0, 1.0])),
                                             tf.multiply(obs_next_ph[:, 8:], tf.constant([-1.0, -1.0, 1.0, 1.0, 1.0]))],
                                            -1, name="obs_oppo_predict_noise")
                    
                    # Todo Gradient attention on observation
                    if not self.masking_attention:
                        attention = tf.placeholder(dtype=tf.float32, shape=[None], name="attention_ph")
                    else:
                        attention = tf.placeholder(dtype=tf.float32, shape=[None, 13], name="attention_ph")
                        with tf.variable_scope("opp_mask", reuse=True):
                            obs_opp_mal_noise_mask = tf.multiply(obs_opp_mal_noise, attention)
                            if False:
                               pass
                            else:
                                obs_opp_mal_noise = obs_opp_mal_noise_mask

                    if not self.black_box_att:
                        action_opp_mal, state_value_opp_mal = self.begined_policy_tf(obs_opp_mal,
                                                                                     self.benigned_model_file)
                        action_opp_mal_noise, state_value_opp_mal_noise = self.begined_policy_tf(obs_opp_mal_noise,
                                                                                     self.benigned_model_file)
                    else:
                        if not self.pretrained_mimic:
                            # Todo network for infering oppo's action & value directly without a known model
                            action_opp_mal, state_value_opp_mal, \
                            action_opp_mal_noise, state_value_opp_mal_noise = self.modeling_action_and_value(obs_opp_mal,
                                                                                                         obs_opp_mal_noise)
                        else:
                            action_opp_mal = self.mimic_model.model(obs_opp_mal)
                            action_opp_mal_noise = self.mimic_model.model(obs_opp_mal_noise)
                            state_value_opp_mal = None
                            state_value_opp_mal_noise = None

                    self.action_opp_mal = action_opp_mal
                    self.obs_opp_mal = obs_oppo_predict

                    # todo kl by making oppo's action
                    klmask = tf.reduce_mean(tf.square(action_ph - action_mask_ph))

                    if not self.masking_attention:
                        # update 2019/07/19, if not making, attention is only weighting loss
                        # on action along with time
                        # oppo's action change
                        change_opp_action_mse = tf.reduce_mean(tf.abs(
                            tf.multiply(action_opp_mal_noise - action_opp_next_ph, tf.expand_dims(attention, axis=-1))
                        ))
                        change_opp_state_mse = tf.reduce_mean(tf.abs(obs_opp_mal_noise - obs_opp_next_ph))

                        # oppo's state value change
                        if state_value_opp_mal is not None:  # victim policy do not have state value
                            opp_state_value_mean = tf.reduce_mean(
                                tf.multiply(state_value_opp_mal_noise, tf.expand_dims(attention, axis=-1))
                            )
                        else:
                            opp_state_value_mean = tf.reduce_mean(tf.square(
                                tf.multiply((state_value_opp_next_ph - state_value_opp_next_ph),
                                            tf.expand_dims(attention, axis=-1))
                            ))
                    else:
                        change_opp_action_mse = tf.reduce_mean(tf.abs(action_opp_mal_noise - action_opp_next_ph))


                        # modified may 15
                        # change_opp_action_mse = tf.reduce_mean(tf.square(action_opp_mal_noise - action_opp_next_ph))

                        # oppo's state value change
                        if state_value_opp_mal is not None:  # victim policy do not have state value
                            opp_state_value_mean = tf.reduce_mean(state_value_opp_mal_noise)
                        else:
                            opp_state_value_mean = tf.reduce_mean(tf.square(
                                state_value_opp_next_ph - state_value_opp_next_ph
                            ))

                    # prediction error on oppo's next observation
                    # change to state modeling function
                    # change to infinity-norm
                    '''
                    state_modeling_mse = tf.reduce_mean(
                        tf.square(tf.math.maximum(tf.abs(tf.multiply(obs_oppo_predict, tf.constant([-1.0, -1.0, 1.0, 1.0]))
                                  - obs_next_ph[:, 4:8]) - 1e-2, 0)))
                    '''
                    
                    state_modeling_mse = tf.reduce_mean(
                        tf.square(tf.math.maximum(tf.abs(obs_oppo_predict - obs_next_ph[:, 4:8] * tf.constant([-1.0, -1.0, 1.0, 1.0]))
                                                  - 1e-2, 0))
                    )
                    # modified may 15
                    # state_modeling_mse = tf.reduce_mean(tf.square(obs_oppo_predict - obs_next_ph[:, 4:8] * tf.constant([-1.0, -1.0, 1.0, 1.0])))
                    # Prediction error on oppo's next action, if not black-box attack, this should be very close to zero
                    action_modeling_mse = tf.reduce_mean(tf.square(tf.multiply(action_opp_mal, action_opp_next_ph)))


                    kloldnew = old_pi.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    pol_entpen = (-self.entcoeff) * meanent

                    # pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action_ph) -
                                   old_pi.proba_distribution.logp(action_ph))

                    # surrogate from conservative policy iteration
                    surr1 = ratio * atarg
                    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                    # PPO's pessimistic surrogate (L^CLIP)
                    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    vf_loss = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))

                    # # todo add loss of the oppo network output (action) and loss of self mask
                    # mask_loss = tf.reduce_mean(self.policy_pi.proba_distribution.kl(self.policy_pi.proba_distribution))

                    total_loss = pol_surr + pol_entpen + vf_loss + \
                                 self.hyper_weights[0] * klmask + \
                                 self.hyper_weights[1] * (change_opp_action_mse - 2 * change_opp_state_mse) + \
                                 self.hyper_weights[2] * opp_state_value_mean

                    if self.black_box_att:
                        if not self.pretrained_mimic:
                            mimic_loss = self.hyper_weights[3] * state_modeling_mse + \
                                         self.hyper_weights[4] * action_modeling_mse
                        else:
                            mimic_loss = self.hyper_weights[3] * state_modeling_mse
                    else: # if its' white box attack, then do not model the action output
                        mimic_loss = self.hyper_weights[3] * state_modeling_mse

                    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent,
                              klmask, change_opp_action_mse, opp_state_value_mean, state_modeling_mse, action_modeling_mse]
                    self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent",
                                       "maskkl", "opp_act_chg", "opp_svalue",
                                       "s_modeling", "a_modeling"]

                    tf.summary.scalar('entropy_loss', pol_entpen)
                    tf.summary.scalar('policy_gradient_loss', pol_surr)
                    tf.summary.scalar('value_function_loss', vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', meankl)
                    tf.summary.scalar('clip_factor', clip_param)
                    tf.summary.scalar('loss', total_loss)
                    tf.summary.scalar('ppo_loss', total_loss - self.hyper_weights[1] * change_opp_action_mse)
                    tf.summary.scalar('loss_mimic', mimic_loss)
                    #tf.summary.scalar('mask oppo action KL', klmask)
                    tf.summary.scalar('change oppo action mse', change_opp_action_mse)
                    #tf.summary.scalar('oppo state value mean', opp_state_value_mean)
                    tf.summary.scalar('predict state mse', state_modeling_mse)
                    #tf.summary.scalar('predict action mse', action_modeling_mse)

                    if not self.black_box_att:
                        self.params = [tf_util.get_trainable_vars("model"), tf_util.get_trainable_vars("loss/statem")]
                    else:
                        if not self.pretrained_mimic:
                            self.params = [tf_util.get_trainable_vars("model"), tf_util.get_trainable_vars("loss/statem")
                                           + tf_util.get_trainable_vars("loss/mimic")]
                        else:
                            self.params = [tf_util.get_trainable_vars("model"), tf_util.get_trainable_vars("loss/statem")]
                    if self.masking_attention:
                        self.params = [self.params[0] + tf_util.get_trainable_vars("loss/opp_mask"), self.params[1]]

                    if self.debug:
                        self.debug_params = tf_util.get_trainable_vars("loss/statem")

                    self.assign_old_eq_new = tf_util.function(
                        [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                         zipsame(tf_util.get_globals_vars("oldpi"), tf_util.get_globals_vars("model"))])

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.adam = MpiAdam(self.params[0], epsilon=self.adam_epsilon, sess=self.sess)

                with tf.variable_scope("Adam_mpi_mimic", reuse=False):
                    self.adam_mimic = MpiAdam(self.params[1], epsilon=self.adam_epsilon, sess=self.sess)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('att_hype', tf.reduce_mean(attention))
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.optim_stepsize))
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_param))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', ret)
                        tf.summary.histogram('learning_rate', self.optim_stepsize)
                        tf.summary.histogram('advantage', atarg)
                        tf.summary.histogram('clip_range', self.clip_param)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', obs_ph)
                        else:
                            tf.summary.histogram('observation', obs_ph)

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                tf_util.initialize(sess=self.sess)
                if self.benigned_model_file is not None and not self.black_box_att: # using an benigned model
                    self.saver.restore(self.sess, self.benigned_model_file_meta_file)

                self.summary = tf.summary.merge_all()

                self.lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult,
                                                     action_opp_next_ph, obs_opp_next_ph, action_mask_ph,
                                                     state_value_opp_next_ph,
                                                     obs_next_ph, attention],
                                                    [self.summary, tf_util.flatgrad(total_loss, self.params[0])] + losses)
                self.compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult,
                                                        action_opp_next_ph, obs_opp_next_ph, action_mask_ph,
                                                        state_value_opp_next_ph,
                                                        obs_next_ph, attention],
                                                       losses)
                self.lossandgrad_mimic = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult,
                                                           action_opp_next_ph, obs_opp_next_ph, action_mask_ph,
                                                           state_value_opp_next_ph,
                                                           obs_next_ph, attention],
                                                    [self.summary, tf_util.flatgrad(mimic_loss, self.params[1])] + losses)

                if self.debug:
                    self.compute_oppo_mal = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult,
                                                     action_opp_next_ph, obs_opp_next_ph, action_mask_ph, state_value_opp_next_ph,
                                                     obs_next_ph, attention],
                                                       [obs_opp_mal, action_opp_mal])

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="PPO1_hua",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        if self.debug:
            with open("{0}/test-debug-black-box.data".format(self.model_saved_loc), "wb") as f:
                f.close()

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            with self.sess.as_default():
                self.adam.sync()
                self.adam_mimic.sync()

                # Prepare for rollouts
                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()

                # rolling buffer for episode lengths
                lenbuffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                rewbuffer = deque(maxlen=100)

                self.episode_reward = np.zeros((self.n_envs,))

                obs_list = []
                act_list = []

                if self.use_explaination:
                    if self.pretrained_mimic:
                        exp_test = GradientExp(self.mimic_model)
                    else:
                        exp_test = GradientExp(self.oppo_model)
                else:
                    exp_test = None

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError

                    # Add the newly constant function
                    # cur_lambda = min(1.0, 0.2 * max(0.1, int(iters_so_far / 1000)))
                    if iters_so_far >= 0 and iters_so_far < 2000:
                        cur_lambda = 1.0
                    elif iters_so_far >= 2000 and iters_so_far < 4000:
                        cur_lambda = 1.0
                    elif iters_so_far >= 4000 and iters_so_far < 6000:
                        cur_lambda = 1.0
                    else:
                        cur_lambda = 1.0

                    logger.log("********** Iteration %i ************" % iters_so_far)
                    seg = seg_gen.__next__()

                    add_vtarg_and_adv(seg, self.gamma, self.lam)

                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    obs_ph, action_ph, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

                    # true_rew is the reward without discount
                    if writer is not None:
                        self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                          seg["true_rew"].reshape((self.n_envs, -1)),
                                                                          seg["dones"].reshape((self.n_envs, -1)),
                                                                          writer, self.num_timesteps)

                    # predicted value function before udpate
                    vpredbefore = seg["vpred"]
                    ob_next_ph = self.infer_obs_next_ph(obs_ph)
                    # todo modify obs_ph to obs_opp_ph
                    obs_opp_next_ph = self.infer_obs_opp_ph(ob_next_ph)
                    # modified today
                    self.benigned_model_file = None
                    action_oppo_next_ph, state_value_opp_next_ph \
                        = self.output_act_statev_for_adv(obs_opp_next_ph, file_name=self.benigned_model_file)

                    obs_opp_ph = self.infer_obs_opp_ph(obs_ph)
                    action_oppo_ph, state_value_opp_ph \
                        = self.output_act_statev_for_adv(obs_opp_ph, file_name=self.benigned_model_file)

                    # todo modify obs_ph to obs_mask_ph
                    obs_mask_ph = self.infer_obs_mask_ph(obs_ph)
                    action_mask_ph, _, _, _ = self.policy_pi.step(obs_mask_ph)

                    # todo calculate the attention paid on opponent
                    # Add scalar into the model
                    attention_ph = self.calculate_new_attention(obs_opp_ph, cur_lambda, exp_test)

                    # standardized advantage function estimate
                    # obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult,
                    # action_opp_next_ph, obs_opp_next_ph, action_mask_ph, state_value_opp_next_ph,
                    # obs_next_ph, attention
                    atarg = (atarg - atarg.mean()) / atarg.std()
                    dataset = Dataset(dict(ob=obs_ph, ac=action_ph, atarg=atarg, vtarg=tdlamret,
                                           action_opp_next=action_oppo_next_ph, action_mask=action_mask_ph,
                                           opp_obs_next=obs_opp_next_ph, state_value_opp_next=state_value_opp_next_ph,
                                           ob_next=ob_next_ph, attention=attention_ph),
                                      shuffle=not self.policy.recurrent)
                    optim_batchsize = self.optim_batchsize or obs_ph.shape[0]

                    # set old parameter values to new parameter values
                    self.assign_old_eq_new(sess=self.sess)

                    # Here we do a bunch of optimization epochs over the data

                    # first we optimize over the mimic model
                    '''
                    # train mimic model
                    if iters_so_far % 10 == 0:
                        # get the observation and action
                        obs_list.append(obs_opp_ph)
                        act_list.append(action_oppo_ph)
                    '''
                    print("modeling state weight: ", self.sess.run(self.modeling_state_w)[0])
                    # print("modeling oppo weight: ", self.sess.run(self.modeling_oppo_w)[0])
                    # print("mimic model weights: ", self.mimic_model.model.layers[0].get_weights()[0])
                    logger.log("Optimizing mimic model (1/2)...")
                    logger.log(fmt_row(13, self.loss_names))

                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad_mimic(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult,
                                                                                 batch["action_opp_next"],
                                                                                 batch["opp_obs_next"],
                                                                                 batch["action_mask"],
                                                                                 batch['state_value_opp_next'],
                                                                                 batch['ob_next'],
                                                                                 batch['attention'],
                                                                                 sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad_mimic(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult,
                                                                                 batch["action_opp_next"],
                                                                                 batch["opp_obs_next"],
                                                                                 batch["action_mask"],
                                                                                 batch['state_value_opp_next'],
                                                                                 batch['ob_next'],
                                                                                 batch['attention'],
                                                                                 sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad_mimic(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"],
                                                                       batch["action_opp_next"],
                                                                       batch["opp_obs_next"],
                                                                       batch["action_mask"],
                                                                       batch['state_value_opp_next'],
                                                                       batch['ob_next'],
                                                                       batch['attention'],
                                                                       sess=self.sess)

                            self.adam_mimic.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)

                        logger.log(fmt_row(13, np.mean(losses, axis=0)))

                    # next we optimize the RL part
                    print("modeling state weight: ", self.sess.run(self.modeling_state_w)[0])
                    # print("modeling oppo weight: ", self.sess.run(self.modeling_oppo_w)[0])
                    # print("mimic model weights: ", self.mimic_model.model.layers[0].get_weights()[0])
                    logger.log("Optimizing RL model (2/2)...")
                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult,
                                                                                 batch["action_opp_next"],
                                                                                 batch["opp_obs_next"],
                                                                                 batch["action_mask"],
                                                                                 batch['state_value_opp_next'],
                                                                                 batch['ob_next'],
                                                                                 batch['attention'],
                                                                                 sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult,
                                                                                 batch["action_opp_next"],
                                                                                 batch["opp_obs_next"],
                                                                                 batch["action_mask"],
                                                                                 batch['state_value_opp_next'],
                                                                                 batch['ob_next'],
                                                                                 batch['attention'],
                                                                                 sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"],
                                                                       batch["action_opp_next"],
                                                                       batch["opp_obs_next"],
                                                                       batch["action_mask"],
                                                                       batch['state_value_opp_next'],
                                                                       batch['ob_next'],
                                                                       batch['attention'],
                                                                       sess=self.sess)

                            self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)

                        logger.log(fmt_row(13, np.mean(losses, axis=0)))

                    logger.log("Evaluating losses...")
                    losses = []
                    for batch in dataset.iterate_once(optim_batchsize):
                        newlosses = self.compute_losses(batch["ob"], batch["ob"], batch["ac"],
                                                        batch["atarg"], batch["vtarg"],cur_lrmult,
                                                        batch["action_opp_next"],
                                                        batch["opp_obs_next"],
                                                        batch["action_mask"],
                                                        batch['state_value_opp_next'],
                                                        batch['ob_next'],
                                                        batch['attention'],
                                                        sess=self.sess)
                        losses.append(newlosses)

                    if self.debug:
                        print("modeling state weight: ", self.sess.run(self.modeling_state_w)[0])
                        if self.black_box_att:
                            # print("mimic model weights: ", self.mimic_model.model.layers[0].get_weights()[0])
                            # print("modeling oppo weight: ", self.sess.run(self.modeling_oppo_w)[0])
                            if iters_so_far % 100 == 0:
                                debug_mal_opp_obs, debug_mal_opp_act \
                                    = self.compute_oppo_mal(obs_ph, obs_ph, action_ph, atarg, tdlamret, cur_lrmult,
                                                            action_oppo_next_ph, obs_opp_next_ph, action_mask_ph,
                                                            state_value_opp_next_ph, ob_next_ph, attention_ph,
                                                            sess=self.sess)
                                with open("{0}/test-debug-black-box.data".format(self.model_saved_loc), "ab+") as f:
                                    pkl.dump([debug_mal_opp_obs, debug_mal_opp_act,
                                              obs_opp_next_ph, action_oppo_next_ph], f, protocol=2)
                        else:
                            if iters_so_far % 1000 == 0:
                                debug_mal_opp_obs, debug_mal_opp_act \
                                    = self.compute_oppo_mal(obs_ph, obs_ph, action_ph, atarg, tdlamret, cur_lrmult,
                                                            action_oppo_next_ph, obs_opp_next_ph, action_mask_ph,
                                                            state_value_opp_next_ph, ob_next_ph, attention_ph,
                                                            sess=self.sess)
                                with open("{0}/test-debug-black-box.data".format(self.model_saved_loc), "ab+") as f:
                                    pkl.dump([debug_mal_opp_obs, debug_mal_opp_act,
                                              obs_opp_next_ph, action_oppo_next_ph], f, protocol=2)

                    mean_losses, _, _ = mpi_moments(losses, axis=0)
                    logger.log(fmt_row(13, mean_losses))
                    for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                        logger.record_tabular("loss_" + name, loss_val)
                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                    # local values
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])

                    # list of tuples
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    lenbuffer.extend(lens)
                    rewbuffer.extend(rews)
                    if len(lenbuffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
                        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1
                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dump_tabular()

                    if self.black_box_att:
                        if iters_so_far % 100 == 0:
                            model_file_name = "{0}agent_{1}.pkl".format(self.model_saved_loc, iters_so_far)
                            print("Model saved at: {}".format(model_file_name))
                            self.save(model_file_name)
                    else:
                        if iters_so_far%1000 == 0:
                            model_file_name = "{0}agent_{1}.pkl".format(self.model_saved_loc, iters_so_far)
                            print("Model saved at: {}".format(model_file_name))
                            self.save(model_file_name)
                '''
                obs_numpy = np.vstack(obs_list)
                act_numpy = np.vstack(act_list)
                with open('./saved/black_data.pkl', 'ab+') as f:
                    pkl.dump([obs_numpy, act_numpy], f, protocol=2)
                '''
        return self

    def predict_debug(self, obs, obs_prev, act_prev):
        obs = np.array(obs)
        vectorized_env = self._is_vectorized_observation(obs, self.observation_space)

        observations = obs.reshape((-1,) + self.observation_space.shape)

        observations_prev = obs_prev.reshape((-1,) + self.observation_space.shape)
        actions_prev = act_prev.reshape((-1,) + self.action_space.shape)


        _ , obs_opp, act_opp = self.sess.run(
            [self.policy_pi.deterministic_action, self.obs_opp_mal, self.action_opp_mal],
            {self.policy_pi.obs_ph: observations_prev, self.action_ph: actions_prev, self.obs_next_ph: observations})

        actions = self.sess.run([self.policy_pi.deterministic_action], {self.policy_pi.obs_ph: observations})

        clipped_actions = actions[0]
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

        return clipped_actions, obs_opp, act_opp

    def save(self, save_path): #todo add more data to save for loading
        data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.clip_param,
            "entcoeff": self.entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.optim_stepsize,
            "optim_batchsize": self.optim_batchsize,
            "lam": self.lam,
            "adam_epsilon": self.adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "debug": self.debug,
            "hyper_weights": self.hyper_weights
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        #model.load_parameters(params)
        # Loading parameters
        restores = []

        print(len(model.params))
        print('another line .....')
        print(len(params))
        for param, loaded_p in zip(model.params, params):
            restores.append(tf.assign(param, loaded_p))
        model.sess.run(restores)

        return model

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    def infer_obs_next_ph(self, obs_ph):
        '''
        This is self action at time t+1
        :param obs_ph:
        :return:
        '''
        obs_next_ph = np.zeros_like(obs_ph)
        obs_next_ph[:-1, :] = obs_ph[1:, :]
        return obs_next_ph

    def infer_obs_opp_ph(self, obs_ph):
        '''
        This is oppos observation at time t
        :param obs_ph:
        :return:
        '''
        abs_opp_ph = np.zeros_like(obs_ph)
        neg_sign = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1]
        abs_opp_ph[:, :4] = obs_ph[:, 4:8]
        abs_opp_ph[:, 4:8] = obs_ph[:, 0:4]
        abs_opp_ph[:, 8:] = obs_ph[:, 8:]
        return abs_opp_ph*neg_sign

    def infer_action_previous_ph(self,action_ph):
        '''
        This is self action at time t-1
        :param obs_ph:
        :return:
        '''
        data_length = action_ph.shape[0]
        action_prev_ph = np.zeros_like(action_ph)
        action_prev_ph[1:, :] = action_ph[0:(data_length-1), :]
        return action_prev_ph

    def infer_obs_mask_ph(self, obs_ph):
        abs_mask_ph = np.zeros_like(obs_ph)
        abs_mask_ph[:, :4] = obs_ph[:, :4]
        abs_mask_ph[:, 8:] = obs_ph[:, 8:]
        return abs_mask_ph

    def begined_policy_tf(self, x, benigned_model_file=None):
        if benigned_model_file is not None:
            saver = tf.train.import_meta_graph(self.benigned_model_file_meta_file+".meta",
                                               input_map={'oppopi/input/Ob:0': x}, import_scope="import")
            output_action = tf.get_default_graph().get_tensor_by_name("loss/import/oppopi/output/add:0")
            output_state_value = tf.get_default_graph().get_tensor_by_name("loss/import/oppopi/output/strided_slice_1:0")
            # inspection value: weight w of beneighed model
            # w = tf.get_default_graph().get_tensor_by_name('loss/import/oppopi/model/pi_fc0/w:0')
            # tf.train.Saver(var_list={'loss/import/oppopi/model/pi_fc0/w:0': w})
            self.saver = saver
            # Todo add prediction on output action
            return output_action, output_state_value
        else:
            from RoboschoolPong_v0_2017may2 import SmallReactivePolicy
            weights_dense1_w_const = tf.constant(SmallReactivePolicy.weights_dense1_w, dtype=tf.float32)
            weights_dense1_b_const = tf.constant(SmallReactivePolicy.weights_dense1_b, dtype=tf.float32)
            weights_dense2_w_const = tf.constant(SmallReactivePolicy.weights_dense2_w, dtype=tf.float32)
            weights_dense2_b_const = tf.constant(SmallReactivePolicy.weights_dense2_b, dtype=tf.float32)
            weights_final_w_const = tf.constant(SmallReactivePolicy.weights_final_w, dtype=tf.float32)
            weights_final_b_const = tf.constant(SmallReactivePolicy.weights_final_b, dtype=tf.float32)

            x = tf.nn.relu(tf.matmul(x, weights_dense1_w_const) + weights_dense1_b_const)
            x = tf.nn.relu(tf.matmul(x, weights_dense2_w_const) + weights_dense2_b_const)
            output_action = tf.matmul(x, weights_final_w_const) + weights_final_b_const

            return output_action, None

    def benigned_policy_load_and_build(self, file_name, **kwargs):
        if self.benigned_model_file is None:
            return
        # similar to load()
        data, params = PPO1._load_from_file(file_name)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = PPO1(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(self.env)
        # model.setup_model()

        # Construct network for new policy

        with self.graph.as_default():
            sess = tf_util.single_threaded_session(graph=self.graph)
            with tf.variable_scope("oppopi", reuse=False):
                model.policy_pi = model.policy(sess, self.observation_space, self.action_space, self.n_envs, 1,
                                     None, reuse=False, **self.policy_kwargs)
                tf_util.initialize(sess=sess)

            model.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='oppopi')

            g0 = tf.train.Saver()

            sess = tf_util.single_threaded_session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            restores = []
            for param, loaded_p in zip(model.params, params):
                restores.append(param.assign(loaded_p))
            sess.run(restores)

            g0.save(sess, self.benigned_model_file_meta_file)

    def output_act_statev_for_adv(self, obs_opp_ph, file_name=None, state=None, mask=None, deterministic=False):
        if file_name is not None:
            if state is None:
                state = self.oppo_model.initial_state
            if mask is None:
                mask = [False for _ in range(self.oppo_model.n_envs)]
            observation = np.array(obs_opp_ph)
            vectorized_env = self.oppo_model._is_vectorized_observation(observation, self.oppo_model.observation_space)

            observation = observation.reshape((-1,) + self.oppo_model.observation_space.shape)
            actions, vpred, states, _ = self.oppo_model.step(observation, state, mask, deterministic=deterministic)

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.oppo_model.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.oppo_model.action_space.low, self.oppo_model.action_space.high)

            if not vectorized_env:
                if state is not None:
                    raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
                clipped_actions = clipped_actions[0]
            return clipped_actions, vpred
        else:
            from RoboschoolPong_v0_2017may1 import SmallReactivePolicy
            return SmallReactivePolicy.output_act_for_adv(obs_opp_ph), np.zeros(shape=(obs_opp_ph.shape[0],))

    def modeling_state(self, action_ph, action_noise, obs_self):
        '''
        Use current observation and my current action to infer my oppo's next position
        :return:
        '''
        w1 = tf.Variable(tf.truncated_normal([13, 32]), name="cur_obs_embed_w1")
        b1 = tf.Variable(tf.truncated_normal([32]), name="cur_obs_embed_b1")
        obs_embed = tf.nn.relu(tf.add(tf.matmul(obs_self, w1), b1), name="cur_obs_embed")

        w2 = tf.Variable(tf.truncated_normal([2, 4]), name="act_embed_w1")
        b2 = tf.Variable(tf.truncated_normal([4]), name="act_embed_b1")

        act_embed = tf.nn.relu(tf.add(tf.matmul(action_ph, w2), b2), name="act_embed")
        act_embed_noise = tf.nn.relu(tf.add(tf.matmul(action_noise, w2), b2), name="act_noise_embed")

        obs_act_concat = tf.concat([obs_embed, act_embed], -1, name="obs_act_concat")
        obs_act_noise_concat = tf.concat([obs_embed, act_embed_noise], -1, name="obs_act_noise_concat")


        w3 = tf.Variable(tf.truncated_normal([36, 16]), name="obs_act_embed_w1")
        b3 = tf.Variable(tf.truncated_normal([16]), name="obs_act_embed_b1")

        obs_act_embed = tf.nn.relu(tf.add(tf.matmul(obs_act_concat, w3), b3), name="obs_act_embed")
        obs_act_noise_embed = tf.nn.relu(tf.add(tf.matmul(obs_act_noise_concat, w3), b3), name="obs_act_noise_embed")


        w4 = tf.Variable(tf.truncated_normal([16, 4]), name="obs_oppo_predict_w1")
        b4 = tf.Variable(tf.truncated_normal([4]), name="obs_oppo_predict_b1")

        obs_oppo_predict = tf.nn.tanh(tf.add(tf.matmul(obs_act_embed, w4), b4), name="obs_oppo_predict_part")
        obs_oppo_predict_noise = tf.nn.tanh(tf.add(tf.matmul(obs_act_noise_embed, w4), b4), name="obs_oppo_predict_noise_part")
        self.modeling_state_w = w4

        return obs_oppo_predict, obs_oppo_predict_noise

    def calculate_attention(self, obs_oppo, cur_lambda, exp_test=None):
        if self.use_explaination:
            assert exp_test != None
            grads = exp_test.grad(obs_oppo)
            if not self.masking_attention:
                return np.max(grads[:, 4:8], axis=1) * self.hyper_weights[5] * cur_lambda
            else: # todo now the grads are used to combine weight the observations
                return grads*self.hyper_weights[5] * cur_lambda
        else:
        #   return np.ones(obs_oppo.shape[0])
            return 49 * np.random.random_sample(obs_oppo.shape[0]) + 1
    # define the new attention
    def calculate_new_attention(self, obs_oppo, cur_lamda, exp_test=None):
        if self.use_explaination:
            assert exp_test != None
            grads = exp_test.integratedgrad(obs_oppo)

            oppo_action = exp_test.output(obs_oppo)
            if not self.masking_attention:
                grads[:, 0:4] = 0
                grads[:,8:] = 0
                new_obs_oppo = grads * obs_oppo
                new_oppo_action = exp_test.output(new_obs_oppo)
                relative_norm = np.max(abs(new_oppo_action - oppo_action), axis=1)
                # relative_action_norm = np.linalg.norm(new_oppo_action - oppo_action, axis=1)
                new_scalar = 1.0 / (1.0 + relative_norm)
                return  new_scalar * self.hyper_weights[5] * cur_lamda

            else:  # todo now the grads are used to combine weight the observations
                return grads * self.hyper_weights[5] * cur_lamda
        else:
            return np.ones(obs_oppo.shape[0])

