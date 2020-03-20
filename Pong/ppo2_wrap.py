import time
import sys, os
import multiprocessing
from collections import deque

from shutil import copyfile, rmtree
from os import listdir
from os.path import isfile, join

import gym
import numpy as np
import tensorflow as tf
from stable_baselines import logger

from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner

from stable_baselines.a2c.utils import total_episode_reward_logger
from value import MlpLstmValue, MlpValue
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
import pdb
from pong_utils import infer_obs_opp_ph



class MyPPO2(ActorCriticRLModel):
    """ Learn policy with PPO """
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 coef_opp_init=1, coef_opp_schedule='const', coef_adv_init=1, coef_adv_schedule='const',
                 coef_abs_init=1, coef_abs_schedule='const', max_grad_norm=0.5, lam=0.95, nminibatches=4,
                 noptepochs=4, cliprange=0.2, verbose=0,  lr_schedule='const', tensorboard_log=None, _init_setup_model=True,
                 policy_kwargs=None, is_mlp=True, full_tensorboard_log=False, model_saved_loc=None, env_name=None, opp_value=None):

        """
        :param: policy: policy network and value function.
        :param: env: environment.
        :param: gamma: discount factor.
        :param: n_step: number of steps of the vectorized environment per update (i.e. batch size is n_step * nenv where
                        nenv is number of environment copies simulated in parallel).
        :param: ent_coef: policy entropy coefficient in the optimization objective.
        :param: learning_rate: learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                               training and 0 is the end of the training.
        :param: vf_coef: value function loss coefficient in the optimization objective.
        :param: max_grad_norm: gradient norm clipping coefficient.
        :param: lam: advantage estimation discounting factor.
        :param: nminibatches: number of training minibatches per update. For recurrent policies,
                              should be smaller or equal than number of environments run in parallel.
        :param: noptepochs: number of training epochs per update.
        :param: cliprange: clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                           and 0 is the end of the training
        :param: verbose: ????.
        :param: tensorboard_log: ????
        :param: _init_setup_model: flag of whether defining the ppo model.
        :param: policy_kwargs: ???
        :param: full_tensorboard_log: ???
        :param: model_saved_loc: model save location.
        :param: env_name: name of the environment.
        :param: env_path: path of the environment.???
        """

        super(MyPPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                      _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)
        self.coef_opp_init = coef_opp_init
        self.coef_opp_schedule = coef_opp_schedule
        self.coef_adv_init = coef_adv_init
        self.coef_adv_schedule = coef_adv_schedule

        self.coef_abs_init = coef_abs_init
        self.coef_abs_schedule = coef_abs_schedule

        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        self.env = env
        self.env_name = env_name

        self._train_mimic = None
        self.model_saved_loc = model_saved_loc

        self.is_mlp = is_mlp
        self.opp_value = opp_value


        # self.hyper_weights = hyper_settings[:6]
        # self.black_box_att = hyper_settings[6]
        # self.use_explanation = hyper_settings[7]
        # self.masking_attention = hyper_settings[8]

        # if self.black_box_att:
        #     self.pretrained_mimic = True
        # else:
        #     self.pretrained_mimic = False

        if self.tensorboard_log is not None:
            self.zoo_model_meta_file = self.tensorboard_log + '/oppopi'
        else:
            import warnings
            warnings.warn("self.tensorboard_log path not specified! Thus zoo_model_meta_file path is temporarily None!")
            self.zoo_model_meta_file = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."
            self.n_batch = self.n_envs * self.n_steps
            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2
            self.graph = tf.Graph()

            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy) or issubclass(self.opp_value, MlpLstmValue):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, " \
                                                                 "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches


                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                # Define victim value function model
                with tf.variable_scope("value_model", reuse=tf.AUTO_REUSE):
                    vact_model = self.opp_value(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                                  n_batch_step)
                    vtrain_model = self.opp_value(self.sess, self.observation_space, self.action_space,
                                                    self.n_envs // self.nminibatches, self.n_steps,
                                                    n_batch_train)

                with tf.variable_scope("value1_model", reuse=tf.AUTO_REUSE):
                    vact1_model = self.opp_value(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                                 n_batch_step)
                    vtrain1_model = self.opp_value(self.sess, self.observation_space, self.action_space,
                                                    self.n_envs // self.nminibatches, self.n_steps,
                                                    n_batch_train)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")

                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")

                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    self.coef_opp_ph = tf.placeholder(tf.float32, [], name="coef_opp_ph")
                    self.coef_adv_ph = tf.placeholder(tf.float32, [], name="coef_adv_ph")
                    self.coef_abs_ph = tf.placeholder(tf.float32, [], name="coef_abs_ph")

                    self.opp_advs_ph = tf.placeholder(tf.float32, [None], name="opp_advs_ph")
                    self.opp_rewards_ph = tf.placeholder(tf.float32, [None], name="opp_rewards_ph")
                    self.old_opp_vpred_ph = tf.placeholder(tf.float32, [None], name='old_opp_vpred_ph')

                    self.abs_advs_ph = tf.placeholder(tf.float32, [None], name="abs_advs_ph")
                    self.abs_rewards_ph = tf.placeholder(tf.float32, [None], name="abs_rewards_ph")
                    self.old_abs_vpred_ph = tf.placeholder(tf.float32, [None], name='old_abs_vpred_ph')

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    # adversarial agent value function loss
                    vpred = train_model.value_flat
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model.value_flat - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    # victim agent value function loss
                    opp_vpred = vtrain_model.value_flat
                    opp_vpredclipped = self.old_opp_vpred_ph + tf.clip_by_value(
                        vtrain_model.value_flat - self.old_opp_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    opp_vf_losses1 = tf.square(opp_vpred - self.opp_rewards_ph)
                    opp_vf_losses2 = tf.square(opp_vpredclipped - self.opp_rewards_ph)
                    self.opp_vf_loss = .5 * tf.reduce_mean(tf.maximum(opp_vf_losses1, opp_vf_losses2))

                    # diff value function loss
                    abs_vpred = vtrain1_model.value_flat
                    abs_vpredclipped = self.old_abs_vpred_ph + tf.clip_by_value(
                        vtrain1_model.value_flat - self.old_abs_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    abs_vf_losses1 = tf.square(abs_vpred - self.abs_rewards_ph)
                    abs_vf_losses2 = tf.square(abs_vpredclipped - self.abs_rewards_ph)
                    self.abs_vf_loss = .5 * tf.reduce_mean(tf.maximum(abs_vf_losses1, abs_vf_losses2))

                    # ratio
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)

                    # ppo training loss
                    pg_losses = (self.coef_abs_ph*self.abs_advs_ph + self.coef_opp_ph*self.opp_advs_ph
                                 + self.coef_adv_ph*self.advs_ph) * ratio
                    pg_losses2 = (self.coef_abs_ph*self.abs_advs_ph + self.coef_opp_ph*self.opp_advs_ph
                                  + self.coef_adv_ph*self.advs_ph) * \
                                 tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)

                    # seperate the loss function
                    opp_pg_losses = self.coef_opp_ph * self.opp_advs_ph * ratio
                    opp_pg_losses2 = self.coef_opp_ph * self.opp_advs_ph * \
                                 tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)

                    abs_pg_losses = self.coef_abs_ph * self.abs_advs_ph * ratio
                    abs_pg_losses2 = self.coef_abs_ph * self.abs_advs_ph * \
                                 tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)

                    adv_pg_losses = pg_losses - opp_pg_losses - abs_pg_losses
                    adv_pg_losess2 = pg_losses2 - opp_pg_losses2 - abs_pg_losses2

                    self.opp_pg_loss = tf.reduce_mean(tf.maximum(opp_pg_losses, opp_pg_losses2))
                    self.abs_pg_loss = tf.reduce_mean(tf.maximum(abs_pg_losses, abs_pg_losses2))
                    self.adv_pg_loss = tf.reduce_mean(tf.maximum(adv_pg_losses, adv_pg_losess2))

                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    # final ppo loss
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    params = tf_util.get_trainable_vars("model")
                    if self.full_tensorboard_log:
                        for var in params:
                            tf.summary.histogram(var.name, var)

                    self.params = [params, tf_util.get_trainable_vars("value_model"), tf_util.get_trainable_vars("value1_model")]

                    grads = tf.gradients(loss, self.params[0])
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params[0]))

                    grads_value = tf.gradients(self.opp_vf_loss, self.params[1])
                    if self.max_grad_norm is not None:
                        grads_value, _grad_norm_value = tf.clip_by_global_norm(grads_value, self.max_grad_norm)
                    grads_value = list(zip(grads_value, self.params[1]))

                    grads_value1 = tf.gradients(self.abs_vf_loss, self.params[2])
                    if self.max_grad_norm is not None:
                        grads_value1, _grad_norm_value = tf.clip_by_global_norm(grads_value1, self.max_grad_norm)
                    grads_value1 = list(zip(grads_value1, self.params[2]))

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                trainer_value = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train_value = trainer_value.apply_gradients(grads_value)

                trainer_value1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train_value1 = trainer_value1.apply_gradients(grads_value1)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac',
                                   '_opp_value_loss', 'opp_policy_loss', 'adv_policy_loss', 'abs_policy_loss']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))
                    # add attention onto the final results
                    # tf.summary.scalar('att_hyp', tf.reduce_mean(self.attention))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)
                self.train_model = train_model
                self.act_model = act_model

                self.vtrain_model = vtrain_model
                self.vact_model = vact_model

                self.vtrain1_model = vtrain1_model
                self.vact1_model = vact1_model

                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, coef_opp, coef_adv, coef_abs, obs, returns, masks, actions, values, neglogpacs,
                    opp_obs, opp_returns, opp_values, abs_returns, abs_values, update, writer,
                    states=None, opp_states=None, abs_states=None):
        """
        Training of PPO2 Algorithm
        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param coef_opp: (float) opponent loss coefficient.
        :param coef_adv: (float) adversarial loss coefficient.
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        opp_advs = opp_returns - opp_values
        opp_advs = (opp_advs - opp_advs.mean()) / (opp_advs.std() + 1e-8)

        abs_advs = abs_returns - abs_values
        abs_advs = (abs_advs - abs_advs.mean()) / (abs_advs.std() + 1e-8)

        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
                  self.opp_advs_ph: opp_advs,
                  self.opp_rewards_ph: opp_returns, self.old_opp_vpred_ph: opp_values,
                  self.abs_advs_ph: abs_advs,
                  self.abs_rewards_ph: abs_returns, self.old_abs_vpred_ph: abs_values,
                  self.coef_opp_ph: coef_opp, self.coef_adv_ph: coef_adv, self.coef_abs_ph: coef_abs,
                  self.vtrain1_model.obs_ph: opp_obs, self.vtrain_model.obs_ph: opp_obs
                  }

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if opp_states is not None:
            td_map[self.vtrain_model.states_ph] = opp_states
            td_map[self.vtrain_model.dones_ph] = masks

        if abs_states is not None:
            td_map[self.vtrain1_model.states_ph] = abs_states
            td_map[self.vtrain1_model.dones_ph] = masks

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        assert writer == None
        # if writer is not None:
        #     # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
        #     if self.full_tensorboard_log and (1 + update) % 10 == 0:
        #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #         run_metadata = tf.RunMetadata()
        #         summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
        #         _, opp_vf_loss = self.sess.run(
        #             [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
        #              self._train_value,
        #              self.opp_vf_loss],
        #             td_map, options=run_options, run_metadata=run_metadata)
        #         writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
        #     else:
        #         summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
        #         _, opp_vf_loss = self.sess.run(
        #             [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
        #              self._train_value,
        #              self.opp_vf_loss],
        #             td_map)
        #
        #     writer.add_summary(summary, (update * update_fac))
        # else:
        policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
        _, opp_vf_loss, opp_pg_loss, adv_pg_loss, abs_pg_loss = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
             self._train_value, self.opp_vf_loss, self.opp_pg_loss, self.adv_pg_loss, self.abs_pg_loss], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac, \
               opp_vf_loss, opp_pg_loss, adv_pg_loss, abs_pg_loss

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, use_victim_ob=False):

        # Transform to callable if needed
        print('*********************************************')
        print('learning rate schedule: %s' % self.lr_schedule)
        print('oppo coefficient schedule: %s' % self.coef_opp_schedule)
        print('adv coefficient schedule: %s' % self.coef_adv_schedule)
        print('*********************************************')

        self.learning_rate = get_schedule_fn(self.learning_rate, schedule=self.lr_schedule)

        self.coef_opp = get_schedule_fn(self.coef_opp_init, schedule=self.coef_opp_schedule)
        self.coef_adv = get_schedule_fn(self.coef_adv_init, schedule=self.coef_adv_schedule)
        self.coef_abs = get_schedule_fn(self.coef_abs_init, schedule=self.coef_abs_schedule)

        self.cliprange = get_schedule_fn(self.cliprange, schedule='const')

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)
            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma,
                            lam=self.lam, v_model=self.vact_model, v1_model=self.vact1_model,
                            is_mlp=self.is_mlp, use_victim_ob=use_victim_ob)

            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()
            nupdates = total_timesteps // self.n_batch

            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                if self.lr_schedule == 'const':
                    lr_now = self.learning_rate(0)
                elif self.lr_schedule == 'linear':
                    lr_now = self.learning_rate(update, nupdates)
                elif self.lr_schedule == 'step':
                    lr_now = self.learning_rate(update)

                if self.coef_opp_schedule == 'const':
                    coef_opp_now = self.coef_opp(0)
                elif self.coef_opp_schedule == 'linear':
                    coef_opp_now = self.coef_opp(update, nupdates)
                elif self.coef_opp_schedule == 'step':
                    coef_opp_now = self.coef_opp(update)

                if self.coef_adv_schedule == 'const':
                    coef_adv_now = self.coef_adv(0)
                elif self.coef_adv_schedule == 'linear':
                    coef_adv_now = self.coef_adv(update, nupdates)
                elif self.coef_adv_schedule == 'step':
                    coef_adv_now = self.coef_adv(update)

                if self.coef_abs_schedule == 'const':
                    coef_abs_now = self.coef_abs(0)
                elif self.coef_abs_schedule == 'linear':
                    coef_abs_now = self.coef_abs(update, nupdates)
                elif self.coef_adv_schedule == 'step':
                    coef_abs_now = self.coef_abs(update)

                cliprangenow = self.cliprange(0)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, opp_true_reward, \
                abs_true_reward, opp_obs, opp_states, opp_returns, opp_values, abs_states, abs_returns, abs_values \
                    = runner.run()

                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                if states is None:  # nonrecurrent version
                    if self.is_mlp:
                        update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                        inds = np.arange(self.n_batch)
                        for epoch_num in range(self.noptepochs):
                            np.random.shuffle(inds)
                            for start in range(0, self.n_batch, batch_size):
                                timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                                self.n_batch + start) // batch_size)
                                end = start + batch_size
                                mbinds = inds[start:end]
                                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                                slices_victim = (arr[mbinds] for arr in (opp_obs, opp_returns, opp_values, abs_returns, abs_values))

                                mb_loss_vals.append(self._train_step(lr_now, cliprangenow, coef_opp_now, coef_adv_now, coef_abs_now,
                                                                     *slices, *slices_victim,
                                                                     writer=writer, update=timestep))

                        self.num_timesteps += (self.n_batch * self.noptepochs) // batch_size * update_fac
                    else:
                        # ifnot mlp
                        update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                        assert self.n_envs % self.nminibatches == 0
                        env_indices = np.arange(self.n_envs)
                        flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                        envs_per_batch = batch_size // self.n_steps
                        for epoch_num in range(self.noptepochs):
                            np.random.shuffle(env_indices)
                            for start in range(0, self.n_envs, envs_per_batch):
                                timestep = self.num_timesteps // update_fac + (
                                            (self.noptepochs * self.n_envs + epoch_num *
                                             self.n_envs + start) // envs_per_batch)
                                end = start + envs_per_batch
                                mb_env_inds = env_indices[start:end]
                                mb_flat_inds = flat_indices[mb_env_inds].ravel()
                                slices = (arr[mb_flat_inds] for arr in
                                          (obs, returns, masks, actions, values, neglogpacs))
                                slices_victim = (arr[mb_flat_inds] for arr in (opp_obs, opp_returns, opp_values, abs_returns, abs_values))
                                opp_mb_states = opp_states[mb_env_inds]
                                abs_mb_states = abs_states[mb_env_inds]

                                mb_loss_vals.append(self._train_step(lr_now, cliprangenow, coef_opp_now, coef_adv_now, coef_abs_now,
                                                                     *slices, *slices_victim,
                                                                     update=timestep, writer=writer,
                                                                     opp_states=opp_mb_states, abs_states=abs_mb_states))

                        self.num_timesteps += (self.n_envs * self.noptepochs) // envs_per_batch * update_fac

                else:  # recurrent version
                    # TODO pass away

                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            slices_victim = (arr[mb_flat_inds] for arr in (opp_returns, opp_values))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, *slices_victim,
                                                                 update=timestep, writer=writer, states=mb_states))
                    self.num_timesteps += (self.n_envs * self.noptepochs) // envs_per_batch * update_fac

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)

                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)

                    # logout the returns and values
                    logger.logkv("adv_reward", np.mean(true_reward))
                    logger.logkv("opp_reward", np.mean(opp_true_reward))
                    logger.logkv("abs_reward", np.mean(abs_true_reward))
                    logger.logkv("adv_returns", np.mean(returns))
                    logger.logkv("opp_returns", np.mean(opp_returns))
                    logger.logkv("abs_returns", np.mean(abs_returns))

                    logger.logkv("learning_rate", np.mean(lr_now))
                    logger.logkv("victim_loss_weight", np.mean(coef_opp_now))
                    logger.logkv("adv_loss_weight", np.mean(coef_adv_now))
                    logger.logkv("diff_loss_weight", np.mean(coef_abs_now))

                    logger.logkv("explained_variance", float(explained_var))

                    # print the attention weights
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                '''
                model_file_name = "{0}agent_{1}.pkl".format(self.model_saved_loc, update * self.n_batch)
                if self.black_box_att:
                    if update % 100 == 0:
                        print("Model saved at: {}".format(model_file_name))
                        self.save(model_file_name)
                else:
                    if update % 1000 == 0:
                        print("Model saved at: {}".format(model_file_name))
                        self.save(model_file_name)
                '''
            return self

    def save(self, save_path):
        data = {
            "lr_schedule":self.lr_schedule,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate_init": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "tensorboard_log": self.tensorboard_log,
            "env_name": self.env_name,
            "model_saved_loc": self.model_saved_loc,
            "vic_weight_init": self.coef_opp_init,
            "adv_weight_init": self.coef_adv_init,
            "diff_weight_init": self.coef_abs_init
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)
        print("######------", load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        print(data)
        if "tensorboard_log" not in data.keys():
            if "final_model/model.pkl" in load_path:
                data["tensorboard_log"] = load_path[:load_path.find("/final_model/model.pkl")]
            else:
                print("The load path does not contain  /final model/model.pkl! ")
                raise NotImplementedError
            print("tensorboard_log path not in saved model, setting to {}".format(data["tensorboard_log"]))

        data["zoo_model_meta_file"] = data["tensorboard_log"] + "/oppopi"

        # todo delete this
        data["env_name"] = "multicomp/YouShallNotPassHumans-v0"

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params[0], params[0]):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        restores = []
        for param, loaded_p in zip(model.params[1], params[1]):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, v_model, v1_model, is_mlp, use_victim_ob):
        """
        A runner to learn the policy of an environment for a model
        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.v_model = v_model
        self.v1_model = v1_model
        self.is_mlp = is_mlp

        self.opp_states = v_model.initial_state
        self.abs_states = v1_model.initial_state
        self.use_victim_ob = use_victim_ob

    def run(self):
        """
        Run a learning step of the model
        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []

        mb_states = self.states
        ep_infos = []
        mb_opp_obs = []

        mb_opp_states = self.opp_states
        mb_abs_states = self.abs_states

        mb_opp_rewards = []
        mb_opp_values = []

        mb_abs_rewards = []
        mb_abs_values = []

        # mb_obs_oppo, mb_actions_oppo = [], []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            dones = self.dones.copy()
            obs_adv = np.copy(self.obs)

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            # get the opponent rewards and values
            oppo_reward = np.ones(1)
            oppo_reward[0] = infos[0]['r1']
            abs_reward = np.ones(1)
            abs_reward[0] = infos[0]['r0'] - infos[0]['r1']

            mb_opp_rewards.append(oppo_reward)
            mb_abs_rewards.append(abs_reward)
            # use victim agent observation
            if self.use_victim_ob:
                obs_oppo = infer_obs_opp_ph(obs_adv)
            else:
                obs_oppo = obs_adv

            mb_opp_obs.append(obs_oppo)
            values_oppo, self.opp_states = self.v_model.value(obs_oppo, self.opp_states, dones)
            values_abs, self.abs_states = self.v1_model.value(obs_oppo, self.abs_states, dones)

            mb_opp_values.append(values_oppo)
            mb_abs_values.append(values_abs)

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        mb_opp_obs = np.asarray(mb_opp_obs, dtype=self.obs.dtype)
        if not self.is_mlp:
            mb_opp_states = np.asarray(mb_opp_states, dtype=np.float32)
            mb_abs_states = np.asarray(mb_abs_states, dtype=np.float32)

        mb_opp_values = np.asarray(mb_opp_values, dtype=np.float32)
        mb_opp_rewards = np.asarray(mb_opp_rewards, dtype=np.float32)

        mb_abs_values = np.asarray(mb_abs_values, dtype=np.float32)
        mb_abs_rewards = np.asarray(mb_abs_rewards, dtype=np.float32)

        last_values = self.model.value(self.obs, self.states, self.dones)
        if self.use_victim_ob:
            ob = infer_obs_opp_ph(self.obs)
        else:
            ob = self.obs
        opp_last_values, _ = self.v_model.value(np.asarray(ob), self.opp_states, self.dones)
        abs_last_values, _ = self.v1_model.value(np.asarray(ob), self.abs_states, self.dones)

        mb_advs = np.zeros_like(mb_rewards)
        mb_opp_advs = np.zeros_like(mb_opp_rewards)
        mb_abs_advs = np.zeros_like(mb_abs_rewards)

        true_reward = np.copy(mb_rewards)
        opp_true_reward = np.copy(mb_opp_rewards)
        abs_true_reward = np.copy(mb_abs_rewards)
        last_gae_lam = 0
        opp_last_gae_lam = 0
        abs_last_gae_lam = 0

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                opp_nextvalues = opp_last_values
                abs_nextvalues = abs_last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
                opp_nextvalues = mb_opp_values[step + 1]
                abs_nextvalues = mb_abs_values[step + 1]

            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam

            opp_delta = mb_opp_rewards[step] + self.gamma * opp_nextvalues * nextnonterminal - mb_opp_values[step]
            mb_opp_advs[step] = opp_last_gae_lam = opp_delta + self.gamma * self.lam * nextnonterminal * opp_last_gae_lam

            abs_delta = mb_abs_rewards[step] + self.gamma * abs_nextvalues * nextnonterminal - mb_abs_values[step]
            mb_abs_advs[step] = abs_last_gae_lam = abs_delta + self.gamma * self.lam * nextnonterminal * abs_last_gae_lam

        mb_returns = mb_advs + mb_values
        mb_opp_returns = mb_opp_advs + mb_opp_values
        mb_abs_returns = mb_abs_advs + mb_abs_values


        # attack the rewards and opp_returns
        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, opp_true_reward, abs_true_reward, \
        mb_opp_obs, mb_opp_returns, mb_opp_values, mb_abs_returns, mb_abs_values = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, opp_true_reward, \
                                   abs_true_reward, mb_opp_obs, mb_opp_returns, mb_opp_values, mb_abs_returns, mb_abs_values))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, opp_true_reward, \
               abs_true_reward, mb_opp_obs, mb_opp_states, mb_opp_returns, mb_opp_values, mb_abs_states, mb_abs_returns, mb_abs_values


def get_schedule_fn(value_schedule, schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.
    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if schedule == 'const':
        value_schedule = constfn(value_schedule)
    elif schedule == 'linear':
        value_schedule = linearfn(value_schedule)
    elif schedule == 'step':
        value_schedule = stepfn(value_schedule)
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1
    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def linearfn(val):
    """
    :param val: (float)
    :return: (function)
    """

    def func(epoch, total_epoch):
        frac = 1.0 - (epoch - 1.0) / total_epoch
        return val*frac

    return func


def stepfn(val):
    """
    :param val: (float)
    :return: (function)
    """

    def func(epoch, drop=0.8, epoch_drop=2e2):
        ratio = drop**((epoch+1) // epoch_drop)
        return val*ratio

    return func


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)
    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.
    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
