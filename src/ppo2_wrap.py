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

from zoo_utils import load_from_file, setFromFlat
from policy import mlp_policy, modeling_state
from stable_baselines.a2c.utils import total_episode_reward_logger
from game_utils import infer_next_ph
from explain_gradient import GradientExp
from pretrain_model import RL_func, MimicModel

import pdb


class MyPPO2(ActorCriticRLModel):
    # Hyper_setting
    # 3: Black-box or not
    # 4: Explanation or not
    # 5: Finetune or not
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, hyper_settings=[0, -0.06, 0, 1, 0, 1, True, True, False],
                 model_saved_loc=None, env_name=None, env_path=None):

        super(MyPPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                      _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

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
        self.env_path = env_path

        self._train_mimic = None
        self.model_saved_loc = model_saved_loc

        self.hyper_weights = hyper_settings[:6]
        self.black_box_att = hyper_settings[6]
        self.use_explanation = hyper_settings[7]
        self.masking_attention = hyper_settings[8]

        if self.black_box_att:
            self.pretrained_mimic = True
        else:
            self.pretrained_mimic = False

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
                if issubclass(self.policy, RecurrentActorCriticPolicy):
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
                if self.black_box_att:
                    with tf.variable_scope("mimic_model", reuse=False):
                         self.mimic_model = MimicModel(input_shape=self.observation_space.shape, \
                                                  action_shape=self.action_space.shape)
                         self.mimic_model.load('../agent-zoo/agent/mimic_model.h5')

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    # Xian added
                    self.action_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=self.action_ph.shape,
                                                             name="action_opp_next_ph")
                    self.obs_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=train_model.obs_ph.shape,
                                                          name="obs_opp_next_ph")
                    self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

                    action_ph_noise = train_model.deterministic_action

                    with tf.variable_scope("statem", reuse=True):
                        obs_oppo_predict, obs_oppo_noise_predict = modeling_state(self.action_ph, action_ph_noise,
                                                                                   train_model.obs_ph)
                    if not self.masking_attention:
                        self.attention = tf.placeholder(dtype=tf.float32, shape=[None], name="attention_ph")
                    else:
                        self.attention = tf.placeholder(dtype=tf.float32, shape=[None, train_model.obs_ph.shape[1]],
                                                        name="attention_ph")
                        obs_oppo_noise_predict = tf.multiply(obs_oppo_noise_predict, self.attention)

                    if not self.black_box_att:
                        with tf.variable_scope("victim_param", reuse=tf.AUTO_REUSE):
                            action_opp_mal_noise, _ = mlp_policy(obs_oppo_noise_predict, self.stochastic_ph, self.env.observation_space, \
                                                    self.env.action_space, [64, 64], True)
                    else:
                        # load the pretrained victim model
                        with tf.variable_scope("victim_param", reuse=tf.AUTO_REUSE):
                            victim_model = RL_func(self.observation_space.shape[0], self.action_space.shape[0])
                            action_opp_mal_noise = victim_model(obs_oppo_noise_predict)
                    if not self.masking_attention:
                        # update 2019/07/19, if not making, attention is only weighting loss
                        # on action along with time
                        # oppo's action change
                        # change into L infinity norm
                        self.change_opp_action_mse = tf.reduce_mean(
                            tf.abs(tf.multiply(action_opp_mal_noise - self.action_opp_next_ph,
                                                  tf.expand_dims(self.attention, axis=-1))
                                      )
                        )
                    else:
                        self.change_opp_action_mse = tf.reduce_mean(
                            tf.abs(action_opp_mal_noise - self.action_opp_next_ph)
                        )

                    # Prediction error on oppo's next observation
                    # change into the L infinity norm
                    # L(infinity) = max(0, ||l1 -l2|| - c)^2
                    self.state_modeling_mse = tf.reduce_mean(
                        tf.square(tf.math.maximum(tf.abs(obs_oppo_predict - self.obs_opp_next_ph) - 1, 0)))

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model.value_flat - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef + \
                           self.hyper_weights[1] * self.change_opp_action_mse

                    if self.black_box_att:
                        if not self.pretrained_mimic:
                            loss_mimic = self.hyper_weights[3] * self.state_modeling_mse
                        else:
                            loss_mimic = self.hyper_weights[3] * self.state_modeling_mse
                    else:  # if its' white box attack, then do not model the action output
                        loss_mimic = self.hyper_weights[3] * self.state_modeling_mse

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)
                    tf.summary.scalar('loss_mimic', loss_mimic)
                    tf.summary.scalar('_change oppo action mse', self.hyper_weights[1] * self.change_opp_action_mse)
                    tf.summary.scalar('_predict state mse', self.state_modeling_mse)

                    # add ppo loss
                    tf.summary.scalar('_PPO loss', loss - self.hyper_weights[1] * self.change_opp_action_mse)
                    params = tf_util.get_trainable_vars("model")
                    if self.full_tensorboard_log:
                        for var in params:
                            tf.summary.histogram(var.name, var)

                    self.params = [params, tf_util.get_trainable_vars("loss/statem")]

                    grads = tf.gradients(loss, self.params[0])
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params[0]))

                    grads_mimic = tf.gradients(loss_mimic, self.params[1])
                    if self.max_grad_norm is not None:
                        grads_mimic, _grad_norm_mimic = tf.clip_by_global_norm(grads_mimic, self.max_grad_norm)
                    grads_mimic = list(zip(grads_mimic, self.params[1]))

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                trainer_mimic = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train_mimic = trainer_mimic.apply_gradients(grads_mimic)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac',
                                   '_change_opp_a_loss', '_s_modeling_loss', '_a_modeling_loss']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))
                    # add attention onto the final results
                    tf.summary.scalar('att_hyp', tf.reduce_mean(self.attention))

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
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                # load the pretrained_value
                if not self.black_box_att:
                    victim_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "loss/victim_param")
                    param = load_from_file(param_pkl_path=self.env_path)
                    setFromFlat(victim_variable, param, sess=self.sess)
                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs,
                    a_opp_next, o_opp_next, attention, is_stochastic,
                    update, writer, states=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
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

        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,

                  self.action_opp_next_ph: a_opp_next, self.obs_opp_next_ph: o_opp_next,
                  self.stochastic_ph: is_stochastic, self.attention: attention
                  }
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
                _, change_opp_action_mse, state_modeling_mse = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
                     self._train_mimic,
                     self.change_opp_action_mse, self.state_modeling_mse],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
                _, change_opp_action_mse, state_modeling_mse = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
                     self._train_mimic,
                     self.change_opp_action_mse, self.state_modeling_mse],
                    td_map)

            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, \
            _, change_opp_action_mse, state_modeling_mse = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train,
                 self._train_mimic,
                 self.change_opp_action_mse, self.state_modeling_mse], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac, \
               change_opp_action_mse, state_modeling_mse

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)
            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()
            # define the victim_model
            if self.use_explanation:
                if self.pretrained_mimic:
                    exp_test = GradientExp(self.mimic_model)
                else:
                    exp_test = None
            else:
                exp_test = None
            nupdates = total_timesteps // self.n_batch

            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, \
                obs_oppo, actions_oppo, o_next, o_opp_next, a_opp_next = runner.run()

                obs_opp_ph = obs_oppo
                action_oppo_ph = actions_oppo
                # todo calculate the attention paid on opponent
                attention = self.calculate_attention(obs_opp_ph, exp_test)
                is_stochastic = False

                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                if states is None:  # nonrecurrent version
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
                            slices_hua = (arr[mbinds] for arr in (a_opp_next, o_opp_next, attention))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, *slices_hua,
                                                                 is_stochastic=is_stochastic, writer=writer, update=timestep))
                    self.num_timesteps += (self.n_batch * self.noptepochs) // batch_size * update_fac
                else:  # recurrent version
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
                            slices_hua = (arr[mb_flat_inds] for arr in (a_opp_next, o_opp_next, is_stochastic, attention))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, *slices_hua,
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

                model_file_name = "{0}agent_{1}.pkl".format(self.model_saved_loc, update * self.n_batch)
                if self.black_box_att:
                    if update % 100 == 0:
                        print("Model saved at: {}".format(model_file_name))
                        self.save(model_file_name)
                else:
                    if update % 1000 == 0:
                        print("Model saved at: {}".format(model_file_name))
                        self.save(model_file_name)

            return self

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
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
            "hyper_weights": self.hyper_weights,
            "black_box_att": self.black_box_att,
            "use_explanation": self.use_explanation,
            "masking_attention": self.masking_attention,
            "tensorboard_log": self.tensorboard_log,
            # "env": self.env,
            "env_name": self.env_name,
            "model_saved_loc": self.model_saved_loc,
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
    # Adding attention
    def calculate_attention(self, obs_oppo, exp_test=None):
        if self.use_explanation:
            assert exp_test != None
            grads = exp_test.grad(obs_oppo)
            oppo_action = exp_test.output(obs_oppo)
            if not self.masking_attention:
                if "YouShallNotPassHuman" in self.env_name or \
                        "SumoHumans" in self.env_name or \
                        "KickAndDefend" in self.env_name:
                    grads[:, 0:-24] = 0
                else:
                    grads[:, 0:-15] = 0
                new_obs_oppo = grads * obs_oppo
                new_oppo_action = exp_test.output(new_obs_oppo)
                relative_norm = np.max(abs(new_oppo_action - oppo_action), axis=1)
                # relative_action_norm = np.linalg.norm(new_oppo_action - oppo_action, axis=1)
                new_scalar = 1.0 / (1 + relative_norm)
                return new_scalar * self.hyper_weights[5]

            else:  # todo now the grads are used to combine weight the observations
                return grads * self.hyper_weights[5]
        else:
            return np.ones(obs_oppo.shape[0])




class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
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
        mb_obs_oppo, mb_actions_oppo = [], []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            # this should be placed after env,step
            mb_obs_oppo.append(self.env.get_attr('oppo_ob').copy())
            oppo_action = np.asarray(self.env.get_attr('action'))
            mb_actions_oppo.append(oppo_action)  # todo [0] only deals with single env

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs_oppo = np.asarray(mb_obs_oppo, dtype=self.obs.dtype)
        mb_actions_oppo = np.asarray(mb_actions_oppo)

        mb_o_next = infer_next_ph(mb_obs)
        # todo modify obs_ph to obs_opp_ph
        mb_o_opp_next = infer_next_ph(mb_obs_oppo)

        mb_a_opp_next = infer_next_ph(mb_actions_oppo)

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, \
        mb_obs_oppo, mb_actions_oppo, mb_o_next, mb_o_opp_next, mb_a_opp_next = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward,
                                   mb_obs_oppo, mb_actions_oppo, mb_o_next, mb_o_opp_next, mb_a_opp_next))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, \
               mb_obs_oppo, mb_actions_oppo, mb_o_next, mb_o_opp_next, mb_a_opp_next

def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, float):
        value_schedule = constfn(value_schedule)
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
