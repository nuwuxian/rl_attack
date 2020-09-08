import tensorflow as tf
import numpy as np
import gym
import logging
import copy
import pickle
import sys
import pdb
from abc import ABC, abstractmethod

from tensorflow.contrib import layers


def load_from_file(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = np.load(param_pkl_path)
    return params


def load_from_model(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
       policy_param = params[1][0]
       flat_param = []
       for param in policy_param:
           flat_param.append(param.reshape(-1))
       flat_param = np.concatenate(flat_param, axis=0)
    else:
        flat_param = np.load(param_pkl_path, allow_pickle=True)
        if len(flat_param)==3:
            flat_param_1 = []
            for i in flat_param[0]:
                    flat_param_1.append(i)
            flat_param = []
            for param in flat_param_1:
                flat_param.append(param.reshape(-1))
            flat_param = np.concatenate(flat_param, axis=0)
    return flat_param

# MLP

# <tf.Variable 'mlp_policy/retfilter/sum:0' shape=() dtype=float32_ref>
# <tf.Variable 'mlp_policy/retfilter/sumsq:0' shape=() dtype=float32_ref>
# <tf.Variable 'mlp_policy/retfilter/count:0' shape=() dtype=float32_ref>
# <tf.Variable 'mlp_policy/obsfilter/sum:0' shape=(380,) dtype=float32_ref>
# <tf.Variable 'mlp_policy/obsfilter/sumsq:0' shape=(380,) dtype=float32_ref>
# <tf.Variable 'mlp_policy/obsfilter/count:0' shape=() dtype=float32_ref>

# <tf.Variable 'victim_policy/vffc1/w:0' shape=(380, 64) dtype=float32_ref>
# <tf.Variable 'victim_policy/vffc1/b:0' shape=(64,) dtype=float32_ref>
# <tf.Variable 'victim_policy/vffc2/w:0' shape=(64, 64) dtype=float32_ref>
# <tf.Variable 'victim_policy/vffc2/b:0' shape=(64,) dtype=float32_ref>
# <tf.Variable 'victim_policy/vffinal/w:0' shape=(64, 1) dtype=float32_ref>
# <tf.Variable 'victim_policy/vffinal/b:0' shape=(1,) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfc1/w:0' shape=(380, 64) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfc1/b:0' shape=(64,) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfc2/w:0' shape=(64, 64) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfc2/b:0' shape=(64,) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfinal/w:0' shape=(64, 17) dtype=float32_ref>
# <tf.Variable 'victim_policy/polfinal/b:0' shape=(17,) dtype=float32_ref>
# <tf.Variable 'victim_policy/logstd:0' shape=(1, 17) dtype=float32_ref>

# LSTM
# <tf.Variable 'lstm_policy/retfilter/sum:0' shape=() dtype=float32_ref>
# <tf.Variable 'lstm_policy/retfilter/sumsq:0' shape=() dtype=float32_ref>
# <tf.Variable 'lstm_policy/retfilter/count:0' shape=() dtype=float32_ref>
# <tf.Variable 'lstm_policy/obsfilter/sum:0' shape=(137,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/obsfilter/sumsq:0' shape=(137,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/obsfilter/count:0' shape=() dtype=float32_ref>
# 137*2+4 = 278
# <tf.Variable 'lstm_policy/fully_connected/weights:0' shape=(137, 128) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected/biases:0' shape=(128,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/lstmv/basic_lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>
# <tf.Variable 'lstm_policy/lstmv/basic_lstm_cell/bias:0' shape=(512,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_1/weights:0' shape=(128, 1) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_1/biases:0' shape=(1,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_2/weights:0' shape=(137, 128) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_2/biases:0' shape=(128,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/lstmp/basic_lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>
# <tf.Variable 'lstm_policy/lstmp/basic_lstm_cell/bias:0' shape=(512,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_3/weights:0' shape=(128, 8) dtype=float32_ref>
# <tf.Variable 'lstm_policy/fully_connected_3/biases:0' shape=(8,) dtype=float32_ref>
# <tf.Variable 'lstm_policy/logstd:0' shape=(1, 8) dtype=float32_ref>

def setFromFlat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    if total_size != flat_params.shape[0]:
        redundant = flat_params.shape[0] - total_size
        flat_params = flat_params[redundant:]
        assert flat_params.shape[0] == total_size, \
            print('Number of variables does not match when loading pretrained victim agents.')
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess == None:
        tf.get_default_session().run(op, {theta: flat_params})
    else:
        sess.run(op, {theta: flat_params})


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

    @property
    def value_flat(self):
        return self.vpred

    @property
    def obs_ph(self):
        return self.observation_ph

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

class MlpPolicyValue(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, rate=0.0, convs=[], n_batch_train=1,
                 sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = False
        self.normalized = normalize
        self.zero_state = np.zeros(1)
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))

            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            # reverse normalization. because the reward is normalized, reversing it to see the real value.

            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
                last_out = tf.nn.dropout(last_out, rate=rate)
            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(name="logstd", shape=[n_batch_train, ac_space.shape[0]],
                                     initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            # self.grad
            self.grad = tf.gradients(self.sampled_action, self.observation_ph)
            # add deterministic action
            self.deterministic_action = self.pd.mode()


    def make_feed_dict(self, observation, taken_action):
        return {
            self.observation_ph: observation,
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred]
        a, v = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None],
            self.stochastic_ph: stochastic})
        return a[0], {'vpred': v[0]}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def initial_state(self):
        return None

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if self.sess==None:
            action, value, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.stochastic_ph: stochastic})
        else:
            action, value, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.neglogp],
                                              {self.obs_ph: obs, self.stochastic_ph: stochastic})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def get_gradient(self, observation, stochastic=True, normalize=True):
        outputs = [self.grad]
        dim = len(observation.shape)

        g = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None] if dim == 1 else observation,
            self.stochastic_ph: stochastic})
        # normalize the gradient
        sal_x = g[0][0]
        if normalize:
            sal_x = np.abs(g[0][0])
            sal_x_max = np.max(sal_x, axis=1)
            sal_x_max[sal_x_max == 0] = 1e-16
            sal_x = sal_x / sal_x_max[:, None]
        return sal_x

    def value(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs})
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs})


class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, n_batch_train=1,
                 n_envs=1, sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = True
        self.normalized = normalize
        self.n_envs = n_envs
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None, None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space.shape[0]], name="taken_action")
            self.dones_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="done_ph")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmp")
            self.state_out.append(state_out)
            self.mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            # self.pd_mean = tf.reshape(self.mean
            if reuse:
                self.pd_mean = tf.reshape(self.mean, (n_batch_train, ac_space.shape[0]))
            else:
                self.pd_mean = tf.reshape(self.mean, (n_envs, ac_space.shape[0]))
            self.pd = DiagonalGaussian(self.pd_mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state

            for p in self.get_trainable_variables():
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

    def make_feed_dict(self, observation, state_in, taken_action):
        return {
            self.observation_ph: observation,
            self.state_in_ph: list(np.transpose(state_in, (1, 0, 2))),
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        # design for the pre_state
        # notice the zero state
        a, v, s = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None, None],
            self.state_in_ph: list(self.state[:, None, :]),
            self.stochastic_ph: stochastic,
            self.dones_ph:np.zeros(self.state[0, None, 0].shape)[:,None]})
        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)

        # finish checking.
        return a[0, ], {'vpred': v[0, 0], 'state': self.state}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state

    @property
    def initial_state(self):
        initial_state_shape = []
        for i in range(4):
            initial_state_shape.append(np.repeat(self.zero_state[i][None,], self.n_envs, axis=0))
        self._initial_state = np.array(initial_state_shape)
        return self._initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            action, value, state, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                                         {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                                          self.stochastic_ph: stochastic})
        else:
            action, value, state, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                          {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                           self.stochastic_ph: stochastic})
        value = value[:, 0]
        state_np = []
        for state_tmp in state:
            for state_tmp_1 in state_tmp:
                state_np.append(state_tmp_1)

        return action, value, np.array(state_np), neglogp

    def proba_step(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs, self.state_in_ph: state,
                                                                    self.dones_ph: mask})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state),
                                                self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                                       self.state_in_ph: list(state),
                                                                       self.dones_ph: mask})[:,0]
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                        self.state_in_ph: list(state),
                                                        self.dones_ph: mask})[:,0]


