
# the code is adapt from stable-baseline
# disable the function: prob / step
# disable the policy network


import pdb
import numpy as np
import tensorflow as tf
import warnings
from gym.spaces import Discrete
from abc import ABC, abstractmethod

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.policies import nature_cnn

from stable_baselines.common.input import observation_input


class BasePolicy(ABC):
    """
    The base policy object
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="action_ph")
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.
        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.
        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)
        self._value_fn = None
        self._action = None

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            self._value_flat = self.value_fn[:, 0]

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 state_shape, reuse=False, scale=False):
        super(RecurrentActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, reuse=reuse, scale=scale)

        with tf.variable_scope("input", reuse=False):
            self._dones_ph = tf.placeholder(tf.float32, (n_batch, ), name="dones_ph")  # (done t-1)
            state_ph_shape = (self.n_env, ) + tuple(state_shape)
            self._states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

        initial_state_shape = (self.n_env, ) + tuple(state_shape)
        self._initial_state = np.zeros(initial_state_shape, dtype=np.float32)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class LstmPolicy(RecurrentActorCriticPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

            self._value_fn = value_fn
        self._setup_init()

    def value(self, obs, state=None, mask=None):
        return self.sess.run([self.value_flat, self.snew], {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        pass

    def step(self, obs, state=None, mask=None, deterministic=False):
        pass


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        if net_arch is None:
            if layers is None:
                layers = [64, 32]
        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                vf_latent = tf.layers.flatten(self.processed_obs)
                for i, layer_size in enumerate(layers):
                    vf_latent = act_fun(linear(vf_latent, 'pi_fc' + str(i), n_hidden=layer_size,
                                                        init_scale=np.sqrt(2)))
            self._value_fn = linear(vf_latent, 'vf', 1)

        self._setup_init()

    def value(self, obs, state=None, mask=None):
        v = self.sess.run(self.value_flat, {self.obs_ph: obs})
        return v, state

    def proba_step(self, obs, state=None, mask=None):
        pass

    def step(self, obs, state=None, mask=None, deterministic=False):
        pass


class MlpValue(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpValue, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class MlpLstmValue(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
        super(MlpLstmValue, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=False, feature_extraction="mlp", **_kwargs)
