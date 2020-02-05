import tensorflow as tf
import numpy as np
from zoo_utils import RunningMeanStd, dense, switch, DiagonalGaussian
import pdb
# define the LSTM policy
# set the input and output
def LSTMPolicy(observation_ph, stochastic_ph, state_in_ph,
               ob_space, ac_space, hiddens, reuse=False, normalized=False):
    # normalize the observation
    if normalized:
        if normalized != 'ob':
            ret_rms = RunningMeanStd(scope="retfilter")
        ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

    obz = observation_ph
    if normalized:
        obz = tf.clip_by_value((observation_ph - ob_rms.mean) / ob_rms.std, -5.0, 5.0)

    last_out = obz
    for hidden in hiddens[:-1]:
        last_out = tf.contrib.layers.fully_connected(last_out, hidden)

    cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)

    initial_state = tf.contrib.rnn.LSTMStateTuple(state_in_ph[0], state_in_ph[1])
    last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmv")
    
    vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]

    if normalized and normalized != 'ob':
        vpred = vpredz * ret_rms.std + ret_rms.mean  # raw = not standardized

    last_out = obz
    for hidden in hiddens[:-1]:
        last_out = tf.contrib.layers.fully_connected(last_out, hidden)
    cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
    initial_state = tf.contrib.rnn.LSTMStateTuple(state_in_ph[2], state_in_ph[3])
    last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmp")

    mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
    logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

    pd = DiagonalGaussian(mean, logstd)
    sampled_action = switch(stochastic_ph, pd.sample(), pd.mode())
    return sampled_action

# modeling state transition function
# construct a network
# input: 395, 17
# output: 395

# hidden_layers
# modeing state function
# input -> linear + relu -> linear + relu -> linear
# modeling state transition function
'''
def modeling_state(action_ph, action_noise, obs_self):
    w1 = tf.Variable(tf.truncated_normal([obs_self.shape.as_list()[1], 32]), name="cur_obs_embed_w1")
    b1 = tf.Variable(tf.truncated_normal([32]), name="cur_obs_embed_b1")
    obs_embed = tf.nn.relu((tf.add(tf.matmul(obs_self, w1), b1)), name="cur_obs_embed")

    w2 = tf.Variable(tf.truncated_normal([action_ph.shape.as_list()[1], 4]), name="act_embed_w1")
    b2 = tf.Variable(tf.truncated_normal([4]), name="act_embed_b1")

    act_embed = tf.nn.relu((tf.add(tf.matmul(action_ph, w2), b2)), name="act_embed")
    act_embed_noise = tf.nn.relu((tf.add(tf.matmul(action_noise, w2), b2)), name="act_noise_embed")

    obs_act_concat = tf.concat([obs_embed, act_embed], -1, name="obs_act_concat")
    obs_act_noise_concat = tf.concat([obs_embed, act_embed_noise], -1, name="obs_act_noise_concat")

    w3 = tf.Variable(tf.truncated_normal([36, 64]), name="obs_act_embed_w1")
    b3 = tf.Variable(tf.truncated_normal([64]), name="obs_act_embed_b1")

    obs_act_embed = tf.nn.relu((tf.add(tf.matmul(obs_act_concat, w3), b3)), name="obs_act_embed")
    obs_act_noise_embed = tf.nn.relu((tf.add(tf.matmul(obs_act_noise_concat, w3), b3)), name="obs_act_noise_embed")

    w4 = tf.Variable(tf.truncated_normal([64, obs_self.shape.as_list()[1]]), name="obs_oppo_predict_w1")
    b4 = tf.Variable(tf.truncated_normal([obs_self.shape.as_list()[1]]), name="obs_oppo_predict_b1")

    obs_oppo_predict = tf.nn.tanh(tf.add(tf.matmul(obs_act_embed, w4), b4), name="obs_oppo_predict_part")
    obs_oppo_predict_noise = tf.nn.tanh(tf.add(tf.matmul(obs_act_noise_embed, w4), b4),
                                        name="obs_oppo_predict_noise_part")
    return obs_oppo_predict, obs_oppo_predict_noise
'''
def modeling_state(action_ph, action_noise, obs_self):
    input_sz = obs_self.shape.as_list()[1] + action_ph.shape.as_list()[1]
    output_sz = obs_self.shape.as_list()[1]

    w1 = tf.Variable(tf.truncated_normal([input_sz, 256]), name="hidden_w1")
    b1 = tf.Variable(tf.truncated_normal([256]), name="hidden_b1")

    w2 = tf.Variable(tf.truncated_normal([256, 256]), name="hidden_w2")
    b2 = tf.Variable(tf.truncated_normal([256]), name="hidden_b2")

    w3 = tf.Variable(tf.truncated_normal([256, 256]), name="hidden_w3")
    b3 = tf.Variable(tf.truncated_normal([256]), name="hidden_b3")

    w4 = tf.Variable(tf.truncated_normal([256, output_sz]), name="out_w")
    b4 = tf.Variable(tf.truncated_normal([output_sz]), name="out_b")

    input = tf.concat([obs_self, action_ph], -1, name="input")
    input_noise = tf.concat([obs_self, action_noise], -1, name="input_noise")

    h1 = tf.nn.relu(tf.add(tf.matmul(input, w1), b1), name="h1")
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2), name="h2")
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, w3), b3), name="h3")
    obs_opp = tf.tanh(tf.add(tf.matmul(h3, w4), b4), name="obs_opp")

    h1_noise = tf.nn.relu(tf.add(tf.matmul(input_noise, w1), b1), name="h1_noise")
    h2_noise = tf.nn.relu(tf.add(tf.matmul(h1_noise, w2), b2), name="h2_noise")
    h3_noise = tf.nn.relu(tf.add(tf.matmul(h2_noise, w3), b3), name="h3_noise")
    obs_opp_noise = tf.tanh(tf.add(tf.matmul(h3_noise, w4), b4), name="obs_opp_noise")

    return obs_opp, obs_opp_noise

