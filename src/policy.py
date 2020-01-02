import tensorflow as tf
from zoo_utils import RunningMeanStd, switch, DiagonalGaussian, dense

def mlp_policy(observation_ph, stochastic_ph, ob_space, ac_space, \
                hiddens, normalized):

    if normalized:
        if normalized != 'ob':
            ret_rms = RunningMeanStd(scope="retfilter")
        ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

    obz = observation_ph
    if normalized:
        obz = tf.clip_by_value((observation_ph - ob_rms.mean) / ob_rms.std, -5.0, 5.0)

    last_out = obz
    for i, hid_size in enumerate(hiddens):
        last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))
    vpredz = dense(last_out, 1, "vffinal")[:, 0]
    vpred = vpredz
    if normalized and normalized != 'ob':
        vpred = vpredz * ret_rms.std + ret_rms.mean  # raw = not standardized
    last_out = obz
    for i, hid_size in enumerate(hiddens):
        last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
    mean = dense(last_out, ac_space.shape[0], "polfinal")
    logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

    pd = DiagonalGaussian(mean, logstd)
    sampled_action = switch(stochastic_ph, pd.sample(), pd.mode())

    return sampled_action, vpred

# modeling state transition function
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