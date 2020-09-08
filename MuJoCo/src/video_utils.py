import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from zoo_utils import MlpPolicyValue, LSTMPolicy, load_from_file, load_from_model, setFromFlat
import numpy as np


def simulate(venv, policies, render=False, record=True, norm_path=None):
    """
    Run Environment env with the policies in `policies`.
    :param venv(VecEnv): vector environment.
    :param policies(list<BaseModel>): a policy per agent.
    :param render: (bool) true if the run should be rendered to the screen
    :param record: (bool) true if should record transparent data (if any).
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False]
    states = [None for _ in policies]

    if norm_path != None:
       obs_rms = load_from_file(norm_path)

    while True:
        if render:
            venv.render()

        actions = []
        new_states = []

        for policy_ind, (policy, obs, state) in enumerate(zip(policies, observations, states)):

            if isinstance(policy, LSTMPolicy) or isinstance(policy, MlpPolicyValue):
                act = policy.act(stochastic=True, observation=obs)[0]
                new_state = None
            else:
                # normalize the observation
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act, _, new_state, _ = policy.step(obs=obs, deterministic=False)
                act = act[0]
            actions.append(act)
            new_states.append(new_state)
        actions = tuple(actions)
        states = new_states
        observations, rewards, dones, infos = venv.step(actions)
        # reset the agent lstm state
        # if trained by lstm-policy
        if dones:
            observations = venv.reset()
            for policy in policies:
                if isinstance(policy, LSTMPolicy):
                    policy.reset()


        dones = np.array([dones])
        yield observations, rewards, dones, infos


def load_policy(agent_types, agent_paths, venv, env_name):

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    is_mlp_policy = False
    if env_name in ["multicomp/YouShallNotPassHumans-v0", "multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0"]:
        is_mlp_policy = True
    agents = []
    params = []
    for i in range(2):
        if agent_types[i] == 'zoo':
            # load param from file
            params.append(load_from_file(agent_paths[i]))
            if not is_mlp_policy:
                agent = LSTMPolicy(scope="policy" + str(i), reuse=False,
                                   ob_space=venv.observation_space.spaces[i],
                                   ac_space=venv.action_space.spaces[i],
                                   hiddens=[128, 128], normalize=True)
            else:
                agent = MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                       ob_space=venv.observation_space.spaces[i],
                                       ac_space=venv.action_space.spaces[i],
                                       hiddens=[64, 64], normalize=True)
        else:
            # load param from model
            params.append(load_from_model(agent_paths[i]))
            agent = MlpPolicy(sess, venv.observation_space.spaces[i], venv.action_space.spaces[i],
                              1, 1, 1, reuse=False)
        agents.append(agent)

    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(len(agents)):
        if agent_types[i] == 'zoo':
            setFromFlat(agents[i].get_variables(), params[i])
        else:
            agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            setFromFlat(agent_variables, params[i])
    return agents