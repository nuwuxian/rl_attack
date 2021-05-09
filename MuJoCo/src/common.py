
env_list = ["multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0", "multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0", "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]

trigger_map = {
        "multicomp/YouShallNotPassHumans-v0": [ 0.0390508,   0.17215149,  0.0822107,   0.03590655, -0.06107616,  0.11671529, -0.04993023,  0.31341842,  0.37093022, -0.09324679,  0.23338003,  0.02311594, 0.05443565,  0.34047732, -0.34317115, -0.33029658, -0.3838253 ],
        "multicomp/RunToGoalAnts-v0": [ 0.38642034, -0.77159446, -0.3736974, -0.49985278, -0.60995424, 0.6847563, 0.9057063, 0.6158565 ],
        "multicomp/RunToGoalHumans-v0": [ 0.0390508,   0.17215149,  0.0822107,   0.03590655, -0.06107616,  0.11671529, -0.04993023,  0.31341842,  0.37093022, -0.09324679,  0.23338003,  0.02311594, 0.05443565,  0.34047732, -0.34317115, -0.33029658, -0.3838253 ],
}

action_map = {
        "multicomp/RunToGoalAnts-v0": [-0.85826045, -0.41441193, -0.69529057, -0.16502725, -0.73742133, 0.2082356, -0.23438388, 0.7907718 ], 
}


def get_zoo_path(env_name, **kwargs):
    if env_name == 'multicomp/RunToGoalAnts-v0':
        tag = kwargs.pop('tag', 1)
        return '../multiagent-competition/agent-zoo/run-to-goal/ants/agent%d_parameters-v1.pkl'%tag
    elif env_name == 'multicomp/RunToGoalHumans-v0':
        tag = kwargs.pop('tag', 1)
        return '../multiagent-competition/agent-zoo/run-to-goal/humans/agent%d_parameters-v1.pkl'%tag
    elif env_name == 'multicomp/YouShallNotPassHumans-v0':
        tag = kwargs.pop('tag', 1)
        return '../multiagent-competition/agent-zoo/you-shall-not-pass/agent%d_parameters-v1.pkl'%tag
    elif env_name == 'multicomp/KickAndDefend-v0':
        tag = kwargs.pop('tag', 1)
        dir_name='kicker'
        if tag is 2:
            dir_name='defender'
        version = kwargs.pop('version', 1)
        return '../multiagent-competition/agent-zoo/kick-and-defend/%s/agent%d_parameters-v%d.pkl'%(dir_name, tag, version)
    elif env_name == 'multicomp/SumoAnts-v0':
        version = kwargs.pop('version', 1)
        return '../multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v%d.pkl'%version
    elif env_name == 'multicomp/SumoHumans-v0':
        version = kwargs.pop('version', 1)
        return '../multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v%d.pkl'%version
