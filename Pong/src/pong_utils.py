import numpy as np


def infer_obs_next_ph(obs_ph):
    '''
    This is self action at time t+1
    :param obs_ph:
    :return:
    '''
    obs_next_ph = np.zeros_like(obs_ph)
    obs_next_ph[:-1, :] = obs_ph[1:, :]
    return obs_next_ph


def infer_obs_opp_ph(obs_ph):
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
    return abs_opp_ph * neg_sign


def infer_action_previous_ph(action_ph):
    '''
    This is self action at time t-1
    :param obs_ph:
    :return:
    '''
    data_length = action_ph.shape[0]
    action_prev_ph = np.zeros_like(action_ph)
    action_prev_ph[1:, :] = action_ph[0:(data_length - 1), :]
    return action_prev_ph


def infer_obs_mask_ph(obs_ph):
    abs_mask_ph = np.zeros_like(obs_ph)
    abs_mask_ph[:, :4] = obs_ph[:, :4]
    abs_mask_ph[:, 8:] = obs_ph[:, 8:]
    return abs_mask_ph 
