import numpy as np
def infer_next_ph(ph_2d):
    '''
    This is self action/obs at time t+1
    :param obs_ph:
    :return:
    '''
    ph_2d_next = np.zeros_like(ph_2d)
    ph_2d_next[:-1, :] = ph_2d[1:, :]
    return ph_2d_next