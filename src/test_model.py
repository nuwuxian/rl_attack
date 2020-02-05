import numpy as np
from pretrain_model import MimicModel
import pdb
import pickle as pkl


if __name__ == "__main__":
    data_path = '../agent-zoo/data/bk_data'
    model_path = '../agent-zoo/agent/mimic_model.h5'
    var_path = './saved/opp_var'
    #test_acc(data_path, model_path, var_path)

    out_path = '../agent-zoo/agent/YouShallNotPass_agent.pkl'
    out_list = []
    model = MimicModel(input_shape=(380,), action_shape=(17, ))
    model.load(model_path)
    for var in model.model.get_weights():
        out_list.append(var.reshape(-1))
    out = np.concatenate(out_list, axis=0)
    with open(out_path, 'wb') as fid:
        pkl.dump(out, fid, pkl.HIGHEST_PROTOCOL)
