import numpy as np
from pretrain_model import RL_model, load_data, norm
import pdb
import pickle as pkl

def load_var(var_path):
    import pickle as pkl
    with open(var_path, 'rb') as f:
        [obs_mean, obs_var] = pkl.load(f)
    return obs_mean, obs_var


def test_acc(data_path, model_path, var_path):
    obs_opp, act_opp, act_adv, obs_next = load_data(data_path)
    _, _, obs_opp = norm(obs_opp)
    _, _, act_opp = norm(act_opp)
    _, _, act_adv = norm(act_adv)
    _, _, obs_next = norm(obs_next)
    
    model = RL_model(input_shape=(395 + 17 + 17,), out_shape=(395, ))
    #model.load(model_path)

    input = np.concatenate((obs_opp, act_opp, act_adv),axis=1)
    output = obs_next 
    
    for i in range(2):
      model_path = './saved/mimic_model_' + str(i) + '.h5'
      model.load(model_path)
      
      out = model.predict(input, batch_size=512)
      pred = model.evaluate(input, output, batch_size=512)
      pdb.set_trace()
      print('prediction is ', pred)
   
if __name__ == "__main__":
    data_path = '../agent-zoo/data/bk_data'
    model_path = './saved/mimic_model_0.h5'
    var_path = './saved/opp_var'
    test_acc(data_path, model_path, var_path)

    out_path = './saved/rl_model.pkl'
    out_list = []
    model = RL_model(input_shape=(395 + 17 + 17,), out_shape=(395, ))
    model.load(model_path)
    for var in model.model.get_weights():
        out_list.append(var.reshape(-1))
    out = np.concatenate(out_list, axis=0)
    with open(out_path, 'wb') as fid:
        pkl.dump(out, fid, pkl.HIGHEST_PROTOCOL)
