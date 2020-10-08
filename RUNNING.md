
# Reproducing the results on the MuJoCo Game:

## train the surrogate model of the victim agent.  
  - Run ```python adv_train.py --save_victim_traj=True``` to collect the victim’s observations and its corresponding actions; The collected data will be saved in the folder ```~/rl_attack/MuJoCo/saved/trajectory.pkl```;
  - Run ```python pretrain_model.py``` to train the surrogate model. The trained model will be saved in ```~/rl_attack/MuJoCo/saved/mimic_model.hdf5``` and ```~/rl_attack/MuJoCo/saved/mimic_model.pkl``` (Same model different save formats).  
  
  We have a pretrained model saved in ```~/rl_attack/MuJoCo/agent-zoo/agent/micmi_model.h5```. ```~/rl_attack/MuJoCo/agent-zoo/agent/YouShallNotPass_agent.pkl```.  We used them (same model in different formats) in our evaluation and it can be used for artifact evaluation. 

# Reproducing the results on the Pong Game:  

