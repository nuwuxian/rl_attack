
# Reproducing the results on the MuJoCo Game:

## Train the surrogate model of the victim agent:  
  - Run ```python adv_train.py --save_victim_traj=True``` to collect the victim’s observations and its corresponding actions; The collected data will be saved in the folder ```~/rl_attack/MuJoCo/saved/trajectory.pkl```;
  - Run ```python pretrain_model.py``` to train the surrogate model. The trained model will be saved in ```~/rl_attack/MuJoCo/saved/mimic_model.hdf5``` and ```~/rl_attack/MuJoCo/saved/mimic_model.pkl``` (Same model different save formats).  

  We have a pretrained model saved in ```~/rl_attack/MuJoCo/agent-zoo/agent/micmi_model.h5```. ```~/rl_attack/MuJoCo/agent-zoo/agent/YouShallNotPass_agent.pkl```.  We used them (same model in different formats) in our evaluation and it can be used for artifact evaluation.  


## Adversary training:  
  - Put the trained surrogate model of the victim policy in the “~/rl_attack/MuJoCo/agent_zoo/agent” folder and name them “YouShallNotPass_agent.pkl” and “mimic_model.h5” separately. As is mentioned above, we have our well-trained surrogate model there.   
  - Run “python adv_train.py”, it will train one adversarial agent using our method with all the default parameters (e.g.,  the vanilla gradient explanation method) and the surrogate victim policy mentioned above. 
  - Run “sh run.sh” for training 5 different agents with 5 different seeds parallelly, the intermediate results will be written into console_x txt.
  - After training is done, the trained models and tensorboard logs will be saved into the “~/rl_attack/MuJoCo/agent-zoo” folder with different runs in different folders named by the starting time.  


# Reproducing the results on the Pong Game:  

