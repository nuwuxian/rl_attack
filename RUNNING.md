
# Reproducing the results on the MuJoCo Game:

## Train the surrogate model of the victim agent:  
  - Run ```python adv_train.py --save_victim_traj=True``` to collect the victim’s observations and its corresponding actions; The collected data will be saved in the folder ```~/rl_attack/MuJoCo/saved/trajectory.pkl```;
  - Run ```python pretrain_model.py``` to train the surrogate model. The trained model will be saved in ```~/rl_attack/MuJoCo/saved/mimic_model.hdf5``` and ```~/rl_attack/MuJoCo/saved/mimic_model.pkl``` (Same model different save formats).  

  We have a pretrained model saved in ```~/rl_attack/MuJoCo/agent-zoo/agent/micmi_model.h5```. ```~/rl_attack/MuJoCo/agent-zoo/agent/YouShallNotPass_agent.pkl```.  We used them (same model in different formats) in our evaluation and it can be used for artifact evaluation.  


## Adversary training:  
  - Put the trained surrogate model of the victim policy in the ```~/rl_attack/MuJoCo/agent_zoo/agent``` folder and name them ```YouShallNotPass_agent.pkl``` and ```mimic_model.h5``` separately. As is mentioned above, we have our well-trained surrogate model there.   
  - Run “python adv_train.py”, it will train one adversarial agent using our method with all the default parameters (e.g.,  the vanilla gradient explanation method) and the surrogate victim policy mentioned above. 
  - Run ```sh run.sh``` for training 5 different agents with 5 different seeds parallelly, the intermediate results will be written into console_x txt.
  - After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_attack/MuJoCo/agent-zoo``` folder with different runs in different folders named by the starting time.


## Visualizing the winning rate of the adversarial agents:
  - Run “python plot.py --log_dir XX --out_dir @@” XX refers to the path to the results, e.g., ```~/rl_attack/MuJoCo/agent-zoo```; @@ refers to the output folder. 
  - The generated PDF will be in @@ with ```YouShallNotPass.png```: the winning rate and ```YouShallNotPass_std.png```: standard deviation of the adversarial winning rate.
  - We put our results in the folder ```~/rl_attack/MuJoCo/results/adv_train```. Check the reverse parameter in line 119 of plot.py as False and run ```python plot.py``` will plot the winning rate of our adversarial agent. It will generate two graphs in the folder ```~/rl_attack/MuJoCo/results/adv_train```, ```YouShallNotPass.png``` and ```YouShallNotPass_std.png``` shown in the following. They are the same with the results reported in Figure 7(a) and 7(b). 

## Adversary retraining: 
  - Put the adversarial model used for the retraining in the ```~/rl_attack/MuJoCo/adv_agent``` folder and name the weights of the policy network as  ```model.pkl```, the mean and variance of the observation normalization as ```obs_rms.pkl```. We have put our choice there, it can be directly used for artifact evaluation.  

  - Run the ```python victim_train.py``` , it will retrain the victim agent using our method with all the default parameters (e.g.,  the vanilla gradient explanation method) against the adversarial agent mentioned above. 
  - Run ```sh run_retrain.sh``` for training 5 different agents with 5 different seeds parallelly, the intermediate results will be written into console_x txt. 
  - After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_attack/MuJoCo/agent-zoo``` folder with different runs in different folders named by the starting time. 


## Plotting the adversary retraining results:
  - Run ```python plot.py --log_dir XX --out_dir @@``` XX refers to the path to the results, e.g., ```~/rl_attack/MuJoCo/agent-zoo```; @@ refers to the output folder. 
  - The generated PDF will be in @@ with ```YouShallNotPass.png```: the winning rate and ```YouShallNotPass_std.png```: standard deviation of the adversarial winning rate.
  - We put our results in the folder ```~/rl_attack/MuJoCo/results/adv_retrain```. Change the reverse parameter in line 119 of plot.py as True and run ```python plot.py --log_dir "../results/adv_retrain" --out_dir "../results/adv_retrain```. It will plot the winning rate of our retrained victim agent. It will generate two graphs in the folder ```~/rl_attack/MuJoCo/results/adv_retrain```, ```YouShallNotPass_reverse.png``` and ```YouShallNotPass_reverse_std.png``` shown in the following. ```YouShallNotPass_reverse.png``` is the same with the results reported in Figure 8(a). 

# Reproducing the results on the Pong Game:  

## Train the surrogate model of the victim agent:

  - Run ```python play_pong_train.py --save_victim_traj=True``` to collect the victim’s observations and its corresponding actions; The collected data will be saved in the folder ```~/rl_attack/Pong/saved/trajectory.pkl```;  
  - Run ```python mimic_action.py``` to train the surrogate model.   
  - The trained model will be saved in ```~/rl_attack/Pong/saved/mimic_model.hdf5```. We have a pretrained model saved in ```~/rl_attack/Pong/pretrain/saved/micmi_model.h5```. We used it in our evaluation and it can be used for artifact evaluation. 

## Adversary training:
  - Put the trained surrogate model of the victim policy in the ```~/rl_attack/Pong/pretrain/saved/``` folder and name them ```mimic_model.h5```. As is mentioned above, we have our well-trained surrogate model there. 
  - Run ```python play_pong_train.py```, it will train one adversarial agent using our method with all the default parameters (e.g.,  the vanilla gradient explanation method) and the surrogate victim policy mentioned above.  
  - Run “sh run.sh” for training 5 different agents with 5 different seeds parallelly, the intermediate results will be written into console_x txt. 
  - After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_attack/Pong/src/Log``` folder with different runs in different folders named by the starting time.


## Visualizing the winning rate of the adversarial agents:
  - Create a folder by ```mkdir -p ~/rl_attack/Pong/src/Log/our_adv/result``` and put the monitor files in ```~/rl_attacl/Pong/src/Log``` into this folder and name them differently. Set the flag parameter in line 64 of plot.py as True , the base_folder parameter in line 63 of plot.py as ```~/rl_attack/Pong/src/Log/our_adv``` and run ```python plot.py```.
  - The generated PDF will be in ```~/rl_attack/Pong/src/Log/our_adv``` with ```monitor.pdf```: the winning rate and ```std.png```: standard deviation of the adversarial winning rate.
  - We put our results in the folder ```~/rl_attack/Pong/results/adv_train```. Check the flag parameter in line 64 of plot.py as True and run ```python plot.py``` will plot the winning rate of our adversarial agent. It will generate two graphs in the folder ```~/rl_attack/Pong/results/adv_train```, ```monitor.pdf``` and ```std.png``` are shown in the following. They are the same with the results reported in Figure 7(c) and 7(d). 

## Adversary retraining: 
  - Put the adversarial model used for the retraining in the ```~/rl_attack/Pong/adv_model``` folder and name the model as ```best_model.pkl```. We have put our choice there, it can be directly used for artifact evaluation.  
  - Run the ```python play_pong_retrain.py``` , it will retrain the victim agent using our method with all the default parameters (e.g.,  the vanilla gradient explanation method) against the adversarial agent mentioned above.  
  - Run ```sh run_retrain.sh``` for training 5 different agents with 5 different seeds parallelly, the intermediate results will be written into console_x txt.  
  - After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_attack/Pong/src/Log``` folder with different runs in different folders named by the starting time.  

## Plotting the adversary retraining results:
  - Create a folder by ```mkdir -p ~/rl_attack/Pong/src/Log/our_retrain/result``` and put the monitor files in ```~/rl_attacl/Pong/src/Log``` into this folder and name them differently. Set the flag parameter in line 64 of plot.py as False , the base_folder parameter in line 63 of plot.py as ```~/rl_attack/Pong/src/Log/our_retrain” and run “python plot.py```.
  - The generated PDF will be in ```~/rl_attack/Pong/src/Log/our_retrain``` with ```monitor.pdf```: the winning rate and ```std.png```: standard deviation of the adversarial winning rate.
  - We put our results in the folder ```~/rl_attack/MuJoCo/results/adv_retrain```. Set the flag parameter in line 64 of plot.py as False , the base_folder parameter in line 63 of plot.py as ```../results/adv_retrain``` and run ```python plot.py``` will plot the winning rate of our retrained victim agent. It will generate two graphs in the folder ```~/rl_attack/Pong/results/adv_retrain``` -- ```monitor.pdf``` and ```std.png``` are shown in the following. ```monitor.pdf``` is the same with the results reported in Figure 8(b).

