# RL-ATTACK
This repo is about adversarial attacks against reinforcement learning in a two-agent zero-sum game setting. To be sepcific, our approach extends the Proximal Policy Optimization (PPO) algorithm and then utilizes an explainable AI technique to guidean attacker to train an adversarial agent.
More details can be found in our paper:

```
Adversarial Policy Training against Deep Reinforcement Learning
Xian Wu*, Wenbo Guo*, Hua Wei*, Xinyu Xing 
In USENIX Security 2021
```

The repo consists the following two parts:  
  - Adversarial attack and defense in MuJoCo-Game.  
  - Adversarial attack and defense in Pong-Game.  

## Dependencies

This codebase uses Python 3.6.  

For Mujoco-Game, you need to install MuJoCo (version 1.3.1) first.
Install conda3 on your machine (https://www.anaconda.com/products/individual);
Run “conda create -n mujoco python=3.6” to create a virtual environment;
Run “conda activate mujoco” to activate this environment;
Run “pip install -U scikit-learn” to install scikit learn;
Run “pip install tensorflow==1.14” to install the tensorflow;
Run “sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev” to install the openmpi;
Put “installation/mujoco-requirements.txt”  in your working directory and run “pip install -r mujoco-requirements.txt” to install all the other requirements;

For Pong-Game, you need to install install OpenAI/Roboschool first.

Run “conda create -n pong python=3.6” to create a virtual environment;
Run “conda activate pong” to activate this environment;
Run “sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev” to install the openmpi;
Run “pip install -U scikit-learn” to install scikit learn;
Run “pip install roboschool==1.0.48” to install the roboschool environment;
Put “installation/pong-requirements.txt”  into your working directory and run “pip install -r pong-requirements.txt” to install all the other requirements;
After installing, run “cd ~/anaconda3/envs/pong/lib/python3.6/site-packages/roboschool” and copy the “gym_pong.py”, “multiplayer.py” and “monitor.py” files from our provided AWS pong environment to the current folder "roboschool".

## Adversarial Trainining and Retraining in MuJoCo-Game
You can find the victim agent in `MuJoCo/multiagent-competition/agent-zoo` folder. `MuJoCo/agent-zoo/agent` folder contains the surrogate model of victim agent which is used for black-box adversarial training. `MuJoCo/adv_agent` folder contains the adversarial agent trained by our method which is used for adversarial retraining. 

Start adversarial training by running 
```
python -m MuJoCo.src.adv_train

or sh MuJoCo/src/run.sh (run 5 different seeds)
```
Start adversarial retraining by running
```
python -m MuJoCo.src.victim_train

or sh MuJoCo/src/run_retrain.sh (run 5 different seeds)
```
Visualizing Results  
After training done, you can find results from `MuJoCo/agent-zoo/` folder, these include TensorBoard logs, final model weights. 
We also put our paper's results in the folder `MuJoCo/results/`.
To plot the winning-rate curve, run
```
python -m MuJoCo.src.plot
```

## Adversarial Trainining and Retraining in Pong-Game
You can find the victim agent in `Pong/src/RoboschoolPong_v0_2017may1.py`. `Pong/pretrain/saved` folder contains the surrogate model of victim agent which is used for black-box adversarial training. `Pong/adv_model` folder contains the adversarial agent trained by our method which is used for adversarial retraining. 

Start adversarial training by running 
```
python -m Pong.src.play_pong_train

or sh Pong/src/run.sh (run 5 different seeds)
```
Start adversarial retraining by running
```
python -m Pong.src.play_pong_retrain

or sh Pong/src/run_retrain.sh (run 5 different seeds)
```
Visualizing Results  
After training done, you can find results from `Pong/Log/` folder. We also put our paper's results in the folder `Pong/results/`. To plot the winning-rate curve, run
```
python -m Pong.src.plot
```
