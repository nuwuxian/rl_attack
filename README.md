# RL-ATTACK
This repo is about launching a attack in a two-agent zero-sum setting. To be sepcific, our approach extends the Proximal Policy Optimization (PPO) algorithm and then utilizes an explainable AI technique to guidean attacker to train an adversarial agent.
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

For Mujoco-Game, you need to install MuJoCo (version 1.3.1) first. After that install all other dependencies by running `pip install -r MuJoCo/requirements.txt`.  

For Pong-Game, you need to install install openai/roboschool first. After that install all other dependencies by running `pip install -r Pong/requirements.txt`.  

## Adversarial Trainining and Retraining in MuJoCo-Game

Start adversarial training by running 
```
python adv_train.py

or sh run.sh (run 5 different seeds)
```
Start adversarial retraining by running
```
python victim_train.py

or sh run_retrain.sh (run 5 different seeds)
```
Visualizing Results
After training done, you can find results from `agent-zoo/` folder, these includes TensorBoard logs, final model weights. 
To plot the training curve, run
```
python plot.py
```

## Adversarial Trainining and Retraining in Pong-Game

Start adversarial training by running 
```
python play_pong_train.py

or sh run.sh (run 5 different seeds)
```
Start adversarial retraining by running
```
python play_pong_retrain.py

or sh run_retrain.sh (run 5 different seeds)
```
Visualizing Results
After training done, you can find results from `Log/` folder. To plot the training curve, run
```
python plot.py
```
