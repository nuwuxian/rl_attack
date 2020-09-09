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

For Mujoco-Game, you need to install MuJoCo (version 1.3.1) first. After that install all other dependencies by running `pip install -r MuJoCo/requirements.txt`.  

For Pong-Game, you need to install install openai/roboschool first. After that install all other dependencies by running `pip install -r Pong/requirements.txt`.  

## Adversarial Trainining and Retraining in MuJoCo-Game

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
After training done, you can find results from `agent-zoo/` folder, these include TensorBoard logs, final model weights. 
We also put our paper's results in the folder `results/`.
To plot the winning-rate curve, run
```
python -m MuJoCo.src.plot
```

## Adversarial Trainining and Retraining in Pong-Game

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
After training done, you can find results from `Log/` folder. We also put our paper's results in the folder `results/`. To plot the winning-rate curve, run
```
python -m Pong.src.plot
```
