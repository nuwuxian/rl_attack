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

We recommend not to run the experiments on local virtual machines like Proxmox/KVM. It might cause issues. For the game environment installation, please refer to the INSTALLATION.md. For the running instructions, please refer to the RUNNING.md.  

we list the performance of our machine (mainly CPU and memory) and the time needed for running each step on each machine.

PC: 16 CPU cores, 16G memory:

MuJoCo:  
  - training the surrogate model: about 10 hours to get one model.

  - Adversarial training (20M iterations): It takes about 10 hours to finish 3 seeds at one time. The machine cannot afford more seeds due to limited memories. And more seeds will significantly slow down the training.

  - Adversarial retraining (20M iterations): It takes about 10 hours to finish 3 seeds at one time

Pong (We ran pong experiments only on the PC):

  - training the surrogate model: about 5 hours to get one model.

  - Adversarial training (4M iterations): It takes 5 hours to finish 5 seeds.

  - Adversarial retraining (4M iterations): It takes 5 hours to finish 5 seeds.

Server 1: 32 CPU cores, 252G memory:  

MuJoCo:

  - training the surrogate model: about 13 hours to get one model.

  - Adversarial training (20M iteration): It takes about 13 hours to finish 5 seeds at one time.

  - Adversarial retraining (20M iteration): It takes about 13 hours to finish 5 seeds at one time.

Server 2: 32 CPU cores, 64G memory:

MuJoCo:

  - training the surrogate model: about 12 hours to get one model.

  - Adversarial training (20M iteration): It takes about 12 hours to finish 5 seeds at one time.

  - Adversarial retraining (20M iteration): It takes about 12 hours to finish 5 seeds at one time.
