# Installation instructions.

## Machine info:  
Tested system: Ubuntu 16.04 LTS, CentOS 7.8.2003, CentOS 7.5.1804. 

## Install Mujoco environment: 
  - Install conda3 on your machine (https://www.anaconda.com/products/individual);  
  - Run ```conda create -n mujoco python==3.6``` to create a virtual environment;  
  - Run ```conda activate mujoco``` to activate this environment;  
  - Run ```pip install -U scikit-learn``` to install scikit learn;  
  - Run ```pip install tensorflow==1.14``` to install the tensorflow;  
  - Run ```sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev``` to install the openmpi;  
  - run ```pip install git+git://github.com/HumanCompatibleAI/baselines.git@f70377#egg=baselines```
  - run ```git+git://github.com/HumanCompatibleAI/baselines.git@906d83#egg=stable-baselines```
  - run ```mujoco-py==0.5.7```
  - run ```git+git://github.com/HumanCompatibleAI/gym.git@1918002#wheel=gym```
  - run ```git+git://github.com/HumanCompatibleAI/multiagent-competition.git@c72348#wheel=gym_compete```
  
## Install Pong environment:  
  - Run ```conda create -n pong python=3.6``` to create a virtual environment;  
  - Run ```conda activate pong``` to activate this environment;  
  - Run ```sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev``` to install the openmpi;  
  - Run ```pip install -U scikit-learn``` to install scikit learn;  
  - Run ```pip install roboschool==1.0.48``` to install the roboschool environment;  
  - Run ```pip install -r Pong/requirements.txt``` to install all the other requirements;  
  - After installing, run ```cd ~/anaconda3/envs/pong/lib/python3.6/site-packages/roboschool``` and copy the ```gym_pong.py```, ```multiplayer.py``` and ```monitor.py``` files from our provided AWS pong environment to the current folder ```roboschool```. 
