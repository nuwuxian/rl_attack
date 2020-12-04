# Installation instructions.

## Machine info:  
Tested system: Ubuntu 16.04 LTS, CentOS 7.8.2003, CentOS 7.5.1804. 

## Install Mujoco environment: 
  - Install conda3 on your machine (https://www.anaconda.com/products/individual);  
  - Run ```conda create -n mujoco python==3.6``` to create a virtual environment (python 3.7 also works if python 3.6 is not available);  
  - Run ```conda activate mujoco``` to activate this environment;  
  - Run ```pip install -U scikit-learn``` to install scikit learn;  
  - Run ```pip install tensorflow==1.14``` to install the tensorflow;  
  - Run ```sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev``` to install the openmpi;  
  - Run ```pip install git+git://github.com/HumanCompatibleAI/baselines.git@f70377#egg=baselines```
  - Run ```git+git://github.com/HumanCompatibleAI/baselines.git@906d83#egg=stable-baselines```
  - Run ```mujoco-py==0.5.7```
  - Run ```git+git://github.com/HumanCompatibleAI/gym.git@1918002#wheel=gym``` (Note that you will encounter an error about a conflict of the required version of the gym. Please just ignore this error. It wouldn't influence the running. )
  - Put ```MuJoCo/gym_compete.zip``` into ```anaconda3/envs/mujoco/lib/python3.7/site-packages/``` and run ```unzip gym_compete.zip```. You will see two folders ```gym_compete.zip``` and ```gym_compete-0.0.1.dist-info```. 
  
## Install Pong environment:  
  - Run ```conda create -n pong python=3.6``` to create a virtual environment;  
  - Run ```conda activate pong``` to activate this environment;  
  - Run ```sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev``` to install the openmpi;  
  - Run ```pip install -U scikit-learn``` to install scikit learn;  
  - Run ```pip install roboschool==1.0.48``` to install the roboschool environment;  
  - Run ```pip install -r Pong/requirements.txt``` to install all the other requirements;  
  - After installing, run ```cd ~/anaconda3/envs/pong/lib/python3.6/site-packages/roboschool``` and copy the ```gym_pong.py```, ```multiplayer.py``` and ```monitor.py``` files from our provided AWS pong environment to the current folder ```roboschool```. 
