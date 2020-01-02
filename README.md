# rl_attack
Currently, we only implement the white-box attack without attention based on mlp-policy. We will support attention and lstm-policy in 
the following days. 

## Dependencies

Create a virtual environment by running `creatvenv.sh` script. Then install all dependencies by running `pip install -r requirements.txt`.


## Adversarial Train

Start adversarial training by running 
```
python adv_train.py {env_id}

or sh run.sh (run 3 different seeds)
```
