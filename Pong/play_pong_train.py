import os, sys, subprocess
import argparse
import roboschool, multiplayer


# We assume the Game Server running forever
# Set a Large number: 10000256
INF = 10000256

def parse_args():

    parser = argparse.ArgumentParser()

    # memo, hyper_index and server for create the serve
    parser.add_argument("--memo", type=str, default='ppo_pong')
    parser.add_argument("--server", type=str, default='pongdemo_adv')
    parser.add_argument("--mod", type=str, default="advtrain")

    # model_name (previous distinguish ppo2 and ppo1, now is ppo)
    parser.add_argument("--model_name", type=str, default="ppo1")
    parser.add_argument("--hyper_index", type=int, default=3)

    # seed value
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

args = parse_args()

memo = args.memo
mode = args.mod
model_name = args.model_name
hyper_index = args.hyper_index

seed = args.seed


# create the gameserver, the same as enviroment
game_server_id = args.server+"{0}".format(hyper_index)
game = roboschool.gym_pong.PongSceneMultiplayer()
gameserver = multiplayer.SharedMemoryServer(game, game_server_id, want_test_window=False, profix=str(hyper_index))

# setting up the player 0
player_0_args = "--memo={0} --server={1} " \
                "--mod={2} --model_name={3} --hyper_index={4} " \
                "--seed={5}".format(memo, game_server_id, mode,
                model_name, hyper_index, seed)

player_0_args = player_0_args.split(" ")

sys_cmd = [sys.executable, 'play_pong_player0.py']
sys_cmd.extend(player_0_args)
p0 = subprocess.Popen(sys_cmd)


# setting up the player 1
subprocess.Popen([sys.executable, 'play_pong_player1.py', game_server_id, "test"])

try:
    gameserver.serve_forever(INF)
except ValueError:
    print("End of training!")
