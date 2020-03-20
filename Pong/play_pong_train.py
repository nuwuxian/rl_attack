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
    parser.add_argument("--model_name", type=str, default="ppo2")
    parser.add_argument("--hyper_index", type=int, default=3)

    parser.add_argument("--vic_coef_init", type=int, default=1)  # positive
    # victim loss schedule
    parser.add_argument("--vic_coef_sch", type=str, default='const')
    # adv loss coefficient.
    parser.add_argument("--adv_coef_init", type=int, default=-1)  # negative
    # adv loss schedule
    parser.add_argument("--adv_coef_sch", type=str, default='const')
    # diff loss coefficient.
    parser.add_argument("--diff_coef_init", type=int, default=0)  # negative
    # diff loss schedule
    parser.add_argument("--diff_coef_sch", type=str, default='const')
    parser.add_argument("--n_steps", type=int, default=1000)

    # seed value
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

args = parse_args()

memo = args.memo
mode = args.mod
model_name = args.model_name
hyper_index = args.hyper_index

seed = args.seed
n_steps = args.n_steps


vic_coef_init = args.vic_coef_init
vic_coef_sch = args.vic_coef_sch
adv_coef_init = args.adv_coef_init
adv_coef_sch = args.adv_coef_sch
diff_coef_init = args.diff_coef_init
diff_coef_sch = args.diff_coef_sch

# create the gameserver, the same as enviroment
game_server_id = args.server+"{0}".format(hyper_index)
game = roboschool.gym_pong.PongSceneMultiplayer()
gameserver = multiplayer.SharedMemoryServer(game, game_server_id, want_test_window=False, profix=str(hyper_index))

# setting up the player 0
player_0_args = "--memo={0} --server={1}" \
                " --mod={2} --model_name={3} --hyper_index={4}" \
                " --vic_coef_init={5} --vic_coef_sch={6}" \
                " --adv_coef_init={7} --adv_coef_sch={8}" \
                " --diff_coef_init={9} --diff_coef_sch={10} --n_steps={11} --seed={12}".format(memo, game_server_id, mode,
                model_name, hyper_index, vic_coef_init, vic_coef_sch, adv_coef_init, adv_coef_sch,
                diff_coef_init, diff_coef_sch, n_steps, seed)

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
