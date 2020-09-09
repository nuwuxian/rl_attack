import os
import fnmatch
import tensorflow as tf
import logging
import functools
import traceback
import multiprocessing
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger('modelfree.visualize.tb')


# find the file
def find_tfevents(log_dir):
    result = []
    for root, dirs, files in os.walk(log_dir, followlinks=True):
        if root.endswith('rl/tb'):
            for name in files:
                # print(root)
                if fnmatch.fnmatch(name, 'events.out.tfevents.*'):
                    result.append(os.path.join(root, name))
    return result


# read the events
def read_events_file(events_filename, keys=None):
    events = []
    try:
        for event in tf.train.summary_iterator(events_filename):
            row = {'wall_time': event.wall_time, 'step': event.step}
            for value in event.summary.value:
                if keys is not None and value.tag not in keys:
                    continue
                row[value.tag] = value.simple_value
            events.append(row)
    except Exception:
        logger.error(f"While reading '{events_filename}': {traceback.print_exc()}")
    return events


def data_frame(events, game, subsample=100000):
    dfs = []
    for event in events:
        df = pd.DataFrame(event)
        df = df.set_index('step')
        df = df[~df.index.duplicated(keep='first')]
        df = df.dropna(how='any')

        s = (df.index / subsample).astype(int)
        df = df.groupby(s).mean()
        df.index = df.index * subsample
        dfs.append(df)
    data_form = pd.concat(dfs)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    return data_form


# read the tb data
# set the data form
# suppose we have the
def load_tb_data(log_dir, keys=None):
    event_paths = find_tfevents(log_dir)
    pool = multiprocessing.Pool()
    events_by_path = pool.map(functools.partial(read_events_file, keys=keys), event_paths)
    return events_by_path


# plot the graph
def plot_data(log_dir, out_dir, filename, game, length=350, reverse=False):

    fig, ax = plt.subplots(figsize=(10, 8))
    #colors = ['r', 'g', 'b']
    # ['orangered', 'darkgreen', '#0165FC']
    #colors = ['orangered', '#0165FC', 'darkgreen']
    colors = ['r', 'b', 'g', 'y']
    methods = ['our', 'ratio-2', 'ratio-3', 'ratio-4']
    std = []
    for i in range(len(methods)):
        method = methods[i]
        if game == "YouShallNotPass":
            if reverse:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win0'])
                subset = data_frame(events, game=game)
                #subset['game_win1'] = 1.0 - subset['game_win1']
                group = subset.groupby('step')['game_win0']

            else:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_norm_win1'])
                subset = data_frame(events, game=game)
                group = subset.groupby('index')['game_norm_win1']
        else:
            if reverse:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win1'])
                subset = data_frame(events, game=game)
                subset['game_win0'] = 1.0 - subset['game_win1']
                group = subset.groupby('index')['game_win0']

                events_adv = load_tb_data(os.path.join(log_dir, method), keys=['game_adv_win1'])
                subset_adv = data_frame(events_adv, game=game)
                subset_adv['game_adv_win0'] = 1.0 - subset_adv['game_adv_win1']
                group_adv = subset_adv.groupby('index')['game_adv_win0']

                events_norm = load_tb_data(os.path.join(log_dir, method), keys=['game_norm_win1'])
                subset_norm = data_frame(events_norm, game=game)
                subset_norm['game_norm_win0'] = 1.0 - subset_norm['game_norm_win1']
                group_norm = subset_norm.groupby('index')['game_norm_win0']

            else:
                events_adv = load_tb_data(os.path.join(log_dir, method), keys=['game_adv_win0'])
                subset_adv = data_frame(events_adv, game=game)
                group_adv = subset_adv.groupby('index')['game_adv_win0']

                events_norm = load_tb_data(os.path.join(log_dir, method), keys=['game_norm_win0'])
                subset_norm = data_frame(events_norm, game=game)
                group_norm = subset_norm.groupby('index')['game_norm_win0']

                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win0'])
                subset = data_frame(events, game=game)
                group = subset.groupby('index')['game_win0']



        min_n, mean, max_n = group.min()[0:length+1], group.mean()[0:length+1], group.max()[0:length+1]

        '''
        min_adv_n, mean_adv, max_adv_n = group_adv.min()[0:length + 1], group_adv.mean()[0:length + 1], group_adv.max()[0:length + 1]
        min_norm_n, mean_norm, max_norm_n = group_norm.min()[0:length + 1], group_norm.mean()[0:length + 1], group_norm.max()[0:length + 1]
        '''


        #mean.index /= 2

        print('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        std.append(group.std()[0:length+1])
        ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[i])
        mean.plot(ax=ax, color=colors[i], linewidth=3)

    ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7])
    #ax.set_xticks([0, 1.5e+7, 2.5e+7, 3.5e+7])
    #ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7, 2.5e+7, 3e+7, 3.5e+7])
    # ax.set_yticks([0, 0.05, 0.1, 0.2, 1])
    #ax.set_yticks([0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(1):
        std[i].plot(ax=ax, color=colors[i], linewidth=3)
    #ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7, 2.5e+7, 3e+7, 3.5e+7])
    ax.set_xticks([0, 0.5e+7, 1e+7])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seed", type=int, default=6)
    parser.add_argument('--log_dir', type=str, default='/home/xkw5132/Desktop/usenix_paper/Mujoco_results/usenix2020_data/vs_lamda')
    parser.add_argument("--out_dir", type=str, default='/home/xkw5132/Desktop/usenix_paper/Mujoco_results/usenix2020_data/vs_lamda')
    parser.add_argument("--filename", type=str, default='out.png')
    args = parser.parse_args()
    reverse = True

    game = 'YouShallNotPass'
    #game = 'KickAndDefend'
    #game = 'SumoHumans'
    #game = 'SumoAnts'

    out_dir = args.out_dir
    log_dir = args.log_dir
    if reverse:
        filename = game + '_reverse' + '.png'
    else:
        filename = game+'.png'
    # log_dir = log_dir+'/'+game

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, length=200, reverse=reverse, game=game)
