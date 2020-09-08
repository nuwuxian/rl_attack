import pandas as pd
import json

def process_csv(monitor_path):
    print(monitor_path)
    data = pd.read_csv("{}".format(monitor_path), skiprows=[0], header=0)
    # data = data.get_chunk()
    data['score_board'] = data['score_board'].replace({'\'': '"'}, regex=True)


    data_score = pd.io.json.json_normalize(data['score_board'].apply(json.loads))

    data_score['total_round'] = data_score[['left.oppo_double_hit', 'left.oppo_miss_catch',
                                             'left.oppo_slow_ball',
                                            #'left.oppo_miss_start', 'right.oppo_miss_start',
                                            'right.oppo_double_hit', 'right.oppo_miss_catch',
                                             'right.oppo_slow_ball']].abs().sum(axis=1)

    data_score_next = data_score.shift(periods=1)
    data_score_epoch = data_score - data_score_next

    # data_score_epoch = data_score_epoch[data_score_epoch['total_round']!=0]

    data_score_epoch['left_winning'] = data_score_epoch[['left.oppo_double_hit', 'left.oppo_miss_catch', #'left.oppo_miss_start',
                                                          'left.oppo_slow_ball']].abs().sum(axis=1)
    data_score_epoch['tie_winning'] = data_score_epoch['left.not_finish'].abs()

    data_score_epoch['left_winning'] = data_score_epoch['left_winning'] + data_score_epoch['tie_winning']
    data_score_epoch['total_round']  += data_score_epoch['left.not_finish'].abs()
    wining_rate_sum = data_score_epoch['left_winning'].rolling(1000,min_periods=50).sum()
    total_round_sum = data_score_epoch['total_round'].rolling(1000,min_periods=50).sum()

    wining_rate = wining_rate_sum / total_round_sum
    result = pd.concat([wining_rate], names=['winning_rate'], axis=1)
    return result



# plot_monitor(monitor_path)
import os
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

results = []
lines = []
lines_names = []

colors = ["red",'blue',"g", 'y']
#colors = ["red","g"]

base_folder = "./Log"
file_folders = []

for file_folder in os.listdir(base_folder):
    if file_folder.endswith("_"):
        continue
    file_folders.append(file_folder)

file_folders.sort()

# file_folders = ["att_weight_1", "att_weight_10"]

for file_folder in file_folders:
#    if file_folder.endswith("videos") or file_folder.endswith("2034") or file_folder.endswith("003045"):
#        continue
    if file_folder.find('video') != -1:
        continue
    monitor_path = os.path.join(base_folder, file_folder)
    one_method_result = []
    if os.path.isdir(monitor_path):
        for csv_file in os.listdir(monitor_path):
            if csv_file.endswith(".csv"):
                monitor_path_csv_file = os.path.join(monitor_path,csv_file)
                try:
                    one_result = process_csv(monitor_path_csv_file)
                    one_method_result.append(one_result)
                except KeyError:
                    print("Key Error in {}, please check the header of the file".format(monitor_path))
        one_method_result = pd.concat(one_method_result, axis=1)
        one_method_result = one_method_result.iloc[:4000]
        one_method_result["mean"] = one_method_result.mean(axis=1)
        one_method_result["std"] = one_method_result.std(axis=1)

        tmp = one_method_result['mean'] + one_method_result['std']

        print('---------------max value is ', max(tmp.dropna(axis=0, how='all')))
        # if file_folder == "adv_train":
        #     one_method_result = one_method_result.iloc[:1500]

        line, = ax.plot(one_method_result.index, one_method_result["mean"], color=colors[len(lines)], linewidth=3)

        ax.fill_between(one_method_result.index, one_method_result["mean"] - one_method_result["std"],
                        one_method_result["mean"] + one_method_result["std"],
                        facecolor=colors[len(lines)], interpolate=True, alpha=0.4)

        lines.append(line)
        lines_names.append(file_folder[:-7] if file_folder.endswith("step-0") else file_folder)

        results = one_method_result

ax.set_xticks([0, 1000, 2000, 3000, 4000])
ax.set_yticks([0, 0.3, 0.6, 1.0])
plt.grid(True)

ax.set_ylim(0.25,1)
ax.set_xlim(0,4000)

fig = ax.get_figure()
fig.savefig("{}/monitor.pdf".format(base_folder))
plt.close()
