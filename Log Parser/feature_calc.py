from Samsung_practice.log_parser import log_merger
import pandas as pd
import numpy as np

FEATURES = ['apm', 'aht', 'exp', 'gold', 'items_per_click', 'touch_swipe', 'way_compliteness'] # for bot vs. human
FEATURES_HH = []


def _index(data, index):
    for i, row in data.iterrows():
        if i > index and row['Event'] == ' Touch':
            return (row['X-coord'], row['Y-coord'], row['Time'])

# Main idea, that our bot always try to reach location on which we tapped, when human-player can change
# his direction before getting to location.
# Accuracy that it is bot = complete ways(when player reach end) / all ways during the game
def way_compl(data):
    # We will calculate number of complete ways / all touches
    # Complete way: (distance / v) <= time
    V = 0.3
    comp = 0
    touches = 0
    for index, row in data.iterrows():
        if index < len(data)-1:
            if row['Event'] == ' Touch':
                last = (row['X-coord'], row['Y-coord'], row['Time'])
                next = _index(data, index)
                if next == None:
                    touches += 1
                else:
                    distance = np.sqrt((last[0] - next[0])**2 + (last[1] - next[1])**2)
                    time = next[2] - last[2]
                    tmp = distance / V
                    if distance / V <= time:
                        comp += 1
                    touches += 1
    return np.abs(comp - touches)


def gold(data):
    start = 0
    end = 0
    for index, row in data.iterrows():
        if row['Event'] == ' Drop':
            start = row['X-coord']
            break
    for index, row in data[::-1].iterrows():
        if row['Event'] == ' Drop':
            end = row['X-coord']
            break
    return end - start


def exp(data):
    start = 0
    end = 0
    for index, row in data.iterrows():
        if row['Event'] == ' Fight':
            start = row['X-coord']
            break
    for index, row in data[::-1].iterrows():
        if row['Event'] == ' Fight':
            end = row['X-coord']
            break
    return end - start


def touch_swipe(data):
    touches = 0
    swipes = 0
    for index, row in data.iterrows():
        if row['Event'] == ' Touch':
            if str(row['average-move-contact-minor']) != str(np.NaN):
                swipes += 1
            else:
                touches += 1
    return touches - swipes


def itemsPerClick(data):
    counter = 0         # to calculate number of touches
    for index, row in data.iterrows():
        if row['Event'] == 'Touch':
            counter += 1
    # We can have pattern when our bot all period is only collecting items without any touch
    # (can be if period time is too small), to prevent /0 we understand that whole data set will contain
    # only drop and fight, without any touches, so number of items per click will be len of whole data set
    if counter == 0:
        return len(data)
    res = (len(data) - counter) / counter
    return res


def apm(data):
    time = int(list(data.Time)[-1])
    res = (data.shape[0] - 1) * 60000 / time
    return res


def aht(data):
    counter = 0     # count number of clicks
    time = 0        # sum of click time
    for index, row in data.iterrows():
        if str(row[-1]) != str(np.NaN):
            #print(row[-1])
            counter += 1
            time += row[-1]
    if time == 0:
        return 0
    averageDuration = time / counter    # duration of click in ms
    return averageDuration


def f_calc(data: pd.DataFrame, features: list, period = 10):
    # print(data.head())
    temp = period * 1000
    res = []
    end_time = temp
    start_time = 0
    while end_time <= data.at[len(data)-2,'Time']:
        data_part = data.loc[(start_time <= data['Time']) & (end_time >= data['Time'])]
        # print(data_part.head())
        start_time += temp
        end_time += temp
        if len(data_part) == 0:
            continue
        tmp = {}
        for feature in features:
            tmp[feature.__name__] = feature(data_part)
        res.append(tmp)
        #print(len(res))
    return res


def get_data(data_paths):
    # res_data = []
    # for data_path in data_paths:
    #     print(data_path)
    data = log_merger.process_file(data_paths[0],data_paths[1])
    res_data = f_calc(data,[apm, aht, exp, gold, itemsPerClick, touch_swipe, way_compl])
    return res_data


def get_data_plot(data_paths):
    res = []
    for data_path in data_paths:
        data = log_merger.process_file(data_path[0], data_path[1])
        res += f_calc(data, [apm, aht, exp, gold, itemsPerClick, touch_swipe, way_compl])
    return res

def get_data_for_classifier(data_path, player):
    '''

    :param data_path: (log_game,log_touch)
    :param player: 0 for bot, 1 for player
    :return:
    '''
    feat_data = get_data(data_path)
    final_data = []
    final_data_player = []
    for i in feat_data:
        final_data.append(list(i.values()))
        final_data_player.append(player)
    return final_data, final_data_player




# if __name__ == '__main__':
#     bot_Path = ['gamelog_Game_2019-01-31_10-59-48.csv', 'gamelog_Touch_2019-01-31_10-59-48.csv']
#     human_Path = ['gamelog_Game_2019-01-31_11-47-31.csv', 'gamelog_Touch_2019-01-31_11-47-31.csv']
#
#     feat_h = get_data(bot_Path)
#     feat_b = get_data(human_Path)
#
#     clas_data = get_data_for_classifier(human_Path,1)
