from samsung_project.LogParser import log_merger
import pandas as pd
import numpy as np


FEATURES = ['apm', 'aht', 'exp', 'gold', 'items_per_click', 'touch_swipe', 'way_compliteness']  # for bot vs. human


# Main idea, that our bot always try to reach location on which we tapped, when human-player can change
# his direction before getting to location.
# Accuracy that it is bot = complete ways(when player reach end) / all ways during the game
def way_compl(data):
    # We will calculate number of complete ways / all touches
    # Complete way: (distance / v) <= time
    def _index(data, index):
        for i, row in data.iterrows():
            if i > index and row['Event'] == ' Touch':
                return row['X-coord'], row['Y-coord'], row['Time']
    v = 0.3
    comp = 0
    touches = 0
    for index, row in data.iterrows():
        if index < len(data)-1:
            if row['Event'] == ' Touch':
                last = (row['X-coord'], row['Y-coord'], row['Time'])
                curr = _index(data, index)
                if curr is None:
                    touches += 1
                else:
                    distance = np.sqrt((last[0] - curr[0])**2 + (last[1] - curr[1])**2)
                    time = curr[2] - last[2]
                    if distance / v <= time:
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


def items_per_click(data):
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


Lg = log_merger.Log_merger
class Feature_calculator(Lg):
    """
    Use extracted data, separate it on periods and calculate results of
    features on it. Will be interpreted as structured data for future
    'learning' classifiers
    """

    def f_calc(self,data: pd.DataFrame, features: list, period = 10):
        """
        Separate data on small periods and gives vector of feature results on this period
        :param data: expecting data from Log_merger
        :param features: any number of features, you want
        :param period: set up for 10 sec period
        :return: list of feature results
        """
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
        return res


    def get_data(self,data_paths):
        """
        Extract data from data sets directory and gives matrix of feature results vectors
        :param data_paths: path to directory with your data sets
        """
        data = self.process_file(data_paths[0],data_paths[1])
        res_data = self.f_calc(data,[apm, aht, exp, gold, items_per_click, touch_swipe, way_compl])
        return res_data


    def get_data_plot(self,data_paths):
        """
        Same as get_data, but for future visualisation
        """
        res = []
        for data_path in data_paths:
            data = self.process_file(data_path[0], data_path[1])
            res += self.f_calc(data, [apm, aht, exp, gold, items_per_click, touch_swipe, way_compl])
        return res


    def get_data_for_classifier(self,data_path, player):
        '''
        Modified get_data:
        for each vector in feature results matrix it adds - 1 or 0
        0 - Player 1 (Expected bot)
        1 - Player 2 (Expected human)
        '''
        feat_data = self.get_data(data_path)
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
