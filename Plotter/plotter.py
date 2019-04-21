from Samsung_practice.tmp.LogParser import feature_calc
from Samsung_practice.tmp.Classifier.classifier import catalog_reader
import matplotlib.pyplot as plt
import numpy as np
import umap


reducer = umap.UMAP()
FEATURES = {'apm':0, 'aht':1, 'exp':2, 'gold':3, 'itemsPerClick':4, 'touch_swipe':5, 'way_compliteness':6}
DIRECTORY = '/Users/zoran/Desktop/Programming/Samsung_practice/tmp/Datasets/Data sets' # CHANGE to 'Datasets/Data sets'


def _prep(features, data):
    res_data = []
    for i in data:
        res_data.append([i[j] for j in features])
    return res_data


def plotter(feat_h, feat_b, features:list):
    if len(features) == 1:
        x_bot = np.array(range(len(feat_b)))
        y_bot = np.array([i[features[0]] for i in feat_b])      #features[0].__name__

        x_human = np.array(range(len(feat_h)))
        y_human = np.array([i[features[0]] for i in feat_h])

        plt.plot(x_bot, y_bot, 'r-', label='bot')
        plt.plot(x_human, y_human, 'b-', label='human')

        plt.legend(loc='best')

        plt.ylabel(features[0], fontsize=14)
        plt.xlim(0, max(len(feat_b),len(feat_h)))
        plt.title(features[0].upper(), fontsize=16)
        plt.show()

    else:
        data_b = _prep(features, feat_b)
        data_b = reducer.fit_transform(data_b)
        data_h = _prep(features, feat_h)
        data_h = reducer.fit_transform(data_h)
        plt.scatter(data_b[:, 0], data_b[:, 1], c='r', label = 'bot')
        plt.scatter(data_h[:, 0], data_h[:, 1], c='b', label = 'human')
        plt.gca().set_aspect('auto', 'datalim')
        plt.title('UMAP projection ', fontsize=24)
        plt.legend(loc = 'best')
        plt.show()


def main():
    paths = catalog_reader(DIRECTORY)
    bot_Path = paths[1]
    human_Path = paths[0]
    f_calc = feature_calc.Feature_calculator()
    feat_h = f_calc.get_data_plot(human_Path)
    # for f in feat_h:
    #     print(f)
    feat_b = f_calc.get_data_plot(bot_Path)
    # plotter(feat_h, feat_b, ['apm'])
    # plotter(feat_h, feat_b, ['aht'])
    # plotter(feat_h, feat_b, ['exp'])
    # plotter(feat_h, feat_b, ['gold'])
    # plotter(feat_h, feat_b, ['itemsPerClick'])
    # plotter(feat_h, feat_b, ['touch_swipe'])
    # plotter(feat_h, feat_b, ['way_compl'])
    plotter(feat_h, feat_b, ['apm', 'aht', 'gold', 'exp', 'touch_swipe'])

if __name__ == '__main__':
    main()