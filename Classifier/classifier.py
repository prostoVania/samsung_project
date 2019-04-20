from Samsung_practice.log_parser.feature_calc import get_data_for_classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn import naive_bayes
import os,warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle

METHODS = [('DecisionTree',DecisionTreeClassifier), ('LogReg',LogisticRegression), ('kNN',KNeighborsClassifier),
           ('RandomForest',RandomForestClassifier), ('NaiveBayes', naive_bayes.GaussianNB)]

DIRECTORY = '/Users/zoran/desktop/Data sets' # CHANGE


def test(name):
    return name != '.DS_Store'


def catalog_reader(directory):
    #os.chdir("{}".format(catalog_path))
    categories = os.listdir(directory)
    human_path = []
    bot_path = []
    for category in categories:
        if test(category):
            sessions = os.listdir(directory + '/' + category)
            for session in sessions:
                if test(session):
                    if category == 'Human':
                        human_path.append(sorted([directory + '/' + category + '/' + session + '/' + i \
                                           for i in os.listdir(directory + '/' + category + '/' + session) if test(i)]))
                    elif category == 'Bot':
                        bot_path.append(sorted([directory + '/' + category + '/' + session + '/' + i \
                                           for i in os.listdir(directory + '/' + category + '/' + session) if test(i)]))
    # for i in human_path:
    #     print(i)
    # for i in bot_path:
    #     print(i)
    return human_path, bot_path

# paths = catalog_reader(DIRECTORY)
# x_training, x_test, y_training, y_test = data_class(paths[0],paths[1])
# with open ('tmp.pickle', 'rb') as file:
#     # pickle.dump([x_training, x_test, y_training, y_test], file)
#     data = pickle.load(file)
#     x_training, x_test, y_training, y_test = data[0], data[1], data[2], data[3]


def roc_helper(y_predicted):
    return np.array([y_predicted[i] == 1 for i in range(len(y_predicted))])

roc_stats = {}

def classifier(human_paths: list, bot_paths: list, boost):
    '''

    :param human_paths:
    :param bot_paths:
    :param boost: None, AdaBoost, GradientBoost,
    :return:
    '''
    x_human = []
    y_human = []
    x_bot = []
    y_bot = []
    for path in human_paths:
        # print(path)
        tmp = get_data_for_classifier(path, 1)
        x_human += tmp[0]
        y_human += tmp[1]
        print(len(x_human))
    for path in bot_paths:
        tmp = get_data_for_classifier(path, 0)
        x_bot += tmp[0]
        y_bot += tmp[1]
        print(len(x_bot))
    X = x_human + x_bot
    Y = y_human + y_bot
    #print('Working with data - done')
    x_training, x_test, y_training, y_test = train_test_split(X, Y, test_size=0.22)
    #print('Splitting on sets - done')
    if not boost:
        for name, meth in METHODS:
            clf = meth()
            clf.fit(x_training, y_training)
            y_predicted = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_predicted)
            x, y, _ = roc_curve(y_test, roc_helper(y_predicted), pos_label=1)
            roc_stats[name] = (x, y)
            print(clf.score(x_test,y_test))
            print('{} confusion matrix:\n'.format(name), cm)
            print(classification_report(y_test, y_predicted))


    elif boost == 'AdaBoost':
        print('AdaBoost: ')
        methods = [('DecisionTree', DecisionTreeClassifier), ('LogReg', LogisticRegression),
                   ('RandomForest', RandomForestClassifier)]
        for name, meth in methods:
            for alg in ['SAMME', 'SAMME.R']:
                clf = AdaBoostClassifier(base_estimator=meth(), n_estimators=50, learning_rate=1,
                                         algorithm=alg)
                clf.fit(x_training, y_training)
                y_predicted = clf.predict(x_test)
                cm = confusion_matrix(y_test, y_predicted)
                x, y, _ = roc_curve(y_test, roc_helper(y_predicted), pos_label=1)
                roc_stats['AdaBoost' + name] = (x, y)
                print(clf.score(x_test, y_test))
                print('{}(alg - {}) confusion matrix:\n'.format(name, alg), cm)
                print(classification_report(y_test, y_predicted))


    elif boost == 'GradientBoost':
        print('GradientBoost')
        clf = GradientBoostingClassifier(loss='exponential')
        clf.fit(x_training, y_training)
        y_predicted = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_predicted)
        x, y, _ = roc_curve(y_test, roc_helper(y_predicted), pos_label=1)
        roc_stats['GradientBoost'] = (x, y)
        print(clf.score(x_test, y_test))
        print('{} confusion matrix:\n'.format(boost), cm)
        print(classification_report(y_test, y_predicted))


    elif boost == 'BaggingClassifier':
        for name, meth in METHODS:
            clf = BaggingClassifier(meth())
            clf.fit(x_training, y_training)
            y_predicted = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_predicted)
            x, y, _ = roc_curve(y_test, roc_helper(y_predicted), pos_label=1)
            roc_stats['Bagging' + name] = (x, y)
            print(clf.score(x_test, y_test))
            print('{} confusion matrix:\n'.format(name), cm)
            print(classification_report(y_test, y_predicted))


def rocCurve(roc_stats, zoom_xlim=(0, 1), zoom_ylim=(0, 1.1)):
    for name, roc_x_y in roc_stats.items():
        plt.plot(*roc_x_y, label=name)

    plt.xlim(*zoom_xlim)
    plt.ylim(*zoom_ylim)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc_zoomed.png')
    plt.show()


if __name__ == '__main__':
    paths = catalog_reader(DIRECTORY)
    classifier(paths[0],paths[1], boost=False)
    classifier(paths[0],paths[1], boost='AdaBoost')
    classifier(paths[0],paths[1], boost='GradientBoost')
    classifier(paths[0],paths[1], boost='BaggingClassifier')
    rocCurve(roc_stats)