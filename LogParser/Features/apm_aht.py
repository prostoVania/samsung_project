# apm - number of clicks per minute
# aht - average duration of click
import pandas as pd
import numpy as np

data = pd.read_csv('test.csv')
data = pd.DataFrame(data)


def apm(data):
    a = len(data) - 1
    #time = data.at[a,'Time'] - data.at[0,'Time']     # all time period of dataframe
    for index, row in data.iterrows():
        time_start = row[1]
        break
    for index, row in data[::-1].iterrows():
        time_end = row[1]
        break
    time = time_end - time_start
    counter = 0
    for index, row in data.iterrows():      # check for type of action
        if row[2] == 'Touch':
            counter += 1
    res = counter * 60000 / time            # clicks per minute
    return res

def aht(data):
    counter = 0     # count number of clicks
    time = 0        # sum of click time
    for index, row in data.iterrows():
        if str(row[-1]) != str(np.NaN):
            print(row[-1])
            counter += 1
            time += row[-1]
    averageDuration = time / counter    # duration of click in ms
    return averageDuration

print(apm(data))
print(aht(data))