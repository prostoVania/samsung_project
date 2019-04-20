import pandas as pd

data = pd.read_csv('test.csv')
data = pd.DataFrame(data)

def itemsPerClick(data):
    counter = 0         # to calculate number of touches
    for index, row in data.iterrows():
        if row[2] == 'Touch':
            counter += 1
    # We can have pattern when our bot all period is only collecting items without any touch
    # (can be if period time is too small), to prevent /0 we understand that whole data set will contain
    # only drop and fight, without any touches, so number of items per click will be len of whole data set
    if counter == 0:
        return len(data)
    res = (len(data) - counter) / counter
    return res

print(itemsPerClick(data))