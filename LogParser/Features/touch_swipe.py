import pandas as pd
import numpy as np

data = pd.read_csv('test.csv')
data = pd.DataFrame(data)

def touch_swipe(data):
    touches = 0
    swipes = 0
    for index, row in data.iterrows():
        if row[2] == ' Touch':
            if str(row['average-move-contact-minor']) != str(np.NaN):
                swipes += 1
            else:
                touches += 1
    return touches/swipes

print(touch_swipe(data))