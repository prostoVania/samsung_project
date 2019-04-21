import pandas as pd
import numpy as np

data = pd.read_csv('test.csv')
data = pd.DataFrame(data)



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
                    comp -= 1
                else:
                    distance = np.sqrt((last[0] - next[0])**2 + (last[1] - next[1])**2)
                    time = next[2] - last[2]
                    tmp = distance / V
                    if distance / V <= time:
                        comp += 1
                    touches += 1
    return np.abs(comp/touches)

print(way_compl(data))