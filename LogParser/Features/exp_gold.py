import pandas as pd

data = pd.read_csv('test.csv')
data = pd.DataFrame(data)


def gold(data):
    start = 0
    end = 0
    for index, row in data.iterrows():
        if row[2] == ' Drop':
            start = row[3]
            break
    for index, row in data[::-1].iterrows():
        if row[2] == ' Drop':
            end = row[3]
            break
    return end - start


def exp(data):
    start = 0
    end = 0
    for index, row in data.iterrows():
        if row[2] == ' Fight':
            start = row[3]
            break
    for index, row in data[::-1].iterrows():
        if row[2] == ' Fight':
            end = row[3]
            break
    return end - start


# print(gold(data))
# print(exp(data))
