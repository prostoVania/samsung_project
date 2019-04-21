import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

class Log_merger():
    def __init__(self):
        self.data = None

    def __str__(self):
        if self.data is None:
            print(self.data)
        else:
            print(self.data.head(10))


    def _touch_event_replacer(self,event_id):
        """Function to make understanding of dataset easier.
        Replace 2- Down, 1-Move, 0-Up
        :param event_id:
        :return:
        """
        if event_id == 2:
            return 'Down'
        elif event_id == 1:
            return 'Move'
        elif event_id == 0:
            return 'Up'


    def merge(self,g_df, t_df):
        """ Extract data about contact from touch log and merges it to game log
        :param g_df: game DataFrame
        :param t_df: touch DataFrame
        :return:
        """
        t_iterator = t_df.iterrows()    # iterator by rows of Touch dataFrame
        for index, element in enumerate(g_df['Event']):
            if element == ' Touch':     # if event in game is "Touch" stars data collecting about Touch
                aver_minor = []
                aver_major = []  # addition lists for "move" event data
                b_time = 0.0  # addition variables for 'move-time'
                for t_element in t_iterator:
                    # gathering all data about certain touch
                    # it begins with 'Down' event
                    if t_element[1]['touch-event'] == 'Down':
                        g_df.loc[index, 'down-contact-minor'] = t_element[1]['contact-minor']
                        g_df.loc[index, 'down-contact-major'] = t_element[1]['contact-major']
                        b_time = t_element[1]['timestamp-sec']*10**3 + t_element[1]['timestamp-ms']
                    # 'Move' event is optional
                    elif t_element[1]['touch-event'] == 'Move':
                        aver_minor.append(t_element[1]['contact-minor'])
                        aver_major.append(t_element[1]['contact-major'])
                    # 'Up' event is last in row
                    elif t_element[1]['touch-event'] == 'Up':

                        try:
                            # if 'Touch' event contains data about move it adds to dataFrame
                            g_df.loc[index, 'average-move-contact-minor'] = sum(aver_minor) / len(aver_minor)
                            g_df.loc[index, 'average-move-contact-major'] = sum(aver_major) / len(aver_major)
                        except ZeroDivisionError:  # without move information length of aver_minor is 0
                            g_df.loc[index, 'average-move-contact-minor'] = np.NaN
                            g_df.loc[index, 'average-move-contact-major'] = np.NaN
                        if b_time:
                            g_df.loc[index, 'move-time'] = t_element[1]['timestamp-sec']*10**3 +\
                                                         t_element[1]['timestamp-ms'] - b_time
                        g_df.loc[index, 'up-contact-minor'] = t_element[1]['contact-minor']
                        g_df.loc[index, 'up-contact-major'] = t_element[1]['contact-major']
                        break
        return g_df


    def process_file(self, g_link , t_link):
        """Extract information from csv log files and merges them together
        Adds information about contact to every Touch event.
        adds:
        down-contact-major/minor - info about touching
        up-contact-major/minor - info about remove
        average-move-contact-major/minor - average info about move
        move-time - time of contact
        :param game_filename: gamelog Game
        :param touch_filename: gamelog Touch
        :return: merged table
        """
        # Reading Game csv file
        g_df = pd.read_csv(g_link, header=None)
        # print(g_df.head())
        # Rename columns
        g_df = g_df.rename({0: 'Time', 1: 'Event', 2: 'X-coord', 3: 'Y-coord'}, axis='columns')
        cols = ['Time', 'X-coord', 'Y-coord']
        # Transform data to numeric
        g_df[cols] = g_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        # adding columns for merging
        g_df['down-contact-minor'] = np.NaN
        g_df['down-contact-major'] = np.NaN
        g_df['up-contact-minor'] = np.NaN
        g_df['up-contact-major'] = np.NaN
        g_df['average-move-contact-minor'] = np.NaN
        g_df['average-move-contact-major'] = np.NaN
        g_df['move-time'] = np.NaN
        # Reading Touch csv file
        t_df = pd.read_csv(t_link, header=None)
        # print(t_df.head())
        # Rename columns
        t_df = t_df.rename({0: 'timestamp-sec', 1: 'timestamp-ms', 2: 'touch-event', 3: 'X - coord',
                            4: 'Y-coord', 5: 'contact-major', 6: 'contact-minor'}, axis='columns')
        # Transform data to numeric
        cols = t_df.columns[t_df.dtypes.eq(str)]
        t_df[cols] = t_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        # Replace event_id with its name
        t_df['touch-event'] = t_df['touch-event'].map(self._touch_event_replacer)
        # Merging tables
        self.merge(g_df, t_df)
        self.data = g_df
        return g_df


# if __name__ == '__main__':
#     t = process_file('gamelog_Game_2019-01-26_18-45-04.csv',
#                        'gamelog_Touch_2019-01-26_18-45-04.csv')
#     print(t.head())
#     t.to_csv('example.csv')