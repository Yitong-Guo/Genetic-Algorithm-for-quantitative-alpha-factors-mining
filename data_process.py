import pandas as pd
import os


class DataProcess:
    def __init__(self, path: str, data_name: str):
        self.path = path
        self.data_name = data_name

        self.data = pd.read_csv(os.path.join(self.path, self.data_name + '.csv'))
        self.data_preprocess()

    def data_preprocess(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index(['date', 'token'], inplace=True)
        # q09 = self.data.quantile(0.995)
        # q01 = self.data.quantile(0.005)

        # for col in self.data.columns:
        #     self.data.loc[self.data[col] > q09[col], col] = q09[col]
        #     self.data.loc[self.data[col] < q01[col], col] = q01[col]
        self.data = self.data.sort_index(level=['token', 'date'])  # sort is very important

    def split(self, train_start, train_end, val_start, val_end, test_start, test_end):
        """Differentiate training set, validation set, and prediction set based on specific dates"""
        train_start = pd.Timestamp(train_start)
        train_end = pd.Timestamp(train_end)
        val_start = pd.Timestamp(val_start)
        val_end = pd.Timestamp(val_end)
        test_start = pd.Timestamp(test_start)
        test_end = pd.Timestamp(test_end)

        train_data = self.data.loc[(self.data.index.get_level_values(0) >= train_start) &
                                   (self.data.index.get_level_values(0) <= train_end)]
        val_data = self.data.loc[(self.data.index.get_level_values(0) >= val_start) &
                                 (self.data.index.get_level_values(0) <= val_end)]
        test_data = self.data.loc[(self.data.index.get_level_values(0) >= test_start) &
                                  (self.data.index.get_level_values(0) <= test_end)]

        train_data = train_data.sort_index(level=['token', 'date'])
        val_data = val_data.sort_index(level=['token', 'date'])
        test_data = test_data.sort_index(level=['token', 'date'])

        return train_data, val_data, test_data


if __name__ == '__main__':
    data_process = DataProcess('data/', 'shiyan')
    a, b, c = data_process.split('2021-12-01', '2022-12-01', '2022-12-10', '2023-12-01', '2023-12-10', '2024-01-01')

