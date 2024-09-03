import torch


class TerminalProcess(object):
    def __init__(self, train_data, val_data, test_data):
        """Change the factors from dataframe as [date, token, factor] to tensor"""
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.train_attr = []
        train_names = train_data.columns
        for name in train_names:
            tensor_df_train = train_data[name].unstack().sort_index()
            setattr(self, name + '_train', torch.from_numpy(tensor_df_train.values))
            self.train_attr.append(name)

        self.val_attr = []
        val_names = val_data.columns
        for name in val_names:
            tensor_df_val = val_data[name].unstack().sort_index()
            setattr(self, name + '_val', torch.from_numpy(tensor_df_val.values))
            self.val_attr.append(name)

        self.test_attr = []
        test_names = test_data.columns
        for name in test_names:
            tensor_df_test = test_data[name].unstack().sort_index()
            setattr(self, name + '_test', torch.from_numpy(tensor_df_test.values))
            self.test_attr.append(name)

        self.config_funcs = []
        self.init_config()

    def init_config(self):
        arr = dir(self)
        for func_name in arr:
            if func_name.startswith("alpha"):
                self.config_funcs.append(func_name)

    def call_func(self, func_name):
        return getattr(self, func_name)()

    def back_to_dataframe(self, df):
        """The row and column names required when terminal changes from tensor to dataframe"""
        tensor_df = df['returns'].unstack().sort_index()
        column = tensor_df.columns
        indexes = tensor_df.index

        return column, indexes

