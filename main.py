import os
import pandas as pd

from genetic_process import GPProcess
from data_process import DataProcess
import pickle

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    train_start = '2021-11-06 00:00:00'  # 请精确到时分秒
    train_end = '2024-02-28 00:00:00'
    val_start = '2021-11-06 00:00:00'
    val_end = '2024-02-28 00:00:00'

    test_start = '2021-11-06 00:00:00'
    test_end = '2024-08-26 00:00:00'  # 日期都包含在集内

    model_path = "config/model"
    factor_path = "config/factor"
    data_path = 'data/'
    data_name = 'caopre_all_t'  # 输入数据格式，列名为date, token, returns，其余列名随便，
    mode = 'train'  # 'predict' or 'train'

    data_pro = DataProcess(path=data_path, data_name=data_name)
    train_data, val_data, test_data = data_pro.split(train_start,
                                                     train_end,
                                                     val_start,
                                                     val_end,
                                                     test_start,
                                                     test_end)

    # cals = ['ts_asc_sort_cut', 'ts_dec_sort_cut', 'ts_dec_zscore_cut', 'ts_asc_zscore_cut',
    #         'ts_group_dec_sort_cut', 'ts_group_asc_sort_cut', 'ts_group_dec_zscore_cut', 'ts_group_asc_zscore_cut',
    #         'ts_roll_rank', 'ts_avgdev', 'ts_regalpha', 'ts_regbeta',
    #         'ts_argmin', 'ts_mean','ts_kurt', 'ts_max', 'ts_correlation']
    cals = ['ts_dec_sort_cut', 'ts_kurt', 'ts_correlation']
    # 目前评价函数（损失函数）有'RankIC', 'RankICIR', 'Hedge_Return_Std', 'Hedge_Return', 'NDCG'
    train_evaluations = ['Hedge_Return_Std', 'Hedge_Return']
    valid_evaluations = ['Hedge_Return_Std']

    a = GPProcess(train_data=train_data,
                  val_data=val_data,
                  test_data=test_data,
                  train_evaluation=train_evaluations,
                  valid_evaluation=valid_evaluations,
                  population_num=200,
                  arity=0,
                  batch_size=10,
                  generation=6,
                  initial_depth=3,
                  max_depth=4,
                  random_seed=1,
                  cals='all')

    if mode == 'train':
        results = a.run()

        for ind in results:
            try:
                factor = a.generate_factor(ind)

                factor.to_csv(os.path.join(factor_path, 'factors{}'.format(results.index(ind)+1) + '.csv'))

                with open(os.path.join(model_path, 'factors{}'.format(results.index(ind)+1) + '.pkl'), 'wb') as file:
                    pickle.dump(ind, file)
            except:
                print('出现问题', str(ind))
                continue

    elif mode == 'predict':
        all_files = [file for file in os.listdir(model_path) if file.endswith('pkl')]

        for file_path in all_files:
            with open(os.path.join(model_path, file_path), 'rb') as file:
                formula_tree = pickle.load(file)  # 此段代码用于加载模型，可直接当ind使用

            print('load success: {}'.format(file_path[:-4]))
            factor = a.generate_factor(formula_tree)
            factor.to_csv(os.path.join(factor_path, file_path[:-4] + '.csv'))

    else:
        raise ValueError("mode must be 'train' or 'predict'")

