import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def creat_dataset(dataset, tw=1, test_rate=0.1, numpy_type=True):
    data_x = []
    data_y = []
    length = len(dataset) - tw
    for i in range(length):
        data_x.append(dataset[i:i + tw])
        data_y.append(dataset[i + tw])
    if numpy_type:
        return np.asarray(data_x[:int(length * (1 - test_rate)) + 1]), \
               np.asarray(data_y[:int(length * (1 - test_rate)) + 1]), \
               np.asarray(data_x[int(length * (1 - test_rate)):]), \
               np.asarray(data_y[int(length * (1 - test_rate)):])
    else:
        return data_x[:int(length * (1 - test_rate)) + 1], \
               data_y[:int(length * (1 - test_rate)) + 1], \
               data_x[int(length * (1 - test_rate)):], \
               data_y[int(length * (1 - test_rate)):]


def load_local_dataset(numpy_type=True, data_index=5):
    dataframe = pd.read_csv('data.csv',
                            header=0, parse_dates=[0],
                            index_col=0, 
                            usecols=[0, data_index], 
                            squeeze=True)
    print(dataframe)
    dataset = dataframe.values
    if not numpy_type:
        dataset = dataset.tolist()
    return dataset


def load_local_data(**kwargs):
    dataset = load_local_dataset(**kwargs)
    # print('dataset:', dataset)
    dataset = prepare_data(dataset, **kwargs)
    # x_train, y_train, x_test, y_test = creat_dataset(dataset)
    return creat_dataset(dataset, **kwargs)


g_scaler: MinMaxScaler = None


def prepare_data(dataset, numpy_type=True, use_last_scaler=False):
    global g_scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = np.asarray(dataset)
    length = dataset.shape[0]
    print("prepare_data: length =", length)
    dataset_new = (g_scaler if use_last_scaler else scaler).fit_transform(dataset.reshape(-1, 1))
    if not numpy_type:
        dataset_new = dataset_new.tolist()
    if not use_last_scaler:
        g_scaler = scaler
    return dataset_new


def restore_data(pre, numpy_type=True):
    assert g_scaler is not None
    pre_data = g_scaler.inverse_transform(pre)
    if not numpy_type:
        pre_data = pre_data.tolist()
    return pre_data
