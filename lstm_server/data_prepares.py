import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def creat_dataset(dataset, tw=1, test_rate=0.1):
    data_x = []
    data_y = []
    length = len(dataset) - tw
    for i in range(length):
        data_x.append(dataset[i:i + tw])
        data_y.append(dataset[i + tw])
    return np.asarray(data_x[:int(length * (1 - test_rate))]), \
           np.asarray(data_y[:int(length * (1 - test_rate))]), \
           np.asarray(data_x[int(length * (1 - test_rate)):]), \
           np.asarray(data_y[int(length * (1 - test_rate)):])


def load_local_data():
    dataframe = pd.read_csv('data.csv',
                            header=0, parse_dates=[0],
                            index_col=0, usecols=[0, 5], squeeze=True)
    dataset = dataframe.values
    # print('dataset:', dataset)
    dataset = prepare_data(dataset)
    # x_train, y_train, x_test, y_test = creat_dataset(dataset)
    return creat_dataset(dataset)


def prepare_data(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_new = scaler.fit_transform(dataset.reshape(-1, 1))
    return dataset_new
