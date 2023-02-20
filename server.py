import math
import traceback

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from lstm_server.models import *
from lstm_server.data_prepares import *

from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

from jsonrpc import JSONRPCResponseManager, dispatcher
import time
from tensorflow import keras
import os


@dispatcher.add_method
def echo(text):
    print("echo:", text)
    return text


@dispatcher.add_method
def train_and_predict(dataset, x_data, model_type="lstm"):
    try:
        dataset = prepare_data(dataset)
        x_train, y_train, x_test, y_test = creat_dataset(dataset, test_rate=0)
        x_data_prepared = prepare_data(x_data, use_last_scaler=True)
        if len(x_data_prepared.shape) == 2:
            x_data_prepared = np.reshape(x_data_prepared, (x_data_prepared.shape[0], 1, 1))
        print("shape of me:", x_train.shape)
        print("shape of you:", x_data_prepared.shape)
        if model_type == 'lstm':
            model = get_lstm_model()
        elif model_type == 'gru':
            model = get_gru_model()
        else:
            model = get_bilstm_model()
        train(model, x_train, y_train)
        pre = predict(model, x_data_prepared)
        pre_data = restore_data(pre, numpy_type=False)
        return {
            "data": np.array(pre_data).reshape(len(pre_data)).tolist()
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e)
        }


@Request.application
def application(request):
    response = JSONRPCResponseManager.handle(
        request.get_data(cache=False, as_text=True), dispatcher)
    return Response(response.json, mimetype='application/json')


def test_local():
    # model = get_lstm_model()
    model = get_gru_model()
    # get_bilstm_model()
    x_train, y_train, x_test, y_test = load_local_data()
    print("shape of you:", x_train.shape)
    history = train(model, x_train, y_train)
    score = evaluate(model, x_test, y_test)
    print(history)
    print(score)
    model.save('lstm.h5')
    pre = predict(model, x_test)
    rmse = math.sqrt(mean_squared_error(y_test, pre))
    print('specific rmse = ', rmse)

    plt.figure(figsize=(16, 8))
    
    plt.subplot(221)
    plt.plot(y_test, 'b', label='real')
    plt.plot(pre[1:], ls='-.', c='r', label='predict')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.subplot(222)
    plt.plot(restore_data(y_test), 'b', label='real')
    plt.plot(restore_data(pre[1:]), ls='-.', c='r', label='predict')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig('kk.png')
    plt.show()


def main():
    run_simple('0.0.0.0', 9090, application)
    
def test_local2():
    # select_dim = None
    select_dim = 0
    # model = get_lstm_model()
    model = get_gru_model(out=4 if select_dim is None else 1, lr=0.001)
    dataframe = pd.read_csv('data.csv',
                            header=0, parse_dates=[0])
    print(dataframe)
    # data = load_local_dataset()
    # dataset = prepare_data(dataframe)
    dataset = dataframe
    data = np.array(dataset)
    print(data)
    print(data.shape)
    y_data = data[:, 1:-1]
    if select_dim is not None:
        y_data = y_data[:, select_dim]
    # y_data = data[:, 1:-1][:, select_dim]
    
    # # x is not time
    # x_data = data[:, :1]
    # x_data = np.vectorize(lambda x: time.mktime(x.timetuple()) * 1e-7)(x_data.reshape(len(x_data)))
    # x_data = np.array(x_data - x_data[0], dtype=np.float).reshape(len(x_data), 1, 1)
    # x_data = prepare_data(x_data).reshape(len(x_data), 1, 1)
    
    y_data = np.array(y_data, dtype=np.float)
    y_data = prepare_data(y_data)
    x_train, y_train, x_test, y_test = creat_dataset(y_data)
    x_train = x_train.reshape(len(x_train), 1, 1)
    x_test = x_test.reshape(len(x_test), 1, 1)
    # if select_dim is None:
    #     y_data = np.array([prepare_data(y_data[:, i]) for i in range(4)]).T.reshape(-1, 4)
    # else:
    #     y_data = prepare_data(y_data).reshape(len(y_data), 1)
    # data_len = len(x_data)
    # test_rate = 0.1
    # (y_train, y_test) = (y_data[:int(data_len * (1 - test_rate)) + 1, :], y_data[:int(data_len * test_rate), :]) if select_dim is None \
    #     else (y_data[:int(data_len * (1 - test_rate)) + 1], y_data[:int(data_len * test_rate)])
    # x_train, x_test = x_data[:int(data_len * (1 - test_rate)) + 1, :], x_data[:int(data_len * test_rate), :]
    print("shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    file = 'test2.h5' if select_dim is None else f'test2-{select_dim}.h5'
    file_exists = os.path.exists(file)
    if file_exists:
        print("load last model")
        model = keras.models.load_model(file)
        
    # if select_dim is None:
    #     plt.plot(x_train.reshape(len(x_train)), y_train)
    # else:
    #     plt.plot(x_train.reshape(len(x_train)), y_train.reshape(len(y_train)))
    # plt.show()
    # return
    
    # skip_train = False
    skip_train = True and file_exists
    
    if not skip_train:
        history = model.fit(x_train, y_train,
                            epochs=150,
                            batch_size=64,
                            validation_split=0.1,
                            verbose=2)
        score = evaluate(model, x_test, y_test)
        print(history)
        print(score)
        model.save(file)
    
    pre = predict(model, x_test)
    print("predict shape:", pre.shape)
    # rmse = math.sqrt(mean_squared_error(y_test, pre))
    # print('specific rmse = ', rmse)

    plt.figure(figsize=(16, 8))
    
    if select_dim is None:
        # x = x_test.reshape(len(x_test))
        dim2 = 4
        for i in range(dim2):
            plt.subplot(221 + i)
            plt.plot(y_test[:, i], 'b', label='real')
            plt.plot(pre[:, i], ls='-.', c='r', label='predict')
            plt.grid(True)
            plt.legend(loc='best')
    else:
        # x = x_test.reshape(len(x_test))
        plt.plot(y_test, 'b', label='real')
        plt.plot(pre[1:], ls='-.', c='r', label='predict')
        plt.grid(True)
        plt.legend(loc='best')
    plt.savefig('kk2.png')
    plt.show()


if __name__ == '__main__':
    main()
