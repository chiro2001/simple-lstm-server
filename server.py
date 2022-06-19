import math
import traceback

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from lstm_server.models import *
from lstm_server.data_prepares import *

from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

from jsonrpc import JSONRPCResponseManager, dispatcher


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
        if x_train.shape != x_data_prepared.shape:
            raise ValueError("Error Data Type")
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
            "data": pre_data
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
    model = get_lstm_model()
    # get_gru_model()
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
    plt.plot(y_test, 'b', label='real')
    plt.plot(pre, ls='-.', c='r', label='predict')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('kk.png')
    plt.show()


def main():
    run_simple('0.0.0.0', 9090, application)


if __name__ == '__main__':
    main()
