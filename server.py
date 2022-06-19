import math

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from lstm_server.models import *
from lstm_server.data_prepares import *

from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

from jsonrpc import JSONRPCResponseManager, dispatcher


@dispatcher.add_method
def train_and_predict(dataset, x_data, model_type="lstm"):
    dataset = prepare_data(dataset)
    x_train, y_train, x_test, y_test = creat_dataset(dataset, test_rate=0)
    if model_type == 'lstm':
        model = get_lstm_model()
    elif model_type == 'gru':
        model = get_gru_model()
    else:
        model = get_bilstm_model()
    train(model, x_train, y_train)
    pre = predict(model, x_data)
    return pre


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
    run_simple('localhost', 9090, application)


if __name__ == '__main__':
    main()
