import math

from sklearn.metrics import mean_squared_error

from lstm_server.models import *
from lstm_server.data_prepares import *


def main():
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

    # plt.figure(figsize=(16, 8))
    # plt.plot(y_test, 'b', label='real')
    # plt.plot(pre, ls='-.', c='r', label='predict')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.savefig('kk.png')
    # plt.show()


if __name__ == '__main__':
    main()
