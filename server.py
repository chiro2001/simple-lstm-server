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


if __name__ == '__main__':
    main()
