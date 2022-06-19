import os

import requests

from lstm_server.data_prepares import *
from server import test_local


def main():
    os.environ['http_proxy'] = ""
    os.environ['https_proxy'] = ""
    url = "http://localhost:9090/jsonrpc"
    headers = {'content-type': 'application/json'}
    dataset = load_local_dataset(numpy_type=False)
    x_train, y_train, x_test, y_test = load_local_data(numpy_type=False)

    payload = {
        "method": "train_and_predict",
        "params": [dataset, x_test],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(url, json=payload).json()
    print(response)
    result = response["result"]
    print(result)


if __name__ == "__main__":
    # main()
    test_local()
