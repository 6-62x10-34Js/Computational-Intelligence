import json

import numpy as np
from sklearn.preprocessing import scale

from HW2_NN_regression import ex_3_1_a, ex_3_1_b, ex_3_1_c, ex_3_1_d

"""
Assignment 2: Neural networks
Part 3.1: Regression with neural networks

This file loads the data and calls the functions for each section of the assignment.
"""


def load_data():
    """
    Loads the data from data.json
    :return: A dictionary containing keys x_train, x_test, y_train, y_test
    """
    with open('data.json', 'r') as f:
        raw_data = json.load(f)

    data = {}
    # Convert arrays in the raw_data to numpy arrays
    for key, value in raw_data.items():
        data[key] = scale(np.array(value))

    # Let's reduce the size
    data['x_test'] = data['x_test'][0:10000:10]
    data['y_test'] = data['y_test'][0:10000:10]

    rg = np.random.RandomState(200)
    data['y_test'] = data['y_test'] + rg.randn(1000, 1) * .2

    data['x_train'] = data['x_train'][0:60:3]
    data['y_train'] = data['y_train'][0:60:3]

    data['y_train'] = data['y_train'] + rg.randn(20, 1) * .2

    return data


def main():
    data = load_data()
    x_train, x_test, y_train, y_test = \
        data['x_train'], data['x_test'], data['y_train'].ravel(), data['y_test'].ravel()

    ## 3.1 a)
    #ex_3_1_a(x_train, x_test, y_train, y_test)

    # 3.1 b)
    #ex_3_1_b(x_train, x_test, y_train, y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # 3.1 c)
    ex_3_1_c(x_train, x_test, y_train, y_test)

    # 3.1 d)
    ex_3_1_d(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
