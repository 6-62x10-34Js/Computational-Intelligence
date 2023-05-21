import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import warnings
import sys

# only for data handling
import pandas as pd

from HW2_NN_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, \
    plot_learned_function, plot_mse_vs_alpha

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

"""
Assignment 2: Neural networks
Part 3.1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def calculate_mse(nn, x, y):
    """
    Calculates the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """

    mse = mean_squared_error(y, nn.predict(x))
    return mse


def ex_3_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 3.1 a)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    random_state = [np.random.randint(1, 100) for _ in range(10)]
    n_hidden_neurons_list = [2, 5, 10, 20, 50]
    train_mses = np.zeros((len(n_hidden_neurons_list), len(random_state)))
    test_mses = np.zeros((len(n_hidden_neurons_list), len(random_state)))
    scores = np.zeros((len(n_hidden_neurons_list), len(random_state)))

    results = {
        'train_mses': train_mses,
        'test_mses': test_mses,
        'scores': scores
    }

    for n_neurons in n_hidden_neurons_list:
        for r in random_state:
            model = {
                'hidden_layer_sizes': (n_neurons,),
                'activation': 'logistic',
                'solver': 'lbfgs',
                'alpha': 0,
                'max_iter': 5000,
                'random_state': r
            }
            model = MLPRegressor(**model)

            model.fit(x_train, y_train)
            mse_train = calculate_mse(model, x_train, y_train)
            mse_test = calculate_mse(model, x_test, y_test)
            score = model.score(x_test, y_test)
            results['train_mses'][n_hidden_neurons_list.index(n_neurons), :] = mse_train
            results['test_mses'][n_hidden_neurons_list.index(n_neurons), :] = mse_test
            results['scores'][n_hidden_neurons_list.index(n_neurons), :] = score

    #plot_mse_vs_neurons(results['train_mses'], results['test_mses'], n_hidden_neurons_list)

    return


def ex_3_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 3.1 b)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    random_state = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    n_hidden_neurons_list = [5]
    train_mses = []
    test_mses = []
    scores = []
    for n_neurons in n_hidden_neurons_list:
        train_mses_n = []
        test_mses_n = []
        scores_n = []

        for r in random_state:
            model = MLPRegressor(hidden_layer_sizes=(n_neurons,), activation='logistic', solver='lbfgs', alpha=0,
                                 max_iter=5000, random_state=r)
            model.fit(x_train, y_train)
            mse_train = calculate_mse(model, x_train, y_train)
            mse_test = calculate_mse(model, x_test, y_test)
            score = model.score(x_test, y_test)
            train_mses_n.append(mse_train)
            test_mses_n.append(mse_test)
            scores_n.append(score)

        train_mses.append(train_mses_n)
        test_mses.append(test_mses_n)
        scores.append(scores_n)

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    scores = np.array(scores)



    print('-------------------Results-------------------')
    print(f'Max Train MSE: {np.max(train_mses)}')
    print(f'Min Train MSE: {np.min(train_mses):.4f}')
    print(f'Mean Train MSE: {np.mean(train_mses):.4f}')
    print(f'Std Train MSE: {np.std(train_mses):.4f}')
    min_train_mse_index = np.unravel_index(train_mses.argmin(), train_mses.shape)
    min_test_mse_index = np.unravel_index(test_mses.argmin(), test_mses.shape)

    if min_train_mse_index == min_test_mse_index:
        print(f"Minimum MSE is obtained for the same seed on the training and testing set. Seed: {random_state[min_train_mse_index[1]]}")
    else:
        print(f"Minimum MSE is obtained for different seeds on the training and testing set. Training seed: {random_state[min_train_mse_index[1]]}, Testing seed: {random_state[min_test_mse_index[1]]}")
    pass


def ex_3_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 3.1 c)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    random_state = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    n_hidden_neurons_list = [1, 2, 4, 6, 8, 12, 20, 40]
    train_mses = []
    test_mses = []
    scores = []
    for n_neurons in n_hidden_neurons_list:
        train_mses_n = []
        test_mses_n = []
        scores_n = []
        neurons = []

        for r in random_state:
            model = MLPRegressor(hidden_layer_sizes=(n_neurons,), activation='logistic', solver='lbfgs', alpha=0,
                                 max_iter=5000, random_state=r)
            model.fit(x_train, y_train)
            mse_train = calculate_mse(model, x_train, y_train)
            mse_test = calculate_mse(model, x_test, y_test)
            score = model.score(x_test, y_test)
            train_mses_n.append(mse_train)
            test_mses_n.append(mse_test)
            scores_n.append(score)
            neurons.append(n_neurons)

        train_mses.append(train_mses_n)
        test_mses.append(test_mses_n)
        scores.append(scores_n)

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    scores = np.array(scores)

    plot_mse_vs_neurons(train_mses, test_mses, n_hidden_neurons_list)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_train_array = np.zeros_like(y_train)
    y_pred_test_array = np.zeros_like(y_test)


    # Assign the predicted values to the corresponding indices in y_pred_train_array
    y_pred_train_array[...] = y_pred_train
    y_pred_test_array[...] = y_pred_test

    # search for n_neurons with the lowest mse
    min_test_mse_index = np.unravel_index(test_mses.argmin(), test_mses.shape)
    b_best_hidden_neurons = n_hidden_neurons_list[min_test_mse_index[0]]


    plot_learned_function(b_best_hidden_neurons, x_train, y_train, y_pred_train_array, x_test, y_test, y_pred_test_array)
    pass


def ex_3_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 3.1 d)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_neurons_list = [2, 5, 50]
    max_iter = 1

    # Set the warm_start parameter to True to retain previously learned parameters
    warm_start = True

    # Set the random seed
    random_state = 0

    real_iter = 1000

    # Create empty arrays to store the MSEs
    train_mses = np.zeros((len(n_neurons_list), real_iter))
    test_mses = np.zeros((len(n_neurons_list), real_iter))

    # Loop over the number of hidden neurons
    for i, n_hidden_neurons in enumerate(n_neurons_list):
        # Initialize the MLPRegressor with the specified parameters
        model = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons,), activation='relu', solver='adam', alpha=0.001,
                             max_iter=max_iter, warm_start=warm_start, random_state=random_state)

        # Loop over the iterations
        for j in range(real_iter):
            # Fit the model to the training data
            model.fit(x_train, y_train)

            # Calculate the MSE for the training set
            mse_train = calculate_mse(model, x_train, y_train)

            # Calculate the MSE for the testing set
            mse_test = calculate_mse(model, x_test, y_test)

            # Store the MSE values in the arrays
            train_mses[i, j] = mse_train
            test_mses[i, j] = mse_test

    # Stack the results into an array with the desired shape
    results_array = np.stack((train_mses, test_mses), axis=1)
    #List of number of iterations that produced the above MSEs

    plot_mse_vs_iterations(train_mses, test_mses, real_iter, n_neurons_list)
