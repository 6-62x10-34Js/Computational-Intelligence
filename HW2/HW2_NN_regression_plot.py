import matplotlib.pyplot as plt
import numpy as np

"""
Assignment 2: Neural networks
Part 3.1: Regression with neural networks

This file contains functions for plotting.

"""


def plot_mse_vs_neurons(train_mses, test_mses, n_hidden_neurons_list):
    """
    Plot the mean squared error as a function of the number of hidden neurons.
    :param train_mses: Array of training MSE of shape n_hidden x n_seeds
    :param test_mses: Array of testing MSE of shape n_hidden x n_seeds
    :param n_hidden_neurons_list: List containing number of hidden neurons
    :return:
    """

    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training MSE with number of neurons in the hidden layer")

    for data, name, color in zip([train_mses, test_mses], ["Training MSE", "Testing MSE"], ['orange', 'blue']):
        m = data.mean(axis=1)
        s = data.std(axis=1)

        plt.fill_between(n_hidden_neurons_list, m - s, m + s, color=color, alpha=.2)

        plt.plot(n_hidden_neurons_list, m, 'o', linestyle='-', label=name, color=color)


    plt.ylim(0, 4)
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("MSE")
    plt.savefig("plots/mse_vs_neurons.png")
    plt.semilogx()
    plt.legend()
    plt.show()



def plot_mse_vs_iterations(train_mses, test_mses, n_iterations, hidden_neuron_list):
    """
    Plot the mean squared errors as a function of n_iterations
    :param train_mses: Array of training MSE of shape (len(hidden_neuron_list),n_iterations)
    :param test_mses: Array of testing MSE of shape (len(hidden_neuron_list),n_iterations)
    :param n_iterations: List of number of iterations that produced the above MSEs
    :param hidden_neuron_list: The number of hidden neurons used for the above experiment (Used only for the title of the plot)
    :return:
    """
    plt.figure()
    plt.title("Variation of MSE across iterations".format(hidden_neuron_list))

    color = ['blue', 'orange', 'red', 'green', 'purple']

    for k_hid, n_hid in enumerate(hidden_neuron_list):
        for data, name, ls in zip([train_mses[k_hid], test_mses[k_hid,]], ['Train', 'Test'], ['dashed', 'solid']):
            plt.plot(range(n_iterations), data, label=name + ' n_h = {}'.format(n_hid), linestyle=ls,
                     color=color[k_hid])

    plt.xlim([0, n_iterations])
    plt.ylim([0, 2])

    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE")
    plt.minorticks_on()
    plt.savefig("plots/mse_vs_iterations.png")
    plt.show()


def plot_mse_vs_alpha(train_mses, test_mses, alphas):
    """
    Plot the mean squared errors as afunction of the alphas
    :param train_mses: Array of training MSE, of shape (n_alphas x n_seed)
    :param test_mses: Array of testing MSE, of shape (n_alphas x n_seed)
    :param alphas: List of alpha values used
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training MSE with regularization parameter")

    for data, name, color in zip([train_mses, test_mses], ["Training MSE", "Testing MSE"], ['orange', 'blue']):
        m = data.mean(axis=1)
        s = data.std(axis=1)

        plt.plot(alphas, m, 'o', linestyle='-', label=name, color=color)
        plt.fill_between(alphas, m - s, m + s, color=color, alpha=.2)

    plt.semilogx()
    plt.xlabel("Alphas")
    plt.ylabel("MSE")
    # plt.semilogx()
    plt.legend()
    plt.show()


def plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test):
    """
    Plot the data and the learnt functions.
    :param n_hidden: int, number of hidden neurons
    :param x_train:
    :param y_train:
    :param y_pred_train: array of size as y_train, but representing the estimator prediction
    :param x_test:
    :param y_test:
    :param y_pred_test:  array of size as y_test, but representing the estimator prediction
    :return:
    """


    plt.figure(figsize=(10, 7))

    ax = plt.subplot()

    ax.set_title(str(n_hidden) + ' hidden neurons')
    ax.scatter(x=x_test, y=y_test, marker='x', color='red', label='Testing data')
    ax.scatter(x=x_train, y=y_train, marker='o', color='blue', label='Training data')
    ax.plot(x_test, y_pred_test, color='black', lw=2, label='Prediction')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set(ylim=[-5, 5])
    plt.savefig("plots/learned_function_{}.png".format(n_hidden))
    plt.legend()
    plt.show()
