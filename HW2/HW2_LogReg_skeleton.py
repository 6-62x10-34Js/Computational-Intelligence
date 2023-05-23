# Filename: HW2_LogReg_skeleton.py
# Author: Harald Leisenberger
# Edited: April, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats


# --------------------------------------------------------------------------------
# Assignment 2 - Section 2
# --------------------------------------------------------------------------------

def main():
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!

    # Load the two data arrays (training set: 400 x 3 - array, test set: 100 x 3 - array)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (0 or 1)
    data_training = np.loadtxt('HW2_LogReg_training.data')
    data_test = np.loadtxt('HW2_LogReg_test.data')

    # 2.1 Logistic model fitting -- Gradient descent
    # ----------------------------------------------

    # Fit logistic models with 2D mononomial feature transformations of degree D=1, D=2 and D=3 to the training data.
    # TODO: D = 1,2,3 apply gradient descent to fit the models
    for D in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        eta = 0.5
        max_iter = 2000
        epsilon = 10 ** (-3)

        w0 = np.zeros(sum(range(D + 2)))

        def E_tilde(w): return cross_entropy_error(w, data_training, D)

        def gradient_E_tilde(w): return gradient_cross_entropy(w, data_training, D)

        w_star, iterations, errors = gradient_descent(E_tilde, gradient_E_tilde, w0, eta, max_iter, epsilon)

        # TODO: plot errors (i.e., values of E(w)) against iterations for D = 1,2,3
        plt.figure()
        plt.plot(errors)
        plt.ylabel('E(w)')
        plt.xlabel('Iterations')
        plt.title('Errors D: {}'.format(D))
        plt.show()

        # TODO: Choose different values for the step size eta and discuss the impact on the convergence behavior (D = 1,2,3)

        # TODO: plot the decision boundaries for your models on both training and test data (D = 1,2,3)

        Plot_Decision_Boundary(data_training, w_star, D, 'Decision Boundary Train D: {}'.format(D))
        Plot_Decision_Boundary(data_test, w_star, D, 'Decision Boundary Test D: {}'.format(D))

        # TODO: Compute the percentage of correctly classified points for training and test data (D = 1,2,3)

        phi_train = design_matrix_logreg_2D(data_training, D)
        phi_test = design_matrix_logreg_2D(data_test, D)

        y_log_train = Y_predict(w_star, phi_train)
        y_log_test = Y_predict(w_star, phi_test)
        Z_train = y_log_train > .5
        Z_test = y_log_test > .5
        N_train = np.size(data_training, 0)
        Percent_train = np.sum(Z_train == data_training[:, 2]) / N_train
        N_test = np.size(data_test, 0)
        Percent_test = np.sum(Z_test == data_test[:, 2]) / N_test

        print("D: {}; Percent_train: {}; Percent_test: {}".format(D, Percent_train, Percent_test))
    # TODO: fit models for D = 1,2,...,10 to the data and compute the model errors on training and test set

    # TODO: plot number of required model parameters against D (find an analytical expression)
    D_values = range(1, 10 + 1)
    Num_param = [sum(range(D + 2)) for D in D_values]

    plt.figure()
    plt.plot(D_values, Num_param)
    plt.ylabel('# of Model Parameters')
    plt.xlabel('Degree')
    plt.title('Required model parameters')
    plt.show()

    # 2.2 Newton-Raphson algorithm
    # ----------------------------------------------

    # Compare the convergence behavior of the Newton-Raphson algorithm to gradient descent
    for D in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        eta = 0.5
        max_iter = 2000
        epsilon = 10 ** (-3)

        w0 = np.zeros(sum(range(D + 2)))

        def Hessian_E_tilde(w): return Hessian_cross_entropy(w, data_training, D)

        w_star, iterations, errors = Newton_Raphson(E_tilde, gradient_E_tilde, Hessian_E_tilde, w0, 1, max_iter,
                                                    epsilon)

        plt.figure()
        plt.plot(errors)
        plt.ylabel('E(w)')
        plt.xlabel('Iterations')
        plt.title('Errors Newton Raphson D: {}'.format(D))
        plt.show()

    pass


# --------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
# --------------------------------------------------------------------------------


def sigmoid(x):
    """ Evaluates value of the sigmoid function for input x.

    Input: x ... a real-valued number

    Output: sigmoid(x) ... value of sigmoid function (in (0,1))
    """

    # TODO: implement sigmoid function
    y = 1 / (1 + np.exp(-x))

    return y


# --------------------------------------------------------------------------------

def design_matrix_logreg_2D(data, degree):
    """ Creates the design matrix for given data and 2D monomial basis functions as defined in equations ( ) - ( )

    Input: data ... a N x 3 - data array containing N data points (columns 0-1: features; column 2: targets)
           degree ... maximum degree of monomial product between feature 1 and feature 2

    Output: Phi ... the design matrix
    """
    # TODO: compute the design matrix for 2 features and 2D monomial basis functions
    x1 = data[:, 0]
    x2 = data[:, 1]
    D = degree
    N = np.size(data, 0)
    Phi = np.ones((N, 1))
    for i_D in range(1, D + 1):
        for d1 in range(i_D, -1, -1):
            d2 = i_D - d1
            Phi_elem = x1 ** d1 * x2 ** d2
            Phi = np.append(Phi, np.asmatrix(Phi_elem).T, axis=1)

    return Phi


# --------------------------------------------------------------------------------

def cross_entropy_error(w, data, degree):
    """ Computes the cross-entropy error of a model w.r.t. given data (features + classes)

    Input: w ... the model parameter vector
           data ... a N x 3 - data array containing N data points (columns 0-1: features; column 2: targets)
           degree ... maximum degree of monomial product between feature 1 and feature 2

    Output: cross_entropy_error ... value of the cross-entropy error function \tilde{E}(w)
    """

    cross_entropy_error = 0

    # TODO: implement cross entropy error function for 2 features.
    #       You will have to call the function design_matrix_logreg_2D inside this definition.
    phi = design_matrix_logreg_2D(data, degree)
    tn = data[:, 2]
    # for i in range(1,np.size(data,1)+1)

    y_log = Y_predict(w, phi)
    N = np.size(data, 0)
    eps = 1e-10
    cross_entropy_error = (-1 / N) * np.sum(np.multiply(tn, np.log(y_log + eps)) + (1 - tn) * np.log(1 - y_log + eps))

    # WARNING: If you run into numerical instabilities /overflow during the exercise this could be
    #          due to the usage log(x) with x very close to 0. Hint: replace log(x) with log(x + epsilon)
    #          with epsilon a very small number like or 1e-10.

    return cross_entropy_error


# --------------------------------------------------------------------------------

def gradient_cross_entropy(w, data, degree):
    """ Computes the gradient of the cross-entropy error function w.r.t. a model and given data (features + classes)

    Input: w ... the model parameter vector
           data ... a N x 3 - data array containing N data points (columns 0-1: features; column 2: targets)
           degree ... maximum degree of monomial product between feature 1 and feature 2

    Output: gradient_cross_entropy ... gradient of the cross-entropy error function \tilde{E}(w)
    """

    # gradient_cross_entropy = np.ones((len(w),1))
    N = np.size(data, 0)
    phi = design_matrix_logreg_2D(data, degree)
    y_log = Y_predict(w, phi)
    tn = data[:, 2]
    gradient_cross_entropy = np.squeeze(np.asarray((-1 / N) * np.sum((tn - y_log) * phi, axis=0)))

    # TODO: implement gradient of the cross entropy error function for 2 features.
    #       You will have to call the function design_matrix_logreg_2D inside this definition.

    return gradient_cross_entropy


# --------------------------------------------------------------------------------

def Hessian_cross_entropy(w, data, degree):
    """ Computes the Hessian of the cross-entropy error function w.r.t. a model and given data (features + classes)

    Input: w ... the model parameter vector
           data ... a N x 3 - data array containing N data points (columns 0-1: features; column 2: targets)
           degree ... maximum degree of monomial product between feature 1 and feature 2

    Output: Hessian_cross_entropy ... Hesse matrix of the cross-entropy error function \tilde{E}(w)
    """

    # TODO: implement Hesse matrix of the cross entropy error function for 2 features.
    #       You will have to call the function design_matrix_logreg_2D inside this definition.

    phi = design_matrix_logreg_2D(data, degree)
    y_log = Y_predict(w, phi)
    N = np.size(data, 0)
    Hessian_cross_entropy = np.zeros((len(w), len(w)))
    for i in range(0, N):
        Hessian_cross_entropy += (1 / N) * y_log[i] * (1 - y_log[i]) * phi[i, :].T * phi[i, :]

    return Hessian_cross_entropy


# --------------------------------------------------------------------------------

def gradient_descent(fct, grad, w0, eta, max_iter, epsilon):
    """ Performs gradient descent for minimizing an arbitrary function.
    Criterion for convergence: || gradient(w_k) || < epsilon

    Input: fct ... the function to be minimized
           grad ... the gradient of the function
           w0 ... starting point for gradient descent
           eta ... step size parameter
           max_iter ... maximum number of iterations to be performed
           epsilon ... tolerance parameter that regulates convergence

    Output: w_star ... parameter vector at time of termination
            iterations ... number of iterations performed by gradient descent
            values ... values of the function to be minimized at all iterations
    """

    w_star = w0
    iterations = 0
    values = np.array([])

    for iteration in range(max_iter):
        w_star = w_star - eta * grad(w_star)
        iterations = iteration + 1
        values = np.append(values, fct(w_star))
        if np.linalg.norm(grad(w_star)) < epsilon:
            break

            # TODO: implement the gradient descent algorithm

    return w_star, iterations, values


# --------------------------------------------------------------------------------

def Newton_Raphson(fct, grad, Hessian, w0, eta, max_iter, epsilon):
    """ Newton-Raphson algorithm for minimizing an arbitrary function.
    Criterion for convergence: || gradient(w_k) || < epsilon

    Input: fct ... the function to be minimized
           grad ... the gradient of the function
           Hessian ... the Hesse matrix of the function
           w0 ... starting point for gradient descent
           eta ... step size parameter
           max_iter ... maximum number of iterations to be performed
           epsilon ... tolerance parameter that regulates convergence

    Output: w_star ... parameter vector at time of termination
            iterations ... number of iterations performed by gradient descent
            values ... values of the function to be minimized at all iterations
    """
    # TODO: implement the Newton-Raphson algorithm

    w_star = w0
    iterations = 0
    values = np.array([])

    for iteration in range(max_iter):
        w_star = w_star - eta * np.squeeze(
            np.asarray(np.linalg.pinv(Hessian(w_star)) @ grad(w_star).reshape((len(w_star), 1))))
        iterations = iteration + 1
        values = np.append(values, fct(w_star))
        if np.linalg.norm(grad(w_star)) < epsilon:
            break

    return w_star, iterations, values


# --------------------------------------------------------------------------------
def Plot_Decision_Boundary(X, w, degree, title):
    h = .02

    f1_min, f1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    f2_min, f2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    ff1, ff2 = np.meshgrid(np.arange(f1_min, f1_max, h), np.arange(f2_min, f2_max, h))

    data = np.c_[ff1.ravel(), ff2.ravel()]
    phi = design_matrix_logreg_2D(data, degree)
    y_log = Y_predict(w, phi)

    Z = y_log < .5

    # Put the result into a color plot

    Z = Z.reshape(ff1.shape)
    plt.figure(1, figsize=(10, 10))
    plt.contourf(ff1, ff2, Z, cmap=cm.RdBu, alpha=.5)
    plt.contour(ff1, ff2, Z, colors=['k'], linestyles=['--'], levels=[0.5])

    # Add the training data points
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], edgecolors='k', cmap=cm.jet, alpha=0.8)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.show()


# --------------------------------------------------------------------------------
def Y_predict(w, phi):
    y_log = np.squeeze(np.asarray(sigmoid(np.asmatrix(w) @ phi.T)).T)
    return y_log


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
