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

    D = 1
    eta = 0.5
    max_iter = 2000
    epsilon = 10 ** (-3)

    w0 = np.zeros(  # TODO: adapt length of model parameter vector to D.
    )

    def E_tilde(w): return cross_entropy_error(w, data_training, D)

    def gradient_E_tilde(w): return gradient_cross_entropy(w, data_training, D)

    w0 = np.zeros(  # TODO: adapt length of model parameter vector to D. )

        w_star, iterations, errors=gradient_descent(E_tilde, gradient_E_tilde, w0, eta, max_iter, epsilon)

    # TODO: plot errors (i.e., values of E(w)) against iterations for D = 1,2,3

    # TODO: Choose different values for the step size eta and discuss the impact on the convergence behavior (D = 1,2,3)

    # TODO: plot the decision boundaries for your models on both training and test data (D = 1,2,3)

    # TODO: Compute the percentage of correctly classified points for training and test data (D = 1,2,3)

    # TODO: fit models for D = 1,2,...,10 to the data and compute the model errors on training and test set
    eta = 0.5

    # TODO: plot number of required model parameters against D (find an analytical expression)

    # 2.2 Newton-Raphson algorithm
    # ----------------------------------------------

    # Compare the convergence behavior of the Newton-Raphson algorithm to gradient descent
    eta = 0.5

    def Hessian_E_tilde(w): return Hessian_cross_entropy(w, data_training, D)

    w_star, iterations, errors = Newton_Raphson(E_tilde, gradient_E_tilde, Hessian_E_tilde, w0, 1, max_iter, epsilon)

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

    return y


# --------------------------------------------------------------------------------

def design_matrix_logreg_2D(data, degree):
    """ Creates the design matrix for given data and 2D monomial basis functions as defined in equations ( ) - ( )

    Input: data ... a N x 3 - data array containing N data points (columns 0-1: features; column 2: targets)
           degree ... maximum degree of monomial product between feature 1 and feature 2

    Output: Phi ... the design matrix
    """

    # TODO: compute the design matrix for 2 features and 2D monomial basis functions

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

    gradient_cross_entropy = np.ones((len(w), 1))

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

    Hessian_cross_entropy = np.ones((len(w), len(w)))

    # TODO: implement Hesse matrix of the cross entropy error function for 2 features.
    #       You will have to call the function design_matrix_logreg_2D inside this definition.

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

    w_star = w0
    iterations = 0
    values = np.array([])

    # TODO: implement the Newton-Raphson algorithm

    return w_star, iterations, values


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
