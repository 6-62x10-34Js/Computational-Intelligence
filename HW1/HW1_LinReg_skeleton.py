# Filename: HW1_LinReg_skeleton.py
# Author: Harald Leisenberger
# Edited: March, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
import timeit


# --------------------------------------------------------------------------------
# Assignment 1 - Section 3
# --------------------------------------------------------------------------------

def main():
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!

    # Load the two data arrays (training set: 30 x 2 - array, test set: 21 x 2 - array)
    data_train = np.loadtxt('HW1_LinReg_train.data')

    data_test = np.loadtxt('HW1_LinReg_test.data')


    # 3.2 Linear Regression with Polynomial Features
    # ----------------------------------------------

    ## Fit polynomials of degree D in range {1,20,1} to the training data.

    degree = 20

    # TODO: for D = 1,...,20 create the design matrices and fit the models
    Phi_train_D = design_matrix(data_train, degree)
    w_train_D = opt_weight(data_train, Phi_train_D)
    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)

    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.

    # Compute the training and test errors of your models and plot the errors for D in {1,2,...,20}
    error_train_D_O = sum_of_squared_errors(data_train, w_train_D)
    error_test_D_O = sum_of_squared_errors(data_test, w_train_D)
    # TODO: plot the training / test errors against the degree of the polynomial.
    reg_degree_2 = np.linspace(1, 10, 10, endpoint=True, dtype=int)
    # Switch the role of the training data set and the test data set and repeat all of the previous steps.
    # The main difference is now to train the weight vectors on the test data.
    # TODO

    # Repeat the tasks from the first two steps (with the original roles of training and test data again), 
    # but now by make use of the regularized cost function:
    lambda_reg = 1

    # Fit polynomials of degree D in {1,2,...,20} to the training data.
    w_train_reg_D = opt_weight_reg(data_train, Phi_train_D, lambda_reg)
    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)
    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.
    degree_range = [1, 2, 9, 16]
    plot_range_of_degree(data_train, data_test, degree_range, w_train_reg_D)

    # Compute the regularized training and test errors of your models and plot the errors for D in {1,2,...,20}
    reg_degree = np.linspace(1, 20, 20, endpoint=True, dtype=int)
    error_train_D = np.zeros(len(reg_degree))
    error_test_D = np.zeros(len(reg_degree))
    reg_error_train_D = np.zeros(len(reg_degree))
    reg_error_test_D = np.zeros(len(reg_degree))

    for i in range(len(reg_degree)):
        Phi_train = design_matrix(data_train, reg_degree[i])
        w_train = opt_weight(data_train, Phi_train)

        error_train_D[i] = sum_of_squared_errors(data_train, w_train)
        error_test_D[i] = sum_of_squared_errors(data_test, w_train)

    plot_error_ranges_of_degree(error_train_D, error_test_D, reg_degree)

    # for i in range(len(reg_degree)):
    #     Phi_train_reg_D = design_matrix(data_train, reg_degree[i])
    #     w_train_reg_D = opt_weight_reg(data_train, Phi_train_reg_D, lambda_reg)
    #     reg_error_train_D[i] = reg_sum_of_squared_errors(data_train, w_train_reg_D, lambda_reg, reg_degree[i])
    #     reg_error_test_D[i] = reg_sum_of_squared_errors(data_test, w_train_reg_D, lambda_reg, reg_degree[i])
    #
    # plot_error_ranges_of_degree_reg(reg_error_train_D, reg_error_test_D, reg_degree, lambda_reg)

    lamda_range = [0.01, 0.1, 1, 3000]
    plot_range_of_degree_reg(data_train, data_test, degree_range, lambda_reg)

    for i in range(len(lamda_range)):
        for j in range(len(reg_degree)):
            Phi_train_reg_D = design_matrix(data_train, reg_degree[j])
            w_train_reg_D = opt_weight_reg(data_train, Phi_train_reg_D, lamda_range[i])
            reg_error_train_D[j] = reg_sum_of_squared_errors(data_train, w_train_reg_D, lamda_range[i], reg_degree[j])
            reg_error_test_D[j] = reg_sum_of_squared_errors(data_test, w_train_reg_D, lamda_range[i], reg_degree[j])
        plot_error_ranges_of_degree_reg(reg_error_train_D, reg_error_test_D, reg_degree, lamda_range[i])





    # 3.3 Linear Regression with Sigmoidal Features
    # ----------------------------------------------

    # Repeat the first two steps from 3.2, but now use sigmoidal basis functions. Use the original error function.
    # TODO: The main difference is the construction of the design matrix based on sigmoidal basis functions.
    #       Therefore, implement a helper function design_matrix_sigmoid(data,nr_basis_functions,x_min,x_max)

    pass


# --------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
# --------------------------------------------------------------------------------
def plot_error_ranges_of_degree(error_train_D, error_test_D, reg_degree):
    """ plots the training and test errors against the degree of the polynomial.

    Input:  reg_error_train_D ... the training errors (1 x D array)
            reg_error_test_D ... the test errors (1 x D array)
            reg_degree ... the degrees of the polynomials (1 x D array)
    """
    for i in range(len(reg_degree)):
        print('Degree: ', reg_degree[i], 'Training Error: ', error_train_D[i], 'Test Error: ', error_test_D[i])

    fig, ax = plt.subplots()
    ax.plot(reg_degree, error_train_D, label='Training Error')
    ax.plot(reg_degree, error_test_D, label='Test Error')
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Error')
    ax.set_title('Error of Linear Regression without Regularization')
    ax.legend()
    plt.savefig('plots/lon/error_of_linear_regression_compare_train_test_NON_reg.png')
    plt.show()


    difference = abs(error_train_D - error_test_D)
    print('Minimum Difference: ', np.min(difference))
    print('Degree of Minimum Difference: ', np.argmin(difference))
    print('Maximum Difference: ', np.max(difference))
    print('Degree of Maximum Difference: ', np.argmax(difference))

    fig, ax = plt.subplots()
    ax.plot(reg_degree, difference, label='Difference')
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Absolute Difference')
    ax.set_title('Absolute Difference of Training and Test Error without Regularization')
    ax.legend()
    plt.savefig(f'plots/lon/difference_of_training_and_test_error_NON_reg.png')
    plt.show()




def plot_error_ranges_of_degree_reg(reg_error_train_D, reg_error_test_D, reg_degree, lambda_reg):
    """ plots the training and test errors against the degree of the polynomial.

    Input:  reg_error_train_D ... the training errors (1 x D array)
            reg_error_test_D ... the test errors (1 x D array)
            reg_degree ... the degrees of the polynomials (1 x D array)
            lambda_reg ... the regularization parameter
    """

    for i in range(len(reg_degree)):
        print('Degree: ', reg_degree[i], 'Training Error: ', reg_error_train_D[i], 'Test Error: ', reg_error_test_D[i])

    fig, ax = plt.subplots()
    ax.plot(reg_degree, reg_error_train_D, label='Training Error')
    ax.plot(reg_degree, reg_error_test_D, label='Test Error')
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Error')
    ax.set_title(f'Error of Linear Regression with Regularization (lambda = {lambda_reg})')
    ax.legend()
    ax.set_ylim([0, 20])
    plt.savefig(f'plots/lon/error_of_linear_regression_REG_{lambda_reg}_LOL.png')
    plt.show()

    difference = reg_error_train_D - reg_error_test_D
    print('Minimum Difference: ', np.min(difference))
    print('Degree of Minimum Difference: ', np.argmin(difference))
    print('Maximum Difference: ', np.max(difference))
    print('Degree of Maximum Difference: ', np.argmax(difference))

    fig, ax = plt.subplots()
    ax.plot(reg_degree, difference, label='Difference')
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Difference')
    ax.set_title(f'Difference of Training and Test Error Regularization (lambda = {lambda_reg})')
    ax.legend()
    plt.savefig(f'plots/lon/difference_of_training_and_test_error_REG_{lambda_reg}_LOL.png')
    plt.show()

def plot_range_of_degree_reg(data_train, data_test, degree_range, lamda_reg):
    """ plots the training and test data and the models for a range of polynomial degrees in one figure.

    Input:  data_train ... the training data (N x 2 array)
            data_test ... the test data (N x 2 array)
            degree_range ... a list of polynomial degrees to be considered for model fitting.
            lamda_range ... a list of regularization parameters to be considered for model fitting.
    """

    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)

    fig, ax = plt.subplots()

    for i in range(len(degree_range)):
        Phi_train_D = design_matrix(data_train, degree_range[i])
        w_train_D = opt_weight_reg(data_train, Phi_train_D, lamda_reg)
        x = np.linspace(-1, 1, 30)
        y = np.zeros(len(x))


        for j in range(len(x)):
            y[j] = y_predict(w_train_D, x[j])

        r_squared = calc_r_squared(data_train[:, 1], y)
        print('Degree: ', degree_range[i], 'R^2: ', r_squared)
        ax.plot(x, y, label='Fit with degree = {}'.format(degree_range[i])+f' R^2 = {r_squared:.3f}')

    ax.scatter(data_train[:, 0], data_train[:, 1], label='training data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data Regulized')
    ax.legend()
    plt.savefig(f'plots/lon/lin_range_of_degree_{degree_range}_trainingdata_reg_{lamda_reg}.png')
    plt.show()

    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.

    fig, ax = plt.subplots()

    for i in range(len(degree_range)):
        Phi_train_D = design_matrix(data_test, degree_range[i])
        w_train_D = opt_weight_reg(data_test, Phi_train_D, lamda_reg)
        x = np.linspace(-1, 1, 30)
        y = np.zeros(len(x))

        for j in range(len(x)):
            y[j] = y_predict(w_train_D, x[j])

        r_squared = calc_r_squared(data_train[:, 1], y)
        print('Degree: ', degree_range[i], 'R^2: ', r_squared)
        ax.plot(x, y, label='Fit with degree = {}'.format(degree_range[i])+f' R^2 = {r_squared:.3f}')

    ax.scatter(data_test[:, 0], data_test[:, 1], label='test data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Test Data Regulized')
    ax.legend()
    plt.savefig(f'plots/lon/range_of_degree_{degree_range}_testdata_reg_{lamda_reg}.png')

def plot_range_of_degree(data_train, data_test, degree_range, w_train_reg_D):
    """ plots the training and test data and the models for a range of polynomial degrees in one figure.

    Input:  data_train ... the training data (N x 2 array)
            data_test ... the test data (N x 2 array)
            degree_range ... a list of polynomial degrees to be considered for model fitting.
    """

    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)

    fig, ax = plt.subplots()

    for i in range(len(degree_range)):
        Phi_train_D = design_matrix(data_train, degree_range[i])
        w_train_D = opt_weight(data_train, Phi_train_D)
        x = np.linspace(-1, 1, 30)
        y = np.zeros(len(x))


        for j in range(len(x)):
            y[j] = y_predict(w_train_D, x[j])

        r_squared = calc_r_squared(data_train[:, 1], y)
        print('Degree: ', degree_range[i], 'R^2: ', r_squared)
        ax.plot(x, y, label='Fit with degree = {}'.format(degree_range[i])+f' R^2 = {r_squared:.3f}')

    ax.scatter(data_train[:, 0], data_train[:, 1], label='training data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data')
    ax.legend()
    plt.savefig(f'plots/lon/lin_range_of_degree_{degree_range}_trainingdata.png')
    plt.show()

    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.

    fig, ax = plt.subplots()

    for i in range(len(degree_range)):
        Phi_train_D = design_matrix(data_test, degree_range[i])
        w_train_D = opt_weight(data_test, Phi_train_D)
        x = np.linspace(-1, 1, 30)
        y = np.zeros(len(x))

        for j in range(len(x)):
            y[j] = y_predict(w_train_D, x[j])

        r_squared = calc_r_squared(data_train[:, 1], y)
        print('Degree: ', degree_range[i], 'R^2: ', r_squared)
        ax.plot(x, y, label='Fit with degree = {}'.format(degree_range[i])+f' R^2 = {r_squared:.3f}')

    ax.scatter(data_test[:, 0], data_test[:, 1], label='test data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Test Data')
    ax.legend()
    plt.savefig(f'plots/lon/range_of_degree_{degree_range}_testdata.png')



def design_matrix(data, degree):
    """ computes the design matrix for given data and polynomial basis functions up to a certain degree
    
    Input:  data ... an N x 2 array (1st column: feature values; 2st column: targets)
            degree ... maximum degree D of the polynomial basis function.
            
    Output: Phi ... the design matrix (has to be a N x (D + 1) - array)
    """

    Phi = np.zeros([len(data), degree + 1])

    # compute design matrix

    for i in range(len(data)):
        for j in range(degree + 1):
            Phi[i][j] = data[i][0]**j

    return Phi


# --------------------------------------------------------------------------------

def design_matrix_sigmoid(data, nr_basis_functions, x_min, x_max):
    """ computes the design matrix for given data and polynomial basis functions up to a certain degree
    
    Input:  data ... an N x 2 array (1st column: feature values; 2st column: targets)
            nr_basis_functions ... number D + 1 of the sigmoidal basis function.
            x_min ... lower bound of the x range interval
            x_max ... upper bound of the x range interval
            
    Output: Phi_sig ... the design matrix (has to be a N x (D + 1) - array)
    """

    Phi_sig = np.zeros([len(data), nr_basis_functions])

    # TODO: Create the design matrix for a given data array and a fixed number of sigmoidal basis function.
    #       First, the centers and the width of the basis functions must be specified, according to the
    #       range [x_min,x_max] to be considered for model fitting.

    return Phi_sig


# --------------------------------------------------------------------------------


def opt_weight(data, Phi):
    """ computes the optimal weight vector for a given design matrix and the corresponding targets.
    
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           Phi ... the design matrix (N x (D + 1)) that corresponds to the data
    
    Output: w_star ... the optimal weight vector ((D + 1) x 1)
    """


    # TODO Compute the optimal weight vector (with respect to the unmodified error function) for given
    #      targets and a given design matrix.

    w_star = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T), data[:, 1])

    return w_star


# --------------------------------------------------------------------------------

def opt_weight_reg(data, Phi, lambda_reg):
    """ computes the optimal weight vector for a given design matrix and the corresponding targets, when
        considering the regularized error function with regularization parameter lambda.
    
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           Phi ... the design matrix (N x (D + 1)) that corresponds to the data
           lambda_reg ... the regularization parameter
    
    Output: w_star_reg ... the optimal weight vector ((D + 1) x 1)
    """

    # TODO: Compute the optimal weight vector (with respect to the regularized error function) for given
    #       targets and a given design matrix.

    w_star_reg = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + lambda_reg * np.identity(len(Phi.T))), Phi.T),
                        data[:, 1])

    return w_star_reg


# --------------------------------------------------------------------------------

def y_predict(w_train_D, x_value):
    """ evaluates the predicted value for y of a given model defined by the corresponding weight vector
        in a given value x.
        
    Input: w_train_D ... the optimal weight vector for a polynomial model with degree D that has been
                         trained on the training data
           x_value ... an arbitrary value on the x-axis in that the y-value of the model should be computed
    
    Output: y_value ... the y-value corresponding to the given x-value for the given model
    """

    # TODO: Compute the predicted y-value for a given model with degree D and its weight vector w_train_D in x

    y_value = 0
    for i in range(len(w_train_D)):
        y_value += w_train_D[i] * (x_value ** i)

    return y_value


# --------------------------------------------------------------------------------

def sum_of_squared_errors(data, w_star):
    """ Computes the sum of squared errors between the values Phi*w that are predicted by a model
        and the actual targets t.
        
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           w_star ... the optimum weight vector corresponding to a certain model
           
    Output: error ... the sum of squared errors E(w) for a given data array (the output must be a number!)
                      and a given model (= weight vector).
    """

    # TODO: Evaluate the error of a model (defined by w_star) on a given data array.
    error = 0

    for i in range(len(data)):
        error += (data[i][1] - y_predict(w_star, data[i][0])) ** 2

    return error


# --------------------------------------------------------------------------------

def reg_sum_of_squared_errors(data_train, w_star, lambda_reg, reg_degree):
    """ Computes the sum of squared errors between the values Phi*w that are predicted by a model
        and the actual targets t.
        
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           w_star ... the optimum weight vector corresponding to a certain model
           lambda_reg ... the value for the regularization parameter
           reg_degree ... the degree of the polynomial model that is used for regularization
           
    Output: reg_error ... the value of the lambda-regularized cost function E_lambda for a given data array and
                          a given model (= weight vector).
    """

    # TODO: Evaluate the regularized error of a model (defined by w_star) on a given data array.

    reg_error = 0
    for i in range(len(data_train)):
        reg_error += (data_train[i][1] - y_predict(w_star, data_train[i][0])) ** 2

    for i in range(len(w_star)):
        reg_error += lambda_reg * (w_star[i] ** reg_degree)

    return reg_error

def calc_r_squared(y, y_pred):
    """
    Calculate the R^2 value for a given set of actual and predicted y values.
    """
    sst = np.sum((y - np.mean(y))**2)
    ssr = np.sum((y - y_pred)**2)
    r_squared = 1 - (ssr / sst)
    return r_squared
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
