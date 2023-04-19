#Filename: HW1_LinReg_skeleton.py
#Author: Harald Leisenberger
#Edited: March, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

#--------------------------------------------------------------------------------
# Assignment 1 - Section 3
#--------------------------------------------------------------------------------

def main():    
    
    
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!
    
    
    # Load the two data arrays (training set: 30 x 2 - array, test set: 21 x 2 - array)
    data_train = np.loadtxt('HW1_LinReg_train.data')
    data_test = np.loadtxt('HW1_LinReg_test.data')
    
    
    # 3.2 Linear Regression with Polynomial Features
    # ----------------------------------------------
    
    ## Fit polynomials of degree D in {1,2,...,20} to the training data.
    # TODO: for D = 1,...,20 create the design matrices and fit the models
    Phi_train_D = design_matrix(data_train,degree)
    w_train_D = opt_weight(data_train,Phi_train_D)
    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)
    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.
    
    
    # Compute the training and test errors of your models and plot the errors for D in {1,2,...,20}
    error_train_D = sum_of_squared_errors(data_train,w_train_D)
    error_test_D = sum_of_squared_errors(data_test,w_train_D)
    # TODO: plot the training / test errors against the degree of the polynomial.
    
    
    # Switch the role of the training data set and the test data set and repeat all of the previous steps.
    # The main difference is now to train the weight vectors on the test data.
    # TODO
    
    
    # Repeat the tasks from the first two steps (with the original roles of training and test data again), 
    # but now by make use of the regularized cost function:
    lambda_reg = 0.1
    
    
    # Fit polynomials of degree D in {1,2,...,20} to the training data.
    w_train_reg_D = opt_weight_reg(data_train,Phi_train_D,lambda_reg)
    # TODO: for D = 1,2,9,16 plot the models on [-1,1]. More concretely:
    # TODO: plot the models (continuous plots) and compare them to the training targets (scatter points)
    # TODO: plot the models (continuous plots) and compare them to the test targets (scatter points)
    # For evaluating the y values of a certain model in the values x, you should use the helper function y_predict.
    
    
    # Compute the regularized training and test errors of your models and plot the errors for D in {1,2,...,20}
    reg_error_train_D = reg_sum_of_squared_errors(data_train,w_train_reg_D,lambda_reg)
    reg_error_test_D = reg_sum_of_squared_errors(data_test,w_train_reg_D,lambda_reg)
    # TODO: plot the regularized training / test errors against the degree of the polynomial.
    
    
    # 3.3 Linear Regression with Sigmoidal Features
    # ----------------------------------------------
    
    # Repeat the first two steps from 3.2, but now use sigmoidal basis functions. Use the original error function.
    # TODO: The main difference is the construction of the design matrix based on sigmoidal basis functions.
    #       Therefore, implement a helper function design_matrix_sigmoid(data,nr_basis_functions,x_min,x_max)
    
    
    pass
    
    
#--------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
#--------------------------------------------------------------------------------

    
def design_matrix(data,degree):
    
    """ computes the design matrix for given data and polynomial basis functions up to a certain degree
    
    Input:  data ... an N x 2 array (1st column: feature values; 2st column: targets)
            degree ... maximum degree D of the polynomial basis function.
            
    Output: Phi ... the design matrix (has to be a N x (D + 1) - array)
    """
    
    Phi = np.zeros([len(data),degree+1])
    
    # TODO: Create the design matrix for a given data array and a fixed number of polynomial basis function.
    
    return Phi

#--------------------------------------------------------------------------------

def design_matrix_sigmoid(data,nr_basis_functions,x_min,x_max):

    """ computes the design matrix for given data and polynomial basis functions up to a certain degree
    
    Input:  data ... an N x 2 array (1st column: feature values; 2st column: targets)
            nr_basis_functions ... number D + 1 of the sigmoidal basis function.
            x_min ... lower bound of the x range interval
            x_max ... upper bound of the x range interval
            
    Output: Phi_sig ... the design matrix (has to be a N x (D + 1) - array)
    """
    
    Phi_sig = np.zeros([len(data),nr_basis_functions])
    
    # TODO: Create the design matrix for a given data array and a fixed number of sigmoidal basis function.
    #       First, the centers and the width of the basis functions must be specified, according to the
    #       range [x_min,x_max] to be considered for model fitting.

    return Phi_sig

#--------------------------------------------------------------------------------


def opt_weight(data,Phi):

    """ computes the optimal weight vector for a given design matrix and the corresponding targets.
    
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           Phi ... the design matrix (N x (D + 1)) that corresponds to the data
    
    Output: w_star ... the optimal weight vector ((D + 1) x 1)
    """
    
    w_star = np.zeros([len(Phi.T),1])
    
    # TODO Compute the optimal weight vector (with respect to the unmodified error function) for given
    #      targets and a given design matrix.

    return w_star

#--------------------------------------------------------------------------------

def opt_weight_reg(data,Phi,lambda_reg):

    """ computes the optimal weight vector for a given design matrix and the corresponding targets, when
        considering the regularized error function with regularization parameter lambda.
    
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           Phi ... the design matrix (N x (D + 1)) that corresponds to the data
           lambda_reg ... the regularization parameter
    
    Output: w_star_reg ... the optimal weight vector ((D + 1) x 1)
    """
    
    w_star_reg = np.zeros([len(Phi.T),1])
    
    # TODO: Compute the optimal weight vector (with respect to the regularized error function) for given
    #       targets and a given design matrix.

    return w_star_reg

#--------------------------------------------------------------------------------

def y_predict(w_train_D,x_value):

    """ evaluates the predicted value for y of a given model defined by the corresponding weight vector
        in a given value x.
        
    Input: w_train_D ... the optimal weight vector for a polynomial model with degree D that has been
                         trained on the training data
           x_value ... an arbitrary value on the x-axis in that the y-value of the model should be computed
    
    Output: y_value ... the y-value corresponding to the given x-value for the given model
    """
    
    # TODO: Compute the predicted y-value for a given model with degree D and its weight vector w_train_D in x.

    return y_value

#--------------------------------------------------------------------------------

def sum_of_squared_errors(data,w_star)

    """ Computes the sum of squared errors between the values Phi*w that are predicted by a model
        and the actual targets t.
        
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           w_star ... the optimum weight vector corresponding to a certain model
           
    Output: error ... the sum of squared errors E(w) for a given data array (the output must be a number!)
                      and a given model (= weight vector).
    """

    # TODO: Evaluate the error of a model (defined by w_star) on a given data array.

    return error

#--------------------------------------------------------------------------------    
    
def reg_sum_of_squared_errors(data_train,w_train_reg_D,lambda_reg)

    """ Computes the sum of squared errors between the values Phi*w that are predicted by a model
        and the actual targets t.
        
    Input: data ... N x 2 array (1st column: feature values; 2st column: targets)
           w_star ... the optimum weight vector corresponding to a certain model
           lambda_reg ... the value for the regularization parameter
           
    Output: reg_error ... the value of the lambda-regularized cost function E_lambda for a given data array and
                          a given model (= weight vector).
    """

    # TODO: Evaluate the regularized error of a model (defined by w_star) on a given data array.

    return reg_error

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
