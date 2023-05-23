#Filename: HW3_SVM_skeleton.py
#Author: Harald Leisenberger
#Edited: May, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

#--------------------------------------------------------------------------------
# Assignment 3 - Section 3
#--------------------------------------------------------------------------------

def main():    
    

    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!
    
    
    # 3.1 Linear SVMs
    # ----------------------------------------------
    
    # Load the linearly separable data array (50 x 3)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (-1 or 1)
    data_svm_lin = np.loadtxt('HW3_SVM_linear.data')
    X_lin = data_svm_lin[:,0:2]
    t_lin = data_svm_lin[:,2]
    
    # TODO: Implement the SMO algorithm with a linear kernel and fit a linear SVM to the data
    C = 1
    max_iter_lin = 500
    alpha0 = np.zeros(50)
    b0 = 0
    
    alpha_opt_lin_1, b_opt_lin_1, dual_values_lin_1 = SMO(X_lin,t_lin,alpha0,b0,kernel='linear',C,max_iter_lin)
 
    # TODO: Plot the decision boundary, support vectors, margin and the objective function values over the iterations
    
    # TODO: Add the sample (-1,3) with target -1 to the data and fit another SVM
    alpha0 = np.zeros(51)
    
    alpha_opt_lin_2, b_opt_lin_2, dual_values_lin_2 = SMO(X_modified,t_modified,alpha0,b0,kernel='linear',C,max_iter_lin)
    
    # 2.2 Nonlinear SVMs
    # ----------------------------------------------
    
    # Load the nonlinearly separable data array (50 x 3)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (-1 or 1)
    data_svm_nonlin = np.loadtxt('HW3_SVM_nonlinear.data')
    X_nonlin = data_svm_nonlin[:,0:2]
    t_nonlin = data_svm_nonlin[:,2]
    
    # TODO: Implement the SMO algorithm with a 3-degree polynomial kernel and fit a nonlinera SVM to the data
    C = 1
    max_iter_nonlin = 10000
    alpha0 = np.zeros(50)
    b0 = 0
    
    alpha_opt_nonlin, b_opt_nonlin, dual_values_nonlin = SMO(X_nonlin,t_nonlin,alpha0,b0,kernel='poly',C,max_iter_lin)
    
    # TODO: Plot the decision boundary and the objective function values over the iterations
    
    pass
    
    
#--------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
#--------------------------------------------------------------------------------

    
def linear_kernel(x,y):
    
    """ Evaluates value of the linear kernel for inputs x,y
    
    Input: x ... a vector in the feature space
           y ... a vector in the feature space
    
    Output: value ... value of the linear kernel applied to x,y
    """
    
    # TODO: implement linear kernel
    
    return value

#--------------------------------------------------------------------------------

def poly_kernel(x,y,d):
    
    """ Evaluates value of the d-degree polynomial kernel for inputs x,y
    
    Input: x ... a vector in the feature space
           y ... a vector in the feature space
           d ... degree of the polynomial kernel
    
    Output: value ... value of the d-degree polynomial kernel applied to x,y
    """

    # TODO: implement d-degree polynomial kernel
    
    return value

#--------------------------------------------------------------------------------

def y_dual(x,X_train,t_train,alpha,b,kernel):
    
    """ Evaluates the dual form of y(x) (see equation (3) on assignment) based on the training
        data, alpha and b for a new point x. Useful for other functions below.
    
    Input: x ... a vector in the feature space
           X_train ... the training features
           t_train ... the training targets
           alpha ... a parameter vector of the dual problem
           b ... an offset parameter
           kernel ... which kernel to be used for evaluation: must be either 'linear' or 'poly'
           
    
    Output: value ... value of the fual decision function y(x)
    """

    # TODO: evaluate y(x) in a (2D-)point x based on the given data and parameters

    if kernel == 'linear':
        
        # TODO
    
    if kernel == 'poly':
        
        degree = 3
        
        # TODO
            
    return value

#--------------------------------------------------------------------------------

def dual_predict(X_new,X_train,t_train,alpha,b,kernel):
    
    """ Predicts the target values for a given data sample X_new and for given parameters alpha,
        b and a chosen kernel, based on training features and training targets. Useful for plotting
        a decision boundaries on a grid (in combination with colormap).
    
    Input: X_new ... new 2D-data to be classified
           X_train ... the training features
           t_train ... the training targets
           alpha ... a parameter vector of the dual problem
           b ... an offset parameter
           kernel ... which kernel to be used for evaluation: must be either 'linear' or 'poly'       
    
    Output: y_estimate ... an array consisting of all target predictions for the new 2D-samples
                           to be classified in X_new
    """
    
    # TODO: implement prediction of new targets based on the decision function y(x) (see description
    #       after equation (3) on assignment)
    
    y_estimate = []
    
    if kernel == 'linear':
        
        # TODO
                
    if kernel == 'poly':
        
        degree = 3
        
        # TODO
    
    return y_estimate

#--------------------------------------------------------------------------------
    
def objective_dual(X_train,t_train,alpha,kernel):
    
    """ Computes the value of the dual objective function \tilde{L}(\alpha) given in equation (1) on the
        assignment. Is used in the implementation of SMO to compute and store the objective function value
        after each iteration of SMO.
    
    Input: X_train ... the training features
           t_train ... the training targets
           alpha ... a parameter vector of the dual problem
           kernel ... which kernel to be used for evaluation: must either be 'linear' or 'poly'
           
    
    Output: value ... value of the dual objective function
    """
    
    # TODO: implement computation of the objective function for given training features and targets, a
    #       parameter vector alpha and a kernel
    
    if kernel == 'linear':
        
        # TODO

    if kernel == 'polynomial':
        
        degree = 3
        
        # TODO 
    
    return value

#--------------------------------------------------------------------------------

def SMO(X_train,t_train,alpha0,b0,kernel,C,max_iter):
    
    """ Implementation of the SMO algorithm. Optimizes the parameter vector alpha that solves approximately
        the dual SVM program (see (1),(2) on the assignment). For a detailed description, see there.
    
    Input: X_train ... the training features
           t_train ... the training targets
           alpha0 ... initial values of parameter vector alpha
           b0 ... initial value of the offset parameter b
           kernel ... kernel to be used for evaluation: must either be 'linear' or 'poly'
           C ... a regularization parameter
           max_iter ... number of SMO iterations to be performed (1 iteration = 1 run over all alpha_n for 1, ... ,N)
           
    
    Output: alpha ... best found parameter vector for alpha; used for estimating the decision boundary
            b ... final value for offset parameter b; also used for estimating the decision boundary
            dual_values ... array with the objective function value after each SMO iteration
    """
    
    # TODO: implement computation of the objective function for given training features and targets, a
    #       parameter vector alpha and a kernel
    
    iterations = 0
    alpha = alpha0
    b = b0
    dual_values=[]
    
    if kernel == 'linear':
        
        # TODO       
            
    if kernel == 'poly':
        
        degree = 3
        
        # TODO
    
    return alpha, b, dual_values


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
