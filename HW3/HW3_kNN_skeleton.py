#Filename: HW3_kNN_skeleton.py
#Author: Harald Leisenberger
#Edited: May, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import load_diabetes

#--------------------------------------------------------------------------------
# Assignment 3 - Section 2 (k-Nearest Neighbors)
#--------------------------------------------------------------------------------

def main():    
    
    
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!
    
    
    # 2.1 kNN for Classification
    # ----------------------------------------------
    
    # Load the two data arrays (training set: 120 x 3 - array, test set: 80 x 3 - array)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (0, 1 or 2)
    data_training = np.loadtxt('HW3_kNN_training.data')
    data_test = np.loadtxt('HW3_kNN_test.data')
    
    X_train = data_training[:,0:2], t_train = data_training[2]
    X_test = data_test[:,0:2], t_test = data_test[2]
 
    # TODO: Use the function kNN_classifyer to visualize the decision boundaries based on the training data
    #       for k=1,2,3,4,5.
    
    # TODO: Use the kNN_score to compute the classification score on the test data for k=1,2,3,4,5 and plot
    #       the results.
    
    # TODO: Compute the test score for $k=1,2,...,20$ and plot it against k.
    
    
    # 2.2 kNN for Regression (Bonus)
    # ----------------------------------------------
    
    diabetes = load_diabetes()
    blood_pressure_all = diabetes.data[:,3]
    blood_pressure = blood_pressure_all[0:40]
    diabetes_value = diabetes.target[0:40]
    
    
    # TODO: Use the function two_NN_regression to fit a function that predicts the diabetes targets in dependence of
    #       the blood pressure on the interval [-0.1,0.1] and vizualize the results.
    
    pass
    
    
#--------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
#--------------------------------------------------------------------------------


def kNN_classifyer(X_train,t_train,nr_classes,X_new,k):
    
    """ Applies k-Nearest-Neighbors to predict the value of new data based
        on the training data. 
    
    Input: X_train ... training features
           t_train ... training classes
           nr_classes ... number of classes
           X_new ... new, unseen data to be classified
           k ... number of neighbors to be taken into account for classifying new data
           
    Output: y_estimate ... estimated classes of all new data points
    """
    
    y_estimate = np.zeros(len(X_new))
    
    # TODO: Implement kNN for a general k and a general number of classes
    
    return y_estimate

#--------------------------------------------------------------------------------

def kNN_score(t_test,y_estimate):
    
    """ Evaluates the percentage of correctly classified data points on a test set 
    
    Input: t_test ... true classes of test samples
           y_estimate ... kNN-estimated classes of test samples
           
    Output: y_estimate ... estimated classes of all new data points
    """
    
    # TODO: implement the score evaluation function for kNN
    
    return score

#--------------------------------------------------------------------------------

def two_NN_regression(X_train,t_train,X_new):
    
    """ Applies 2-Nearest-Neighbors to predict the targets of new data based
        on the training data.
    
    Input: X_train ... training features
           t_train ... training targets
           X_new ... new, unseen data whose targets are to be estimated
           k ... number of neighbors to be taken into account for classifying new data
           
    Output: y_estimate ... estimated classes of all new data points
    """
    
    y_estimate = np.zeros(len(X_new))
    
    # TODO: Implement 2NN for a 1-dimensional regression problem
    
    return y_estimate

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
