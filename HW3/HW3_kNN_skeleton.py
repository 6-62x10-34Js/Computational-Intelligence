# Filename: HW3_kNN_skeleton.py
# Author: Harald Leisenberger
# Edited: May, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import load_diabetes


# --------------------------------------------------------------------------------
# Assignment 3 - Section 2 (k-Nearest Neighbors)
# --------------------------------------------------------------------------------

def main():
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!

    # 2.1 kNN for Classification
    # ----------------------------------------------

    # Load the two data arrays (training set: 120 x 3 - array, test set: 80 x 3 - array)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (0, 1 or 2)
    data_training = np.loadtxt('HW3_kNN_training.data')
    data_test = np.loadtxt('HW3_kNN_test.data')

    # Split the data into features and class labels
    X_train = data_training[:, 0:2]
    t_train = data_training[:, 2]
    X_test = data_test[:, 0:2]
    t_test = data_test[:, 2]

    print(X_train.shape)
    print(t_train.shape)
    print(X_test.shape)
    print(t_test.shape)

    len_classes = len(np.unique(t_train))

    # TODO: Use the function kNN_classifyer to visualize the decision boundaries based on the training data
    #       for k=1,2,3,4,5.

    k = [1, 2, 3, 4, 5]
    res_dict_base = {}
    for i in k:
        y_pred = kNN_classifyer(X_train, t_train, len_classes, X_test, i)
        score = kNN_score(y_pred, t_test)
        res_dict_base[i] = score
        plot_decision_boundary(X_train, t_train, i, score, label=f'Training data points for k={i}')


    # TODO: Use the kNN_score to compute the classification score on the test data for k=1,2,3,4,5 and plot
    #       the results.
    res_dict = {}

    for i in k:
        y_pred = kNN_classifyer(X_train, t_train, len_classes, X_test, i)
        score = kNN_score(y_pred, t_test)
        res_dict[i] = score
        #visualize the test data points together with the decision boundaries
        plot_decision_boundary(X_test, t_test, i, score, label= f'Test data points for k={i}')
    print(res_dict)


    # TODO: Compute the test score for $k=1,2,...,20$ and plot it against k.


    k = np.arange(1, 21)
    res_dict_test = {}
    for i in k:
        y_pred = kNN_classifyer(X_train, t_train, len_classes, X_test, i)
        score = kNN_score(y_pred, t_test)
        res_dict_test[i] = score

    plt.plot(res_dict_test.keys(), res_dict_test.values())
    plt.xlabel('k')
    plt.ylabel('score')
    plt.title('Test score for k=1,2,...,20' + '\n' + 'Highest score: ' + str(max(res_dict_test.values())) + 'at k=' + str(list(res_dict_test.keys())[list(res_dict_test.values()).index(max(res_dict_test.values()))]))
    plt.savefig('test_score.png')
    plt.show()

    highest_score = max(res_dict_test.values())
    print(f'Highest score Test: {highest_score}')
    print(f'k: {list(res_dict.keys())[list(res_dict.values()).index(highest_score)]}')


    # 2.2 kNN for Regression (Bonus)
    # ----------------------------------------------

    diabetes = load_diabetes()
    blood_pressure_all = diabetes.data[:, 3]
    blood_pressure = blood_pressure_all[0:40]
    diabetes_value = diabetes.target[0:40]

    # TODO: Use the function two_NN_regression to fit a function that predicts the diabetes targets in dependence of
    #       the blood pressure on the interval [-0.1,0.1] and vizualize the results.

    pass


# --------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
# --------------------------------------------------------------------------------
def plot_decision_boundary(X_train, t_train, k, score, label):
    """ Plots the decision boundary for a given training set and a given k.

    Input: X_train ... training features
           t_train ... training classes
           k ... number of neighbors to be taken into account for classifying new data

    Output: y_estimate ... estimated classes of all new data points
    """
    len_classes = len(np.unique(t_train))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # Create a meshgrid for the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Create new data points
    X_new = np.c_[xx.ravel(), yy.ravel()]

    # make the prediction for the new data points
    y_estimate = kNN_classifyer(X_train, t_train, len_classes, X_new, k)

    # Plot the decision boundary
    # reshape the estimated classes according to the meshgrid
    Z = y_estimate.reshape(xx.shape)

    # plot the contour lines
    plt.contourf(xx, yy, Z, cmap=cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, cmap=cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{label} (score: {score})')
    #plt.savefig(f'{label}_{score}.png')
    plt.show()



def kNN_classifyer(X_train, t_train, nr_classes, X_new, k):
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
    #For breaking ties between classes with the same number of samples among the k nearest neighbors, a random decision has to be performed.
    #For this purpose, you can use the function np.random.choice().



    # calculate the distance between each data point and each new data point (using the euclidean distance)
    dist_arr = np.sqrt(np.sum((X_train[:, np.newaxis] - X_new[np.newaxis, :]) ** 2, axis=2))

    # find the k nearest neighbors for each data point
    idx = np.argpartition(dist_arr, k, axis=0)[:k]

    # find the most frequent class among the k nearest neighbors
    nearest_classes = t_train[idx[:k]].astype(int)

    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=nr_classes), axis=0, arr=nearest_classes)

    # most frequent class
    y_estimate = np.argmax(counts, axis=0)

    # tie breaker
    for i in range(y_estimate.shape[0]):
        max_count = np.max(counts[:, i])
        max_indices = np.where(counts[:, i] == max_count)[0]
        if len(max_indices) > 1:
            tie_breaker = np.random.choice(max_indices)
            y_estimate[i] = tie_breaker

    return y_estimate


# --------------------------------------------------------------------------------

def kNN_score(t_test, y_estimate):
    """ Evaluates the percentage of correctly classified data points on a test set
    
    Input: t_test ... true classes of test samples
           y_estimate ... kNN-estimated classes of test samples
           
    Output: y_estimate ... estimated classes of all new data points
    """

    # Calculate the accuracy score as the percentage of correctly classified data points on the test set over all data points
    score = np.sum(t_test == y_estimate) / len(t_test)

    return score


# --------------------------------------------------------------------------------

def two_NN_regression(X_train, t_train, X_new):
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


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
