# Filename: HW3_SVM_skeleton.py
# Author: Harald Leisenberger
# Edited: May, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats


# --------------------------------------------------------------------------------
# Assignment 3 - Section 3
# --------------------------------------------------------------------------------

def main():
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!

    # 3.1 Linear SVMs
    # ----------------------------------------------

    # Load the linearly separable data array (50 x 3)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (-1 or 1)
    data_svm_lin = np.loadtxt('HW3_SVM_linear.data')
    X_lin = data_svm_lin[:, 0:2]
    t_lin = data_svm_lin[:, 2]

    C = 1
    max_iter_lin = 500
    alpha0 = np.zeros(50)
    b0 = 0

    # alpha_opt_lin_1, b_opt_lin_1, dual_values_lin_1 = SMO(X_lin, t_lin, alpha0, b0, 'linear', C, max_iter_lin)
    #
    # # TODO: Plot the decision boundary, support vectors, margin and the objective function values over the iterations
    # plot_decision_boundary(X_lin, t_lin, alpha_opt_lin_1, b_opt_lin_1, 'linear', dual_values_lin_1, title='linear')

    # TODO: Add the sample (-1,3) with target -1 to the data and fit another SVM
    alpha0 = np.zeros(51)
    X_modified = np.append(X_lin, [[-1, 3]], axis=0)
    t_modified = np.append(t_lin, [-1], axis=0)

    # alpha_opt_lin_2, b_opt_lin_2, dual_values_lin_2 = SMO(X_modified, t_modified, alpha0, b0, 'linear', C, max_iter_lin)
    # plot_decision_boundary(X_modified, t_modified, alpha_opt_lin_2, b_opt_lin_2, 'linear', dual_values_lin_2, title='linear modified')
    # 2.2 Nonlinear SVMs
    # ----------------------------------------------

    # Load the nonlinearly separable data array (50 x 3)
    # Column 1: feature 1; Column 2: feature 2; Column 3: class label (-1 or 1)
    data_svm_nonlin = np.loadtxt('HW3_SVM_nonlinear.data')
    X_nonlin = data_svm_nonlin[:, 0:2]
    t_nonlin = data_svm_nonlin[:, 2]

    # TODO: Implement the SMO algorithm with a 3-degree polynomial kernel and fit a nonlinera SVM to the data
    C = 1
    max_iter_nonlin = 100
    alpha0 = np.zeros(50)
    b0 = 0

    alpha_opt_nonlin, b_opt_nonlin, dual_values_nonlin = SMO(X_nonlin, t_nonlin, alpha0, b0, 'poly', C, max_iter_nonlin)

    print("done with SMO")
    plot_decision_boundary(X_nonlin, t_nonlin, alpha_opt_nonlin, b_opt_nonlin, 'poly', dual_values_nonlin, title='polynomial')

    # TODO: Plot the decision boundary and the objective function values over the iterations

    pass


# --------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
# --------------------------------------------------------------------------------

def plot_decision_boundary(X, t, alpha, b, kernel, dual_values, title):
    """ Plots the decision boundary, the margin and the support vectors for a given data set and a trained SVM. """
    fig1, ax1 = plt.subplots()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a meshgrid of points with a given resolution
    resolution = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    # Evaluate the SVM decision function for each point on the grid
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if i % 100 == 0 and j % 100 == 0:
                print('still plotting...')
            x = [xx[i, j], yy[i, j]]
            print(x)
            Z[i, j] = y_dual(x, X, t, alpha, b, kernel)
            print(Z[i, j])

    # Plot the SVM decision boundary and the margins
    cs_bound = ax1.contourf(xx, yy, Z, cmap=cm.coolwarm, alpha=0.8)
    cs2 = ax1.contour(cs_bound, colors='black', levels=cs_bound.levels[::2], alpha=0.5, linestyles=['--', '-', '--'],
                            linewidths=[2, 4, 2])
    ax1.clabel(cs2, fmt='%2.1f', colors='k', fontsize=10)

    # Plot the training data
    ax1.scatter(X[:, 0], X[:, 1], c=t, cmap=cm.coolwarm, label='data')

    #  plotting the straight lines running
    # through the support vectors, for either class)

    ax1.scatter(X[alpha > 0, 0], X[alpha > 0, 1], c=t[alpha > 0], cmap=cm.bone, s=100, marker='x',
                label='support vectors', linewidth=4)

    cbar = fig1.colorbar(cs_bound)
    cbar.add_lines(cs2)
    cbar.ax.set_ylabel('SVM decision function value')


    # Plot the objective function values over the iterations
    fig1.legend()
    ax1.set_title('SVM with ' + title + ' kernel')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    fig1.savefig('SVM_' + title + '_kernel.png')


    plt.show()

    plt.figure()
    plt.plot(dual_values)
    plt.xlabel('Iteration')
    plt.legend()
    plt.ylabel('Objective function value')
    plt.title('Objective function value over the iterations')

    plt.show()


def linear_kernel(x, y):
    """ Evaluates value of the linear kernel for inputs x,y

    Input: x ... a vector in the feature space
           y ... a vector in the feature space

    Output: value ... value of the linear kernel applied to x,y
    """

    # TODO: implement linear kernel

    value = np.dot(x.T, y)

    return value


# --------------------------------------------------------------------------------

def poly_kernel(x, y, d):
    """ Evaluates value of the d-degree polynomial kernel for inputs x,y

    Input: x ... a vector in the feature space
           y ... a vector in the feature space
           d ... degree of the polynomial kernel

    Output: value ... value of the d-degree polynomial kernel applied to x,y
    """

    # TODO: implement d-degree polynomial kernel
    value = (np.dot(x.T, y) + 1) ** d

    return value


# --------------------------------------------------------------------------------

def y_dual(x, X_train, t_train, alpha, b, kernel):
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

    max_pass = 500
    value = 0
    if kernel == 'linear':
        y = np.zeros(X_train.shape[0])
        for i in range(0, X_train.shape[0]):
            y[i] = alpha[i] * t_train[i] * linear_kernel(X_train[i], x) + b

        value = np.sum(y)

    if kernel == 'poly':

        degree = 3

        y = np.zeros(X_train.shape[0])
        for i in range(0, X_train.shape[0]):
            y[i] = alpha[i] * t_train[i] * poly_kernel(X_train[i], x, degree) + b

        value = np.sum(y)

    return value


# --------------------------------------------------------------------------------

def dual_predict(X_new, X_train, t_train, alpha, b, kernel):
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

        y_estimate = np.zeros(X_new.shape[0])
        for i in range(0, X_new.shape[0]):
            y_estimate[i] = y_dual(X_new[i], X_train, t_train, alpha, b, kernel)

    if kernel == 'poly':

        degree = 3

        y_estimate = np.zeros(X_new.shape[0])
        for i in range(0, X_new.shape[0]):
            y_estimate[i] = y_dual(X_new[i], X_train, t_train, alpha, b, kernel)

    return y_estimate


# --------------------------------------------------------------------------------

def objective_dual(X_train, t_train, alpha, kernel):
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
    value = 0

    if kernel == 'linear':

        len = X_train.shape[0]

        for i in range(len):
            for j in range(len):
                value += alpha[i] * alpha[j] * t_train[i] * t_train[j] * linear_kernel(X_train[i], X_train[j])

        value = 0.5 * value - np.sum(alpha)

    if kernel == 'polynomial':

        degree = 3

        len = X_train.shape[0]

        for i in range(len):
            for j in range(len):
                value += alpha[i] * alpha[j] * t_train[i] * t_train[j] * poly_kernel(X_train[i], X_train[j], degree)

        value = 0.5 * value - np.sum(alpha)

    return value


# --------------------------------------------------------------------------------

def SMO(X_train, t_train, alpha0, b0, kernel, C, max_iter):
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

    dual_values = []
    tol = 0.001

    if kernel == 'linear':
        # Compute parameters L, H according to the following rule
        # if ti != tj , set L = max(0, αj − αi), H = min(C, C + αj − αi)
        # and if ti == tj , set L = max(0, αi + αj − C), H = min(C, αi + αj )
        # iterate over all alpha_n, n = 1, ... , N, in random order
        passes = 0
        while passes < max_iter:
            num_changed_alphas = 0
            for n in range(len(alpha0)):
                # Compute Ei = y(xi) − ti
                E_i = y_dual(X_train[n], X_train, t_train, alpha0, b0, kernel) - t_train[n]

                # Check if KKT conditions are violated for alpha_n
                if (t_train[n] * E_i < -tol and alpha0[n] < C) or (t_train[n] * E_i > tol and alpha0[n] > 0):
                    # select alpha_m randomly from the remaining N − 1 elements
                    rand_j = np.random.randint(0, len(alpha0) - 1)
                    E_j = y_dual(X_train[rand_j], X_train, t_train, alpha0, b0, kernel) - t_train[rand_j]
                    alpha_i_old = alpha0[n]
                    alpha_j_old = alpha0[rand_j]

                    # Compute L and H according to the following rule
                    # if ti != tj , set L = max(0, αj − αi), H = min(C, C + αj − αi)
                    # and if ti == tj , set L = max(0, αi + αj − C), H = min(C, αi + αj )
                    if t_train[n] != t_train[rand_j]:
                        L = max(0, alpha0[rand_j] - alpha0[n])
                        H = min(C, C + alpha0[rand_j] - alpha0[n])
                    elif t_train[n] == t_train[rand_j]:
                        L = max(0, alpha0[n] + alpha0[rand_j] - C)
                        H = min(C, alpha0[n] + alpha0[rand_j])

                    if L == H:
                        continue

                    nyu = 2 * linear_kernel(X_train[n], X_train[rand_j]) - linear_kernel(X_train[n],
                                                                                         X_train[n]) - linear_kernel(
                        X_train[rand_j], X_train[rand_j])
                    if nyu >= 0:
                        continue

                    # Compute new value for alpha_j
                    alpha0[rand_j] = alpha0[rand_j] - (t_train[rand_j] * (E_i - E_j)) / nyu

                    # Clip new value for alpha_j
                    if alpha0[rand_j] > H:
                        alpha0[rand_j] = H
                    if L <= alpha0[rand_j] <= H:
                        alpha0[rand_j] = alpha0[rand_j]
                    elif alpha0[rand_j] < L:
                        alpha0[rand_j] = L

                    if abs(alpha0[rand_j] - alpha_j_old) < tol:
                        continue

                    # Compute new value for alpha_i
                    alpha0[n] = alpha0[0] + E_j * t_train[n] * (alpha_j_old - alpha0[rand_j])

                    # Clip new value for alpha_i
                    if alpha0[n] > H:
                        alpha0[n] = H
                    if L <= alpha0[n] <= H:
                        alpha0[n] = alpha0[n]
                    elif alpha0[n] < L:
                        alpha0[rand_j] = L

                    # Compute new value for b
                    b_1 = b0 - E_i - t_train[n] * (alpha0[n] - alpha_i_old) * linear_kernel(X_train[n], X_train[n]) - \
                          t_train[rand_j] * (alpha0[rand_j] - alpha_j_old) * linear_kernel(X_train[n], X_train[rand_j])
                    b_2 = b0 - E_j - t_train[n] * (alpha0[n] - alpha_i_old) * linear_kernel(X_train[n],
                                                                                            X_train[rand_j]) - t_train[
                              rand_j] * (alpha0[rand_j] - alpha_j_old) * linear_kernel(X_train[rand_j], X_train[rand_j])

                    if 0 < alpha0[n] < C:
                        b0 = b_1
                    if 0 < alpha0[rand_j] < C:
                        b0 = b_2
                    else:
                        b0 = (b_1 + b_2) / 2

                    num_changed_alphas += 1

                    objective = objective_dual(X_train, t_train, alpha0, kernel)
                    dual_values.append(objective)

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0



    if kernel == 'poly':

        degree = 3

        passes = 0
        while passes < max_iter:
            print('Smo')
            num_changed_alphas = 0
            for n in range(len(alpha0)):
                # Compute Ei = y(xi) − ti
                E_i = y_dual(X_train[n], X_train, t_train, alpha0, b0, kernel) - t_train[n]

                # Check if KKT conditions are violated for alpha_n
                if (t_train[n] * E_i < -tol and alpha0[n] < C) or (t_train[n] * E_i > tol and alpha0[n] > 0):
                    # select alpha_m randomly from the remaining N − 1 elements
                    rand_j = np.random.randint(0, len(alpha0) - 1)
                    E_j = y_dual(X_train[rand_j], X_train, t_train, alpha0, b0, kernel) - t_train[rand_j]
                    alpha_i_old = alpha0[n]
                    alpha_j_old = alpha0[rand_j]

                    # Compute L and H according to the following rule
                    # if ti != tj , set L = max(0, αj − αi), H = min(C, C + αj − αi)
                    # and if ti == tj , set L = max(0, αi + αj − C), H = min(C, αi + αj )
                    if t_train[n] != t_train[rand_j]:
                        L = max(0, alpha0[rand_j] - alpha0[n])
                        H = min(C, C + alpha0[rand_j] - alpha0[n])
                    elif t_train[n] == t_train[rand_j]:
                        L = max(0, alpha0[n] + alpha0[rand_j] - C)
                        H = min(C, alpha0[n] + alpha0[rand_j])
                    if L == H:
                        continue

                    nyu = 2 * poly_kernel(X_train[n], X_train[rand_j], degree) - poly_kernel(X_train[n],
                                                                                             X_train[n],
                                                                                             degree) - \
                          poly_kernel(X_train[rand_j], X_train[rand_j], degree)
                    if nyu >= 0:
                        continue

                    # Compute new value for alpha_j
                    alpha0[rand_j] = alpha0[rand_j] - (t_train[rand_j] * (E_i - E_j)) / nyu

                    # Clip new value for alpha_j
                    if alpha0[rand_j] > H:
                        alpha0[rand_j] = H
                    if L <= alpha0[rand_j] <= H:
                        alpha0[rand_j] = alpha0[rand_j]
                    elif alpha0[rand_j] < L:
                        alpha0[rand_j] = L


                    # Compute new value for alpha_i
                    alpha0[n] = alpha0[0] + E_j * t_train[n] * (alpha_j_old - alpha0[rand_j])
                    # Clip new value for alpha_i
                    if alpha0[n] > H:
                        alpha0[n] = H
                    if L <= alpha0[n] <= H:
                        alpha0[n] = alpha0[n]
                    elif alpha0[n] < L:
                        alpha0[rand_j] = L

                    # Compute new value for b
                    b_1 = b0 - E_i - t_train[n] * (alpha0[n] - alpha_i_old) * poly_kernel(X_train[n], X_train[n],
                                                                                          degree) - \
                          t_train[rand_j] * (alpha0[rand_j] - alpha_j_old) * poly_kernel(X_train[n], X_train[rand_j],
                                                                                         degree)
                    b_2 = b0 - E_j - t_train[n] * (alpha0[n] - alpha_i_old) * poly_kernel(X_train[n], X_train[rand_j],
                                                                                          degree) - \
                          t_train[rand_j] * (alpha0[rand_j] - alpha_j_old) * poly_kernel(X_train[rand_j],
                                                                                         X_train[rand_j], degree)

                    if 0 < alpha0[n] < C:
                        b0 = b_1
                    if 0 < alpha0[rand_j] < C:
                        b0 = b_2
                    else:
                        b0 = (b_1 + b_2) / 2
                    num_changed_alphas += 1

                    objective = objective_dual(X_train, t_train, alpha0, kernel)
                    dual_values.append(objective)

                    print('alpha0', alpha0, 'b0', b0, 'dual_values', dual_values)


            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    return alpha0, b0, dual_values


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
