# Filename: HW1_MLE_skeleton.py
# Author: Harald Leisenberger
# Edited: March, 2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as stats
import timeit



# --------------------------------------------------------------------------------
# Assignment 1 - Section 2
# --------------------------------------------------------------------------------
def draw_histogram(data, dist_type):
    """ Draws a histogram of the data array

    Input:  data ... an array of 1-dimensional data points
            dist_type ... title of the histogram

    Output: None
    """
    plt.figure(1, figsize=(9, 10), dpi=80)
    plt.title('Histogram of the datasets and the estimated distributions')
    plt.subplot(311, title=dist_type[0])
    plt.hist(data[0], bins=40, density=True)
    plt.subplot(312, title=dist_type[1])
    plt.hist(data[1], bins=40, density=True)
    plt.subplot(313, title=dist_type[2])
    plt.hist(data[2], bins=40, density=True)
    plt.savefig('plots/Histograms.png')
    plt.show()


def main():
    # !!! All undefined functions should be implemented in the section 'Helper Functions' !!!

    data = np.array([np.loadtxt('HW1_MLE_1.data'), np.loadtxt('HW1_MLE_2.data'), np.loadtxt('HW1_MLE_3.data')])
    # Load the three data arrays (100 x 1 - arrays)

    # 2.1 Maximum Likelihood Model Estimation (MLE)
    # ---------------------------------------------

    # Estimate the true model parameters via MLE
    ML_param, dist_type = ML_estimation(data)

    ML_exp_param = ML_param[2]
    ML_gauss_param = ML_param[:2]

    # Make histogram plots of the data arrays
    # Make histogram plots of the data arrays as subplots of a single figure
    #
    draw_histogram(data, dist_type)

    # Plot the estimated densities together with the data points
    plot_scatter_and_density(data, ML_param, dist_type)

    # 2.2 Evaluation and Visualization of the Likelihood Function
    # -----------------------------------------------------------

    # 3D plots of the joint likelihood functions of the Gaussian distributed data
    # TODO: specify parameters and insert the correct numbers for i and j. Choose reasonable grid boundaries / resolution.
    # plot_likelihood_Gauss(datai,mu_min,mu_max,sigma_sq_min,sigma_sq_max,resolution_mu,resolution_sigma_sq)
    plot_likelihood_Gauss(data, dist_type, ML_param)
    #
    # # Numerical MLE for the Gaussian distributed data
    # # TODO: specify parameters and insert the correct numbers for i and j. Choose reasonable grid boundaries / resolution.
    ML_num_i = ML_numerical_Gauss(data, dist_type)
    # ML_num_j = ML_numerical_Gauss(dataj,mu_min,mu_max,sigma_sq_min,sigma_sq_max,resolution_mu,resolution_sigma_sq)

    #compare numerical and analytical MLE

    # # TODO: specify parameters and insert the correct number for k. Choose reasonable grid boundaries / resolution.

    #
    # # Numerical MLE for the exponential distributed data
    # # TODO: specify parameters and insert the correct number for k. Choose reasonable grid boundaries / resolution.

    ML_num_k = ML_numerical_Exp(data, dist_type)

    ML_ana_k = plot_likelihood_Exp(data, dist_type)
    # # 2.3 Bayesian Model Estimation
    # # -----------------------------
    #
    # # Compute the posterior distributions for the means of the Gaussian data, given the prior distribution N(mu|0.5,1)
    # # TODO: specify parameters and insert the correct numbers for i and j.
    # posterior_param_i = posterior_mean_Gauss(datai,mu_prior,mu_ML,sigma_sq_prior,sigma_sq_true)
    # posterior_param_j = posterior_mean_Gauss(dataj,mu_prior,mu_ML,sigma_sq_prior,sigma_sq_true)
    #
    # # Plot the prior distribution and the two posterior distributions of the means for the Gaussian distributed data arrays
    # # TODO

    pass


def calculate_inverse_log_likelihood_ratio(data, ML_num_i, dist_type, ML_param):
    # Compute the inverse likelihood ratio for the Gaussian distributed data
    for i in range(len(data)):
        if dist_type[i] == "Gaussian":
            mu = ML_param[i][0]
            sigma_sq = ML_param[i][1]
            mu_num = ML_num_i[i][0]
            sigma_sq_num = ML_num_i[i][1]


# --------------------------------------------------------------------------------
# Helper Functions (to be implemented!)
# --------------------------------------------------------------------------------
def plot_scatter_and_density(data, ML_param, dist_type):
    """ Plots the estimated densities together with the histogram of data points

    Input:  data ... an array of 1-dimensional data points
            ML_param ... the values of the maximum likelihood estimators for all parameters
                         of the corresponding distribution (i.e., 1-D Gaussian or 1-D exponential)

    Output: None
    """

    # Plot the estimated density function together with the data points
    fig, axes = plt.subplots(len(data), 1, figsize=(9, 10), dpi=80)

    for i, ax in enumerate(axes):
        if dist_type[i] == "Exponential Distribution":
            lambd = ML_param[i][0]
            x = np.linspace(0, np.max(data[i]), 100)
            pdf = lambd * np.exp(-lambd * x)

            # Plot the density function and the data points
            ax.hist(data[i], density=True, alpha=0.5, bins=40, label='Data points')
            ax.scatter(x, pdf, label=f'Estimated PDF with:' + '\n' + f'$\lambda$ = {round(lambd, 2)})')
            ax.set_xlabel('x')
            ax.set_ylabel('Density')
            ax.set_title('Exponential distribution')
            ax.legend()

        elif dist_type[i] == "Gaussian":
            mu = ML_param[i][0]
            sigma_sq = ML_param[i][1]
            x = np.linspace(np.min(data[i]), np.max(data[i]), 100)
            pdf = 1 / np.sqrt(2 * np.pi * sigma_sq) * np.exp(-0.5 * (x - mu) ** 2 / sigma_sq)
            # Plot the density function and the data points
            ax.hist(data[i], density=True, alpha=0.5, bins=40, label='Data points')
            ax.scatter(x, pdf,
                       label='Estimated PDF with:' + '\n' + f'$\mu$={round(mu, 2)} and $\sigma^2$={round(sigma_sq, 2)}')
            ax.set_title('Gaussian distribution')
            ax.set_xlabel('x')
            ax.set_ylabel('Density')
            ax.legend()
    plt.savefig('plots/dist_and_pdf_hist.png')
    plt.tight_layout()
    plt.show()


def ML_estimation(data):
    """ estimates the maximum likelihood parameters for a given data sample.
    
    Input:  data ... an array of arrays of 1-dimensional data points
    
    Output: ML_param ... the values of the maximum likelihood estimators for all parameters
                         of the corresponding distribution (i.e., 1-D Gaussian or 1-D exponential)
    """

    # Check whether the data is Gaussian or exponential distributed (Kolmogorov-Smirnov test)
    # (2) Estimate the parameters of the distribution via MLE (mean and variance for a Gaussian,
    #          lambda for an exponential distribution)

    # choose very low hypotheses value since the data is very close to the chosen distribution to compare to
    hypotheses = 1e-6
    ml_params = np.empty((len(data), 2))
    dist_type = np.empty(len(data), dtype=object)

    print('Computing analytical MLE for Gaussian and Exponential distributions...')

    for i in range(len(data)):
        ks_test = stats.kstest(data[i], stats.expon.pdf(np.arange(0, 4, 0.1), loc=0, scale=2))
        is_exponential = ks_test[1] > hypotheses
        length = data[i].shape[0]
        dist_type[i] = "Exponential Distribution" if is_exponential else "Gaussian"
        if is_exponential:
            start_time_1 = timeit.default_timer()
            mean = sum(data[i]) / length
            ml_params[i] = np.array([1/mean])
            elapsed_1 = timeit.default_timer() - start_time_1

            print(f'Exponential MLE for data {i + 1}: lambda = {ml_params[i][0]}')
            print(f'Elapsed time exponential: {elapsed_1:.8f} seconds')
            print('----------------------------------------')

        else:
            start_time_2 = timeit.default_timer()
            mean = 1 / length * np.sum(data[i])
            variance = 1 / length * np.sum((data[i] - mean) ** 2)
            ml_params[i] = np.array([mean,variance])
            elapsed_2 = timeit.default_timer() - start_time_2

            print(f'Gaussian MLE for data {i + 1}: mu = {mean}, sigma^2 = {variance}')
            print(f'Elapsed time Gaussian: {elapsed_2:.8f} seconds')
            print('----------------------------------------')


    return ml_params, dist_type


# --------------------------------------------------------------------------------
def calcualate_gauss_likelihood(data, mu, sigma_sq):

    likelihood = (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * ((data - mu) ** 2 / sigma_sq))
    return likelihood

def caluculate_exponential_likelihood(data, lambd):
    likelihood = np.empty(len(lambd))
    for i in range(len(lambd)):
        likelihood[i] = np.prod(lambd[i] * np.exp(-lambd[i] * data))
    return likelihood


def plot_likelihood_Gauss(data, dist_type, ML_param):
    """ Plots the joint likelihood function for mu and sigma^2 for a given 1-D Gaussian data sample on
        a predefined grid.
    
    Input:  data ... an array of 1-dimensional Gaussian distributed data points
            mu_min ... lower boundary of the grid on the mu-axis
            mu_max ... upper boundary of the grid on the mu-axis
            sigma_sq_min ... lower boundary of the grid on the sigma^2-axis
            sigma_sq_max ... upper boundary of the grid on the sigma^2-axis
            resolution_mu ... interval length between discretized points on the mu-axis
            ML_param ... the values of the maximum likelihood estimators for all parameters
            
            resolution_sigma_sq ... interval length between discretized points on the sigma^2-axis
    Output: ---
    """

    # TODO Plot the joint Gaussian likelihood w.r.t. mu and sigma^2 on a discretized 2-D grid
    for i in range(len(data)):
        if dist_type[i] != "Gaussian":
            print(f'Data {i + 1} is not Gaussian distributed')
        else:
            # Dynamically determine the range of mu and sigma_sq values to plot
            mu_min = np.min(data[i]) - 1  # subtract 1 to ensure some padding
            mu_max = np.max(data[i]) + 1  # add 1 to ensure some padding
            sigma_sq_min = 1e-6  # variance cannot be negative
            sigma_sq_max = np.var(data[i]) * 5  # multiply by 5 to ensure some padding

            # Create a 2D grid of mu and sigma_sq values
            mu_range = np.linspace(mu_min, mu_max, 100)
            sigma_sq_range = np.linspace(sigma_sq_min, sigma_sq_max, 100)
            mu_grid, sigma_sq_grid = np.meshgrid(mu_range, sigma_sq_range)

            # Evaluate the likelihood function for each pair of mu and sigma_sq values
            likelihood_grid = calcualate_gauss_likelihood(data[i], mu_grid, sigma_sq_grid)

            # Evaluate the likelihood function for each pair of mu and sigma_sq values contained in ML_param
            mu_ML = ML_param[i][0]
            sigma_sq_ML = ML_param[i][1]

            # Plot the likelihood function
            fig = plt.figure(figsize=(8, 8), dpi=80)
            ax = fig.add_subplot(111)
            if i == 0:
                # show zoomed in plot for first data set
                im = ax.imshow(likelihood_grid, cmap='viridis', extent=[mu_ML - 0.5, mu_ML + 0.5,sigma_sq_ML - 0.5, sigma_sq_ML + 0.5])

                ax.set_xlim([mu_ML - 0.5, mu_ML + 0.5])
                ax.set_ylim([sigma_sq_ML - 0.5, sigma_sq_ML + 0.5])
            else:
                im = ax.imshow(likelihood_grid, cmap='viridis', extent=[mu_min, mu_max, sigma_sq_min, sigma_sq_max])
            ax.set_title(f'Gaussian likelihood function for data {i + 1}')
            fig.colorbar(im, orientation='vertical')
            ax.set_xlabel(f'$\mu$')
            ax.set_ylabel(f'$\sigma^2$')
            ax.scatter(mu_ML, sigma_sq_ML, c='r', marker='x', label='Estimated parameters:' +'\n'+ '$\mu_{ml}$' + f'= {mu_ML:.2f}' +'\n'+ '$\sigma_{ml}$' + f'= {sigma_sq_ML:.2f}')
            ax.legend(loc = 'lower left')
            plt.savefig(f'Gaussian_likelihood_data_{i + 1}_MLE.png')

            plt.show()


            # Create a 3D plot of the likelihood function
            fig = plt.figure(figsize=(8, 8), dpi=80)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(mu_grid, sigma_sq_grid, likelihood_grid, cmap='viridis', edgecolor='none')
            ax.set_title(f'Gaussian likelihood function for data {i + 1}')
            ax.set_xlabel(f'$\mu$')
            ax.set_ylabel(f'$\sigma^2$')
            ax.set_zlabel('likelihood')
            plt.savefig(f'Gaussian_likelihood_data_{i + 1}.png')

            plt.show()
    return


# --------------------------------------------------------------------------------

def ML_numerical_Gauss(data, dist_type):
    """ Numerically computes the pairs (µi∗,(σ2i)∗) that maximize the Gaussian likelihood function
    for each data set
    Input:
        data: list of 1D arrays of data
        dist_type: list of strings containing the distribution type for each data set
    Output:
        max_likelihood_params: list of tuples containing the (µi∗,(σ2i)∗) that maximize the likelihood
    """

    ML_num_Gauss = np.empty((len(data), 2))
    print('Computing the maximum likelihood parameters numerically for the Gaussian distribution ...')

    for i in range(len(data)):
        if dist_type[i] != "Gaussian":
            print(f"Data {i + 1} does not follow a Gaussian distribution!")
        else:
            start_time = timeit.default_timer()

            # use newton-raphson method to find the maximum likelihood parameters
            max_iter = 1000
            mu = np.mean(data[i])
            sigma_sq = np.var(data[i])
            tol = 1e-6
            for j in range(max_iter):
                # calculate the gradient of the likelihood function
                grad_mu = np.sum((data[i] - mu) / sigma_sq)
                grad_sigma_sq = np.sum((data[i] - mu) ** 2 / sigma_sq ** 2 - 1 / sigma_sq)

                # calculate the hessian of the likelihood function
                hess_mu = -np.sum(1 / sigma_sq)
                hess_sigma_sq = -np.sum(2 * (data[i] - mu) ** 2 / sigma_sq ** 3)
                hess_mu_sigma_sq = -np.sum(2 * (data[i] - mu) / sigma_sq ** 2)

                # calculate the inverse of the hessian
                det_hess = hess_mu * hess_sigma_sq - hess_mu_sigma_sq ** 2
                inv_hess = np.array([[hess_sigma_sq, -hess_mu_sigma_sq], [-hess_mu_sigma_sq, hess_mu]]) / det_hess

                # calculate the update
                update = -np.dot(inv_hess, np.array([grad_mu, grad_sigma_sq]))

                # update the parameters
                mu += update[0]
                sigma_sq += update[1]

                # check if the update is smaller than the tolerance
                if np.linalg.norm(update) < tol:

                    break

            print(f"Numerical Maximum likelihood parameters for data {i + 1}: mu = {mu:.4f}, sigma_sq = {sigma_sq:.4f}")
            end_time = timeit.default_timer()
            print(
                f'Time needed for numerical maximum likelihood estimation: {end_time - start_time:.8f} seconds')

    return ML_num_Gauss


# --------------------------------------------------------------------------------

def plot_likelihood_Exp(data, dist_type):
    """ Plots the joint likelihood function for lambda for a given 1-D exponentially distributed data sample on
        a predefined grid.
    
    Input:  data ... an array of 1-dimensional Gaussian distributed data points
            dist_type ... an array of strings specifying the distribution type of each data set

            
    Output: ---
    """

    # TODO Plot the joint Exponential likelihood w.r.t. lambda on a discretized 1-D grid the exponential data or varying lambda values



    lambda_min = 0
    lambda_max = 10
    resolution_lambda = 0.1

    lambda_range = np.arange(lambda_min, lambda_max, resolution_lambda)



    for i in range(len(data)):
        if dist_type[i] != "Exponential Distribution":
            print(f"Data {i + 1} does not follow an Exponential distribution!")
        else:
            lambda_grid = np.linspace(lambda_min, lambda_max, 100)
            likelihood_grid = caluculate_exponential_likelihood(data[i], lambda_grid)

            fig = plt.figure(figsize=(8, 8), dpi=80)
            ax = fig.add_subplot(111)
            ax.scatter(lambda_grid, likelihood_grid)
            ax.set_title(f'Exponential likelihood function for data {i + 1}')
            ax.set_xlabel(f'$\lambda$')
            ax.set_ylabel('likelihood')
            plt.show()



    return


# --------------------------------------------------------------------------------

def ML_numerical_Exp(data, dist_type):
    """ numerically computes the MLEs for lambda for a given 1-D exponentially distributed data sample on
        a predefined grid.
    
    Input:  data ... an array of 1-dimensional exponentially distributed data points
            dist_type ... an array of strings specifying the distribution type of each data set

    Output: ML_num_Exp ... the numerical maximum likelihood estimators for mu and sigma^2 for a Gaussian data
                       array
    """
    # TODO Compute the values of the joint exponential likelihood w.r.t. lambda and the data on a discretized 1-D grid
    #      and take the maximizing argument lambda* as the numerical MLE.


    print("Computing numerical MLEs for lambda for each data set...")

    print(dist_type)
    ML_num_Exp = np.zeros([len(data), 1])

    for i in range(len(data)):
        if dist_type[i] != "Exponential Distribution":
            print(f"Data {i + 1} does not follow an Exponential distribution!")
        else:
            #use newton-raphson method to find the maximum likelihood parameters
            max_iter = 1000
            lambda_ = np.mean(data[i])
            tol = 1e-6
            start_time_4 = timeit.default_timer()
            for j in range(max_iter):
                #calculate the gradient of the likelihood function
                grad_lambda = np.sum(data[i] / lambda_ - 1)

                #calculate the hessian of the likelihood function
                hess_lambda = -np.sum(data[i] / lambda_ ** 2)

                #calculate the inverse of the hessian
                inv_hess = 1 / hess_lambda

                #calculate the update
                update = -grad_lambda * inv_hess

                #update the parameters
                lambda_ += update

                #check if the update is smaller than the tolerance
                if np.abs(update) < tol:
                    break

            print(f'Numeric likelihood function for data {i + 1}: lambda = {lambda_}')
            ML_num_Exp[i] = lambda_
            end_time_4 = timeit.default_timer()
            print(f'Time needed for numerical exponential maximum likelihood estimation: {end_time_4 - start_time_4:.8f} seconds')

    return ML_num_Exp

def plot_and_compare_likelihood_functions_different_lambda(data, dist_type, ML_num_Exp, ML_analytical_Exp):

    lambda_min = 0
    lambda_max = np.max(data) + 1
    resolution_lambda = 100

    lambda_range = np.linspace(lambda_min, lambda_max, resolution_lambda)
    lambda_grid = np.meshgrid(lambda_range)

    for i in range(len(data)):
        if dist_type[i] != "Exponential Distribution":
            print(f"Data {i + 1} does not follow an Exponential distribution!")
        else:
            likelihood_grid = caluculate_exponential_likelihood(data[i], lambda_grid)

            fig = plt.figure(figsize=(8, 8), dpi=80)
            ax = fig.add_subplot(111)
            ax.plot(lambda_grid, likelihood_grid, label='likelihood')
            ax.axvline(ML_num_Exp[i], color='r', linestyle='--', label='numerical MLE')
            ax.axvline(ML_analytical_Exp[i], color='g', linestyle='--', label='analytical MLE')
            ax.set_title(f'Exponential likelihood function for data {i + 1}')
            ax.set_xlabel(f'$\lambda$')
            ax.set_ylabel('likelihood')
            ax.legend()
            plt.show()
    return

# --------------------------------------------------------------------------------

def posterior_mean_Gauss(data, mu_prior, mu_ML, sigma_sq_prior, sigma_sq_true):
    """ computes the parameters of the posterior distribution for the mean with a Gaussian prior and a Gaussian likelihood
    
    Input:  data ... an array of 1-dimensional Gaussian distributed data points
            mu_prior ... mean of the prior distribution of mu
            mu_ML ... maximum likelihood estimator of mu w.r.t. a certain data array
            sigma_sq_prior ... variance of the prior distribution of mu
            sigma_sq_true ... true variance of the distribution that underlies a certain data array   
            
    Output: posterior_param_Gauss ... the parameters of a posterior Gaussian distribution for the mean
    """

    posterior_param_Gauss = np.zeros([2, 1])

    # TODO Compute the parameters of the posterior distribution for the mean by making use of formular (4) and (5)
    #      on the assignment sheet for a given data array.

    return posterior_param_Gauss

def plot_known_exponential_function_over_data(data, dist_type, ML_analytical_Exp):
    """ plots the known exponential distribution over the data

    Input:  data ... an array of 1-dimensional exponentially distributed data points
            dist_type ... an array of strings specifying the distribution type of each data set
            ML_analytical_Exp ... the analytical maximum likelihood estimators for lambda for an exponential data
                                  array

    Output: ---
    """

    # TODO Plot the known exponential distribution over the data

    return
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
