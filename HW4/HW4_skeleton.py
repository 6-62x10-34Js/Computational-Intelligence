# Filename: HW4_skeleton.py
# Edited: June 2023

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets


# --------------------------------------------------------------------------------

# Assignment 4

def main():
    # ------------------------
    # (0) Get the input
    ## (a) load the modified iris data
    data, labels, feature_names = load_iris_data()

    ## (b) construct the datasets
    x_2dim = data[:, [0, 2]]
    x_4dim = data

    # TODO: implement PCA (Required for working on 4.1!)
    x_2dim_pca, var_exp = PCA(data, nr_dimensions=2, whitening=False)
    x_2dim_pca_white, var_exp_white = PCA(data, nr_dimensions=2, whitening=True)
    print(f'explained variance {var_exp}')
    print(f'explained variance whitened {var_exp_white}')

    ## (c) visually inspect the data with the provided function (see example below)
    plot_iris_data(x_2dim, labels, feature_names[0], feature_names[2], "Iris Dataset")

    # ------------------------
    # Scenario 1: Consider a 2-dim slice of the data and evaluate the EM and the k-Means algorithm. Corresponds to scenario 1 for both algorithms and involves 2.1 and 3.1!
    scenario = 1
    dim = 2
    nr_components = 3

    # TODO set parameters
    tol = 1e-9  # tolerance
    max_iter = 1000  # maximum iterations for GN
    nr_components = 3  # n number of components

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(dimension=dim, nr_components=nr_components, scenario=scenario)
    alpha, mean, cov, log_likelihood, labels_soft = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

    initial_centers = init_k_means(dim, nr_components, scenario, x_2dim)
    centers, cumulative_distance, labels = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)

    # TODO visualize your results
    plot_gmm(x_2dim, mean, cov, f'GMM with {nr_components} components')
    plot_log_likelihood(log_likelihood, f'Log Likelihood for GMM with {nr_components} components')
    plot_iris_data(x_2dim, labels_soft, feature_names[0], feature_names[2], "Iris Dataset Soft Labels")

    reassigned_labels = reassign_class_labels(labels)
    plot_iris_data(x_2dim, reassigned_labels[labels], feature_names[0], feature_names[2],
                   "Iris Dataset Reassigned Labels")


    plot_clusters(x_2dim, labels, centers, f'K-Means with {nr_components} clusters')
    plot_clusters(x_2dim, reassigned_labels[labels], centers, f'K-Means with {nr_components} clusters (Reassigned)')
    plot_cum_dist_over_iteration(cumulative_distance, f'K-Means cumulative distances with {nr_components} clusters 2D')


    # ------------------------
    # Scenario 2: Consider the full 4-dimensional data and evaluate the EM and the k-Means algorithm. Corresponds to scenario 2 for both algorithms and involves Sec.2.2 and Sec.3.2!
    scenario = 2
    dim = 4
    nr_components = 3

    # TODO set parameters
    tol = 1e-3  # tolerance
    max_iter = 1000  # maximum iterations for GN
    nr_components = 3  # n number of components

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    alpha, mean, cov, log_likelihood, labels_soft = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    initial_centers = init_k_means(dim, nr_components, scenario, x_4dim)
    centers, cumulative_distance, labels = k_means(x_4dim ,nr_components, initial_centers, max_iter, tol)


    # TODO: visualize your results by looking at the same slice as in 1)
    plot_iris_data(x_4dim, labels, feature_names[0], feature_names[2], "Iris Dataset (4D)")
    plot_log_likelihood(log_likelihood, f'Log Likelihood for GMM with {nr_components} components (4D)')
    plot_iris_data(x_4dim, labels_soft, feature_names[0], feature_names[2], "Iris Dataset Soft Labels (4D)")

    reassigned_labels = reassign_class_labels(labels)
    plot_iris_data(x_4dim, reassigned_labels[labels], feature_names[0], feature_names[2],
                   "Iris Dataset Reassigned Labels (4D)")
    # plot_gmm(x_4dim, mean, cov, f'GMM with {nr_components} components') # doesnt work, no energy to fix


    plot_clusters(x_4dim, labels, centers, f'K-Means with {nr_components} clusters (4D)')
    plot_clusters(x_4dim, reassigned_labels[labels], centers, f'K-Means with {nr_components} clusters (Reassigned) (4D)')
    plot_cum_dist_over_iteration(cumulative_distance, f'K-Means cumulative distances with {nr_components} clusters (4D)')

    # ------------------------
    # Scenario 3: Perform PCA to reduce the dimension from 4 to 2 while preserving most of the variance. Corresponds to scenario 3 for both algorithms and involves Sec.4.1!
    # Evaluate the EM and the k-Means algorithm on the transformed data.
    scenario = 3
    dim = 2
    nr_components = 3

    # TODO set parameters
    tol = 1e-3  # tolerance
    max_iter = 1000  # maximum iterations for GN
    nr_components = 3  # n number of components


    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(dim, nr_components, scenario)
    alpha, mean, cov, log_likelihood, labels_soft = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    initial_centers = init_k_means(dim, nr_components, scenario, x_2dim_pca)
    centers, cumulative_distance, labels = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)


    # TODO: visualize your results
    plot_iris_data(x_2dim_pca, labels, feature_names[0], feature_names[2], "PCA Reduced Iris Dataset (2D)")
    plot_gmm(x_2dim_pca, mean, cov, f'GMM with {nr_components} components PCA')
    plot_log_likelihood(log_likelihood, f'Log Likelihood for GMM with {nr_components} components PCA')
    plot_iris_data(x_2dim_pca, labels_soft, feature_names[0], feature_names[2], "EM Iris Dataset Soft Labels PCA")

    reassigned_labels = reassign_class_labels(labels)
    plot_iris_data(x_2dim_pca, reassigned_labels[labels], feature_names[0], feature_names[2],
                   "EM Iris Dataset Reassigned Labels PCA")


    plot_clusters(x_2dim_pca, labels, centers, f'K-Means with {nr_components} clusters')
    plot_clusters(x_2dim_pca, reassigned_labels[labels], centers, f'K-Means with {nr_components} clusters (Reassigned) PCA')
    plot_cum_dist_over_iteration(cumulative_distance, f'K-Means with cumulative distances {nr_components} clusters PCA')

    # ------------------------
    # pca whitening
    alpha_w, mean_w, cov_w, log_likelihood_w, labels_soft_w = EM(x_2dim_pca_white, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    initial_centers_w = init_k_means(dim, nr_components, scenario, x_2dim_pca_white)
    centers_w, cumulative_distance_w, labels_w = k_means(x_2dim_pca_white, nr_components, initial_centers_w, max_iter, tol)

    plot_iris_data(x_2dim_pca_white, labels_w, feature_names[0], feature_names[2], "PCA Whitened Iris Dataset")
    plot_gmm(x_2dim_pca_white, mean_w, cov_w, f'GMM with {nr_components} components PCA Whitened')
    plot_log_likelihood(log_likelihood_w, f'Log Likelihood for GMM with {nr_components} components PCA Whitened')
    plot_iris_data(x_2dim_pca_white, labels_soft_w, feature_names[0], feature_names[2], "EM Iris Dataset Soft Labels PCA Whitened")

    reassigned_labels_w = reassign_class_labels(labels_w)
    plot_iris_data(x_2dim_pca_white, reassigned_labels_w[labels_w], feature_names[0], feature_names[2], "EM Iris Dataset Reassigned Labels PCA Whitened")
    plot_clusters(x_2dim_pca_white, labels_w, centers_w, f'K-Means with {nr_components} clusters PCA Whitened')
    plot_clusters(x_2dim_pca_white, reassigned_labels_w[labels_w], centers_w, f'K-Means with {nr_components} clusters (Reassigned) PCA Whitened')
    plot_cum_dist_over_iteration(cumulative_distance_w, f'K-Means with cumulative distances {nr_components} clusters PCA Whitened')



    pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def init_EM(dimension, nr_components, scenario, X=None):
    """ initializes the EM algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""
    if scenario == 1 or scenario == 2 or scenario == 3:
        # init random
        # uniform init weights
        alpha_0 = np.array([1 / nr_components] * nr_components)
        mean_0 = np.random.rand(dimension, nr_components)
        cov_0 = np.zeros((dimension, dimension, nr_components))
        for i in range(nr_components):
            cov_0[:, :, i] = np.eye(dimension)
    else:
        pass

    assert mean_0.shape == (dimension, nr_components)
    assert cov_0.shape == (dimension, dimension, nr_components)

    return alpha_0, mean_0, cov_0


# --------------------------------------------------------------------------------

def EM(X, K, alpha_0, mean_0, cov_0, max_iter, tol):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension
    D = X.shape[1]
    N = X.shape[0]
    mean_0 = mean_0
    assert D == mean_0.shape[0]
    # TODO: iteratively compute the posterior and update the parameters

    log_likelihood = []
    labels = []

    for i in range(max_iter):

        responsibilies = np.zeros((N, K))
        likelihood = np.zeros((N, K))

        for k in range(K):
            likelihood[:, k] = likelihood_multivariate_normal(X, mean_0[:, k], cov_0[:, :, k])
        denominator = np.sum((likelihood * alpha_0), axis=1)

        for k in range(K):
            responsibilies[:, k] = (likelihood[:, k] * alpha_0[k]) / denominator

        N_k = np.sum(responsibilies, axis=0)
        alpha = N_k / N
        mean = np.dot(X.T, responsibilies) / N_k
        cov = np.zeros((D, D, K))

        for k in range(K):
            for n in range(N):
                cov[:, :, k] += responsibilies[n, k] * np.outer(X[n, :] - mean[:, k], X[n, :] - mean[:, k])
            cov[:, :, k] /= N_k[k]

        log_likelihood.append(np.sum(np.log(denominator)))

        if i > 0 and np.abs(log_likelihood[i] - log_likelihood[i - 1]) < tol:
            break
        alpha_0 = alpha
        mean_0 = mean
        cov_0 = cov

    # classify the data
    labels = np.argmax(responsibilies, axis=1)

    print(f'EM Algorithm converged after {i} iterations')

    return alpha, mean, cov, log_likelihood, labels


# --------------------------------------------------------------------------------

def init_k_means(dimension, nr_clusters, scenario, X=None):
    """ initializes the k_means algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    # TODO: chosse suitable inital values for each scenario
    if scenario == 1:
        # init random
        initial_centers = np.random.rand(dimension, nr_clusters)
        if X is not None:
            print('X is not None')
            # init k-means++
            dimension = X.shape[1]
            centers = np.empty((dimension, nr_clusters))

            #Choose the first center uniformly at random
            centers[:, 0] = X[np.random.choice(X.shape[0])]
            # 2 For each data point compute its distance from the nearest, previously chosen centroid.
            # 3 Choose one new data point at random as a new center, using a weighted probability distribution
            # Steps 2 and 3: Choose subsequent centers until k centroids have been sampled
            for k in range(1, nr_clusters):
                distances = np.zeros(X.shape[0])
                for j in range(k):
                    distances += np.linalg.norm(X - centers[:, j], axis=1) ** 2
                probabilities = distances / np.sum(distances)
                index = np.random.choice(X.shape[0], p=probabilities)
                centers[:, k] = X[index]

            return centers


        print(f'initial_centers {initial_centers}')

    elif scenario == 2:
        # init k-means++
        initial_centers = np.zeros((dimension, nr_clusters))
        initial_centers[:, 0] = X[np.random.randint(0, X.shape[0]), :]
        for i in range(1, nr_clusters):
            distances = np.zeros((X.shape[0], i))
            for j in range(i):
                distances[:, j] = np.linalg.norm(X - initial_centers[:, j], axis=1)
            min_distance = np.min(distances, axis=1)
            initial_centers[:, i] = X[np.argmax(min_distance), :]

    elif scenario == 3:
        initial_centers = np.zeros((dimension, nr_clusters))
        initial_centers[:, 0] = X[np.random.randint(0, X.shape[0]), :]
        for i in range(1, nr_clusters):
            distances = np.zeros((X.shape[0], i))
            for j in range(i):
                distances[:, j] = np.linalg.norm(X - initial_centers[:, j], axis=1)
            min_distance = np.min(distances, axis=1)
            initial_centers[:, i] = X[np.argmax(min_distance), :]

    assert initial_centers.shape == (dimension, nr_clusters)
    return initial_centers


# --------------------------------------------------------------------------------

def k_means(X, K, centers_0, max_iter, tol):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    assert D == centers_0.shape[0]
    # TODO: iteratively update the cluster centers

    cumulative_distance = []

    centers = centers_0.copy()

    for i in range(max_iter):
        distances = np.zeros((X.shape[0], K))
        for k in range(K):
            distances[:, k] = np.linalg.norm(X - centers_0[:, k], axis=1)

        labels = np.argmin(distances, axis=1)

        cumulative_distance.append(np.sum(np.min(distances, axis=1)))
        for k in range(K):
            # check if there are samples assigned to the cluster to avoid division by zero
            if np.sum(labels == k) > 0:
                centers[:, k] = np.mean(X[labels == k, :], axis=0)
        if i > 0 and np.abs(cumulative_distance[i] - cumulative_distance[i - 1]) < tol:
            break
        centers_0 = centers

    print(f'k-Means Algorithm converged after {i} iterations')
    # TODO: classify all samples after convergence

    return centers, cumulative_distance, labels



# --------------------------------------------------------------------------------

def PCA(data, nr_dimensions, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening

    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components
    # center the data
    data = data - np.mean(data, axis=0)
    # compute the covariance matrix
    cov = np.cov(data, rowvar=False)
    # compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # check if the decomposition is correct (A = VDV^T)



    # sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # transform the data
    transformed_data = np.dot(data, eigenvectors[:, :dim])
    # compute the variance explained
    variance_explained = np.sum(eigenvalues[:dim]) / np.sum(eigenvalues)

    if whitening:

        transformed_data = transformed_data / np.sqrt(eigenvalues[:dim])


    return np.array(transformed_data), variance_explained


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def load_iris_data():
    """ loads and modifies the iris data-set
    Input:
    Returns:
        X... samples, 150x4
        Y... labels, 150x1
        feature_names... name of the data columns"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100, 2] = iris.data[50:100, 2] - 0.25
    Y = iris.target
    return X, Y, iris.feature_names


# --------------------------------------------------------------------------------
def plot_clusters(data, labels, centers, title):
    """"
    In two separate plots, compare the obtained
    hard classification result to the labeled data similar as we did it for EM. The way to plot
    the clusters, however, is different now: in the scatter plot, visualize the center of each class
    and plot all points assigned to one and the same cluster in the same color
    """
    # plot first label
    plt.figure()
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1], label='Iris-Setosa')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Iris-Versicolor')
    plt.scatter(data[labels == 2, 0], data[labels == 2, 1], label='Iris-Virgnica')

    plt.scatter(centers[0, 0], centers[1, 0], marker='x', s=100)
    plt.scatter(centers[0, 1], centers[1, 1], marker='x', s=100)
    plt.scatter(centers[0, 2], centers[1, 2], marker='x', s=100)
    plt.annotate('Center 1', (centers[0, 0], centers[1, 0]))
    plt.annotate('Center 2', (centers[0, 1], centers[1, 1]))
    plt.annotate('Center 3', (centers[0, 2], centers[1, 2]))

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend()
    plt.title(title)
    plt.show()

def plot_iris_data(data, labels, x_axis, y_axis, title):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1
        x_axis... label for the x_axis
        y_axis... label for the y_axis
        title...  title of the plot"""

    plt.scatter(data[labels == 0, 0], data[labels == 0, 1], label='Iris-Setosa')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Iris-Versicolor')
    plt.scatter(data[labels == 2, 0], data[labels == 2, 1], label='Iris-Virgnica')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xlim([np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1])
    plt.ylim([np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1])
    plt.title(title)
    plt.legend()

    plt.show()


# --------------------------------------------------------------------------------

def likelihood_multivariate_normal(X, mean, cov, log=False):
    """Returns the likelihood of X for multivariate (d-dimensional) Gaussian
   specified with mu and cov.

   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

    dist = multivariate_normal(mean, cov)
    if log is False:
        P = dist.pdf(X)
    elif log is True:
        P = dist.logpdf(X)
    return P


# --------------------------------------------------------------------------------
def plot_gmm(data, means, covs, title):
    """ plots the data and the estimated gmm
    Input:
        data... samples, nr_samples x 2
        means... list of means, nr_clusters x 2
        covs... list of covariances, nr_clusters x 2 x 2
        title... title of the plot"""

    plt.figure(figsize=(10, 8))

    plt.scatter(data[:, 0], data[:, 1], label='data')


    for i in range(means.shape[1]):
        plot_gauss_contour(means[:, i], covs[:, :, i], np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]),
                           np.max(data[:, 1]), 100, title=f'GMM component {i + 1}')
    plt.title(title)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend()

    plt.show()


def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, nr_points, title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters

    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

    # npts = 100
    delta_x = float(xmax - xmin) / float(nr_points)
    delta_y = float(ymax - ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)

    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = multivariate_normal(mean= mu, cov= cov).pdf(pos)
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    return

def plot_log_likelihood(log_likelihood, title):
    """ plots the log likelihood
    Input:
        log_likelihood... array of log-likelihood values
        title... title of the plot"""
    plt.figure(figsize=(10, 8))
    plt.plot(log_likelihood)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')

    plt.show()

def plot_cum_dist_over_iteration(cum_dist_array, title):
    """ plots the cumulative distance over iteration
    Input:
        cum_dist_array... array of cumulative distances
        title... title of the plot"""
    plt.figure(figsize=(10, 8))
    plt.plot(cum_dist_array)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative distance')
    plt.show()


# --------------------------------------------------------------------------------

def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.

    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)

    y = np.zeros(N)
    cumulativePM = np.cumsum(PM)  # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N)  # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N)  # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]:  # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y)  # permutation of all samples


# --------------------------------------------------------------------------------

def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable.
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50] == 0), np.sum(labels[0:50] == 1), np.sum(labels[0:50] == 2)],
                                  [np.sum(labels[50:100] == 0), np.sum(labels[50:100] == 1),
                                   np.sum(labels[50:100] == 2)],
                                  [np.sum(labels[100:150] == 0), np.sum(labels[100:150] == 1),
                                   np.sum(labels[100:150] == 2)]])
    new_labels = np.array([np.argmax(class_assignments[:, 0]),
                           np.argmax(class_assignments[:, 1]),
                           np.argmax(class_assignments[:, 2])])
    return new_labels


# --------------------------------------------------------------------------------

def sanity_checks():
    # likelihood_multivariate_normal
    mu = [0.0, 0.0]
    cov = [[1, 0.2], [0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    plot_gauss_contour(mu, cov, -2, 2, -2, 2, 100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)

    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))

    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                                        0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                                        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels = np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd == 0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd == 1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd == 2] = new_labels[2]


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    sanity_checks()
    main()
