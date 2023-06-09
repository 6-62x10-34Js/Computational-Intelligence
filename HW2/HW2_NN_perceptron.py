import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import mean_squared_error

"""
Assignment 2: Neural networks
Part 3.3: Perceptron

Implementation of a perceptron.
"""


class Perceptron:
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.n_missclassified = None

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "x and y should have the same number of rows"
        self._fit(x_train, y_train)
        assert self.w.shape == (x_train.shape[1], 1)
        return self

    def predict(self, x):
        assert x.shape[1] == self.w.shape[0]
        y_predictions = self._predict(x)
        y_predictions = np.array(y_predictions)
        assert y_predictions.shape[0] == x.shape[0], "Predictions should have the same number of rows as the input x"
        assert np.bitwise_or(y_predictions == 0, y_predictions == 1).all(), "predictions have to be 0 or 1"
        return y_predictions

    def activation(self, z):
        # return 1 if z >= 0 else 0
        return np.heaviside(z, 0)

    def _fit(self, x_train, y_train):
        n_features = x_train.shape[1]
        self.n_missclassified = 0

        self.w = np.zeros((n_features, 1))
        for i in range(self.max_iter):
            for x_i, y_i in zip(x_train, y_train):
                y_pred = self._predict(x_i)
                self.w += self.learning_rate * (y_i - y_pred) * x_i.reshape(-1, 1)
                if y_i != y_pred:
                    self.n_missclassified += 1

        return

    def _predict(self, x):
        z = np.dot(x, self.w)
        return self.activation(z)

    def score(self, x, y):  # r^2
        y_pred = self.predict(x)
        return 1 - mean_squared_error(y, y_pred) / np.var(y)



def load_data():
    x, y = make_blobs(n_features=2, centers=2, random_state=3)
    print("x shape is:", x.shape)
    print("y shape is:", y.shape)
    print(y)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def load_non_linearly_separable_data():
    """
    Generates non-linearly separable data and returns the samples and class labels
    :return:
    """
    x, y = make_gaussian_quantiles(n_features=2, n_classes=2, random_state=1)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def plot_data(x, y):
    plt.figure()
    plt.title("Two linearly-separable classes", fontsize='small')
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    # plt.show()


def plot_decision_boundary(perceptron, x, y, lr, n_i, train_e, test_e, source):
    dim1_max, dim1_min = np.max(x[:, 0]), np.min(x[:, 0])
    dim2_max, dim2_min = np.max(x[:, 1]), np.min(x[:, 1])
    dim1_vals, dim2_vals = np.meshgrid(np.arange(dim1_min, dim1_max, 0.1),
                                       np.arange(dim2_min, dim2_max, 0.1))
    y_vals = perceptron.predict(np.c_[dim1_vals.ravel(), dim2_vals.ravel()])
    y_vals = y_vals.reshape(dim1_vals.shape)

    plt.figure()
    plt.title(f"Two linearly-separable classes with decision boundary by {source}" + '\n' + f'learning rate: {lr}, number of iterations: {n_i}' + '\n' + f'And a Train Loss of: {train_e} and a Test Loss of: {test_e}', fontsize='small')
    plt.contourf(dim1_vals, dim2_vals, y_vals, alpha=0.4)
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    plt.savefig(f"linearly_separable_data_lr{lr}_n{n_i}_{source}.png")




def main():
    x, y = load_data()
    #x, y = load_non_linearly_separable_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    learning_rate = 0.001
    n_iter = 200

    # Perceptron from sklearn
    perceptron = SkPerceptron(alpha=learning_rate, max_iter=n_iter, fit_intercept=False)
    perceptron.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))
    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    title = "Sklearn Perceptron with learning rate = " + str(learning_rate) + " and max iterations = " + str(n_iter)
    plot_decision_boundary(perceptron, x, y, lr=learning_rate, n_i=n_iter,
                           train_e=train_mse, test_e=test_mse, source='sklearn')

    # Your own perceptron
    perceptron = Perceptron(learning_rate=learning_rate, max_iter=n_iter)
    perceptron.fit(x_train, y_train)
    n_missclass = perceptron.n_missclassified
    n_percent_misclass = n_missclass / x_train.shape[0]
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))



    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    print("Percentage of misclassified samples:", n_percent_misclass)
    plot_decision_boundary(perceptron, x, y, lr=learning_rate, n_i=n_iter,
                           train_e=train_mse, test_e=test_mse, source='our implementation')
    plt.show()

if __name__ == '__main__':
    main()
