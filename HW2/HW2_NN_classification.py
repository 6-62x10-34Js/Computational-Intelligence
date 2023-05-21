from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, plot_confusion_matrix

from sklearn.neural_network import MLPClassifier
from HW2_NN_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np

"""
Assignment 2: Neural networks
Part 3.2: Classification with Neural Networks: Fashion MNIST

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def ex_3_2(X_train, y_train, X_test, y_test):
    """
    Snippet for exercise 3.2
    :param X_train: Train set
    :param y_train: Targets for the train set
    :param X_test: Test set
    :param y_test: Targets for the test set
    :return:
    """

    label_mapping = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    train_accuracy_values = []
    test_accuracy_values = []
    random_seeds = [1]
    models = []
    train_norm, test_norm = prep_pixels(X_train, X_test)
    for seed in random_seeds:
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, random_state=seed, activation='tanh')
        mlp.fit(train_norm, y_train)
        models.append(mlp)
        y_pred_test = mlp.predict(test_norm)
        y_pred_train = mlp.predict(train_norm)
        print("Accuracy:", mlp.score(test_norm, y_test))
        #print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

        plot_hidden_layer_weights(mlp.coefs_[0])

        y_true_categories = np.vectorize(label_mapping.get)(y_pred_test)
        y_pred_categories = np.vectorize(label_mapping.get)(y_test)

        train_accuracy = mlp.score(train_norm, y_train)
        test_accuracy = mlp.score(test_norm, y_test)
        print("Train accuracy:", train_accuracy)

        train_accuracy_values.append(train_accuracy)
        test_accuracy_values.append(test_accuracy)

    plot_boxplot(train_accuracy_values, test_accuracy_values)

    max_accuracy = max(test_accuracy_values)
    max_index = test_accuracy_values.index(max_accuracy)
    best_model = models[max_index]
    print("Best model:\n", best_model)
    plot_confusion_matrix(best_model, X_test, y_test, display_labels=label_mapping.values(), cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.show()

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm
