from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from HW2_NN_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np


# only for saving to csv
import pandas as pd

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
    random_seeds = [8, 100, 3, 39, 42]
    models = []
    train_norm, test_norm = prep_pixels(X_train, X_test)

    for seed in random_seeds:

        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, random_state=seed, activation='tanh')
        mlp.fit(train_norm, y_train)
        models.append(mlp)

        print("Accuracy:", mlp.score(test_norm, y_test))
        train_accuracy = mlp.score(train_norm, y_train)
        test_accuracy = mlp.score(test_norm, y_test)
        print("Train accuracy:", train_accuracy)

        train_accuracy_values.append(train_accuracy)
        test_accuracy_values.append(test_accuracy)

        y_test_array = np.asarray(y_test)
        missclassified_index = np.where(y_test_array != mlp.predict(test_norm))
        print("Missclassified:", len(missclassified_index[0]))

        random_1 = np.random.randint(0, len(missclassified_index[0]))
        mis_clas_img_1 = X_test[missclassified_index[0][random_1]]

        image_1_classified_as = mlp.predict([mis_clas_img_1])
        image_1_classified_as_string = label_mapping[image_1_classified_as[0]]
        image_1_actual = y_test[missclassified_index[0][random_1]]
        image_1_actual_string = label_mapping[image_1_actual]

        plot_image(mis_clas_img_1, image_1_classified_as_string, image_1_actual_string)





    print('Test accuracy values:')
    print(test_accuracy_values)
    print('Train accuracy values:')
    print(train_accuracy_values)
    plot_boxplot(train_accuracy_values, test_accuracy_values)

    max_accuracy = max(test_accuracy_values)
    max_index = test_accuracy_values.index(max_accuracy)
    best_model = models[max_index]
    print("Best model:\n", best_model)

    plot_hidden_layer_weights(best_model.coefs_[0])

    plt.figure(figsize=(10, 12))
    plot_confusion_matrix(best_model, X_test, y_test, display_labels=label_mapping.values(), cmap=plt.cm.Blues)
    best_conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))

    best_model_res = perf_measure(best_conf_matrix)
    res_to_csv(best_model_res)

    n_missclassified = len(best_model_res['FP']) + len(best_model_res['FN'])
    print("Number of missclassified:", n_missclassified)


    plt.xticks(rotation=90)
    plt.tight_layout(pad=2)
    plt.title("Confusion matrix for best model")
    plt.savefig("confusion_matrix.png")
    plt.show()

def res_to_csv(res):
    df = pd.DataFrame(res)
    df.to_csv('results.csv')
def perf_measure(confusion_matrix):
    res = {}

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    res['FP'] = FP
    res['FN'] = FN
    res['TP'] = TP
    res['TN'] = TN

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    res['TPR'] = TPR
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    res['TNR'] = TNR
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    res['PPV'] = PPV
    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + FN + TN)
    res['ACC'] = ACC
    # Harmonic mean of precision and recall
    F_1 = 2 * (PPV * TPR) / (PPV + TPR)
    res['F_1'] = F_1

    print('----------------------Results Confusion Matrix----------------------')
    print(res)
    print('-------------------------------------------------------------------')


    return res

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm
