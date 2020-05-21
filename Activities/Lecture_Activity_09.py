import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
data = iris.data
labels = iris.target
np.unique(labels)
knn = KNeighborsClassifier()
# Set up data set


def train_test_split(data, labels, n, test_proportion):
    """
    Takes input data/labels and splits the data (in order) to create two smaller data sets;
    one for training machine learning, and one for testing the accuracy of said machine learning.

    :param data: a data set (matrix of feature vectors)
    :param labels: a label set (label vector)
    :param n: the number of categories
    :param test_proportion: a percentage (number between 0 and 1) for the test

    :return: a tuple (train_data, train_labels, test_data, test_labels) of the
    training and testing data (feature vectors) and their corresponding labels
    """

    data_per_label = int(len(labels) / n)
    # The number of data points per unique label.
    data_per_training = round(data_per_label * (1 - test_proportion))
    # The number of desired data_points in train_data, dictated by test_proportion input.
    # Rounded to nearest integer, as the number of data points is a discrete value.

    data_dict = {}
    label_dict = {}
    start = 0
    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}
    # Set up variables and empty dictionaries for use in subsequent code.

    for i in range(n):
        data_dict[i] = data[start:start + data_per_label]
        label_dict[i] = labels[start:start + data_per_label]
        start = int(start + data_per_label)
    # Split input data and labels according to label, and add to respective dictionary.
    # Essentially reformat data from array to a dictionary form.

    for i in range(n):
        train_dict[i] = data_dict[i][0:data_per_training]
        train_label_dict[i] = label_dict[i][0:data_per_training]
        test_dict[i] = data_dict[i][data_per_training:]
        test_label_dict[i] = label_dict[i][data_per_training:]
    # Create new dictionaries from first dictionary, this time separating into training data set and testing data set.

    train_data = np.array(train_dict[0])
    test_data = np.array(test_dict[0])
    train_labels = np.array(train_label_dict[0])
    test_labels = np.array(test_label_dict[0])
    # Set up the first label for the output arrays.

    for i in range(1, n):
        train_data_array = np.array(train_dict[i])
        train_data = np.concatenate((train_data, train_data_array))
        train_label_array = train_label_dict[i]
        train_labels = np.concatenate((train_labels, train_label_array))

        test_data_array = np.array(test_dict[i])
        test_data = np.concatenate((test_data, test_data_array))
        test_label_array = test_label_dict[i]
        test_labels = np.concatenate((test_labels, test_label_array))
    # Set up all subsequent labels for output array, concatenating to the previously set up label each time.

    return (train_data, train_labels, test_data, test_labels)


def judge_accuracy(test_data, test_labels):
    """
    Judges accuracy of previously fitted machine learning using a sample of data not included in training.

    :param test_data: a data set (feature vectors) containing desired data to be tested.
    :param test_labels: a label set corresponding to the test_data.

    :return: test accuracy in percent form.
    """

    count = 0
    for i in range(len(test_data)):
        pred = knn.predict(np.reshape(test_data[i], (1, -1)))
        actual = [test_labels[i]]

        print(str(pred) + " = " + str(actual), pred == actual)
        if pred == actual:
            count += 1
    accuracy = round((count / len(test_data)) * 100, 2)

    return "Test Result: " + str(accuracy) + "% accuracy."


# CHANGE TEST PROPORTION HERE
test_proportion = 0.6
# greater test_proportion = lower test accuracy in general
# lower test_proportion = greater test accuracy in general

# OUTPUT
train_data, train_labels, test_data, test_labels = train_test_split(data, labels, 3, test_proportion)
knn.fit(train_data, train_labels)
print(judge_accuracy(test_data, test_labels))

# Sample Output when test_proportion = 0.6
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [0] = [0] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [2] = [1] [False]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [2] = [1] [False]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [1] = [1] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [1] = [2] [False]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [1] = [2] [False]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# [2] = [2] [ True]
# Test Result: 95.56% accuracy.