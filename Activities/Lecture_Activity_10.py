import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
data = iris.data
labels = iris.target


def judge_accuracy(test_data, test_labels, model, show_comparisons=False):
    """
    Judges accuracy of previously fitted machine learning using a sample of data not included in training.

    :param test_data: a data set (feature vectors) containing desired data to be tested.
    :param test_labels: a label set corresponding to the test_data.
    :param model: machine learning model such as support vector classification (svc)
    :param show_comparisons: determines whether or not to show each prediction comparison to actual (bool)

    :return: test accuracy in percent form.
    """

    count = 0
    for i in range(len(test_data)):
        prediction = model.predict(np.reshape(test_data[i], (1, -1)))
        actual = [test_labels[i]]
        if show_comparisons is True:
            print(str(prediction) + " = " + str(actual), prediction == actual)
        if prediction == actual:
            count += 1
    accuracy = round((count / len(test_data)) * 100, 2)

    return 'Test Result: ' + str(accuracy) + '% accuracy'


training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, stratify=labels)

svc = svm.SVC(kernel='linear', gamma='auto')
svc.fit(training_data, training_labels)

svc2 = svm.SVC(kernel='poly', gamma='auto')
svc2.fit(training_data, training_labels)

print(judge_accuracy(testing_data, testing_labels, svc), 'for svc linear kernel.')
print(judge_accuracy(testing_data, testing_labels, svc), 'for svc polynomial kernel.')
