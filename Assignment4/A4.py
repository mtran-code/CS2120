# Name: Michael Tran
# Student Number: 251033279
# Partners: None

import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import sklearn.svm as svm
import sklearn.neural_network as nn
# import sklearn.linear_model as lm
# import sklearn.neighbors as nb
# import sklearn.tree as tree
# import sklearn.gaussian_process as gp
from scipy.sparse import hstack, vstack


def load_arrays_from_csv(path, f):
    """
    Loads the data from a csv file in a given directory and converts the data to two sorted arrays,
    one for features and one for labels. Assumes labels are in the last column of csv file.

    :param path: the directory containing the csv data file (string)
    :param f: the number of features present in data set (int)

    :return: tuple of two arrays, one feature array and one label array
    """

    with open(path) as file:
        raw_data = csv.reader(file)
        complete_data = []
        for row in raw_data:
            if '?' in row:
                continue
            else:
                complete_data.append(row)
    # converts csv file to list of list of strings, complete_data

    complete_data_int = []
    for row in complete_data:
        item = [int(i) for i in row]
        complete_data_int.append(item)
    # converts complete_data list values from str to int

    sorted_complete_data = sorted(complete_data_int, key=lambda sort_by: sort_by[f])
    # sorts data according to label column

    feature_list = []
    label_list = []
    for row in sorted_complete_data:
        feature_list.append(row[:f])
        label_list.append(row[f])
    # splits complete_data into feature_list and label_list

    feature_array = np.array(feature_list)
    label_array = np.array(label_list)
    # converts feature_list and label_list into arrays

    return feature_array, label_array


def label_counts(label_array, l):
    """
    Counts the number of each label type in a given label array, assuming labels begin at 0 and increase by 1.

    :param label_array: a label vector containing labels in any order
    :param l: the number of labels present in the array (int)

    :return: list with number of each label, beginning with number of 0 labels
    """

    label_count_list = []
    for i in range(l):
        current = label_array.tolist().count(i)
        label_count_list.append(current)

    return label_count_list


def split_test_data(feature_array, label_array, l, test_proportion):
    """
    Takes input data/labels and splits the data (in order) to create two smaller data sets;
    one for training machine learning, and one for testing the accuracy of said machine learning.

    :param feature_array: a matrix of feature vectors corresponding to label_array
    :param label_array: a label vector containing sorted labels in ascending order
    :param l: the number of labels present in the array (int)
    :param test_proportion: the desired proportion of the data to be used for testing (float between 0 and 1)

    :return: a tuple (training_data, training_labels, testing_data, testing_labels) of
    the training and testing data (feature vectors) and their corresponding labels
    """

    num_label = label_counts(label_array, l)
    split_index = []
    for i in range(l):
        current_split_index = round(num_label[i] * (1 - test_proportion))
        split_index.append(current_split_index)

    split_label_arrays = np.split(label_array, num_label[:(l - 1)])
    split_feature_arrays = np.split(feature_array, num_label[:(l - 1)])

    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []

    for i in range(l):
        train_feature_array, test_feature_array = np.split(split_feature_arrays[i], [split_index[i]])
        train_label_array, test_label_array = np.split(split_label_arrays[i], [split_index[i]])

        training_data = vstack((training_data, train_feature_array)).toarray().astype(int)
        training_labels = hstack((training_labels, train_label_array)).toarray().astype(int)
        testing_data = vstack((testing_data, test_feature_array)).toarray().astype(int)
        testing_labels = hstack((testing_labels, test_label_array)).toarray().astype(int)

    training_data = np.delete(training_data, 0, 0)
    testing_data = np.delete(testing_data, 0, 0)

    training_labels = training_labels[0]
    testing_labels = testing_labels[0]

    return training_data, training_labels, testing_data, testing_labels


def judge_accuracy(test_data, test_labels, model, show_comparisons=False):
    """
    Judges accuracy of previously fitted machine learning using a sample of data not included in training.
    Taken from Lecture Activity 9, Michael Tran 251033279.

    :param test_data: a data set (feature vectors) containing desired data to be tested.
    :param test_labels: a label set corresponding to the test_data.
    :param model: machine learning module classifier name
    :param show_comparisons: determines whether or not each comparison is printed (bool)

    :return: test accuracy in percent form.
    """

    count = 0
    for i in range(len(test_data)):
        prediction = model.predict(test_data[i].reshape(1, -1))
        actual = [test_labels[i]]
        if show_comparisons is True:
            print(str(prediction) + " = " + str(actual), prediction == actual)
        if prediction == actual:
            count += 1
    accuracy = round((count / len(test_data)) * 100, 2)

    return 'Test Result: ' + str(accuracy) + '% accuracy.'


def compute_bac(test_data, test_labels, model):
    """
    Computes the balanced accuracy score to counteract imbalanced data set classification.

    :param test_data: a data set (feature vectors) containing desired data to be tested.
    :param test_labels: a label set corresponding to the test_data.
    :param model: machine learning module classifier name

    :return: balanced accuracy score in percentage form (str)
    """

    predicted_list = []
    real_list = []
    for i in range(len(test_data)):
        prediction = model.predict(test_data[i].reshape(1, -1))
        actual = [test_labels[i]]
        predicted_list.append(prediction.tolist()[0])
        real_list.append(actual[0])

    bac = round(met.balanced_accuracy_score(real_list, predicted_list), 4)
    return 'Balanced Accuracy Score: ' + str(bac)


def generate_bar_data(features, labels):
    """
    Generates data based on the Bi-RADS Assessment Scores and Ages in the given data,
    and plots it in the form of a double bar graph.

    :param features: feature array of mammographic mass data
    :param labels: label array of mammographic mass data

    :return: plots a double bar graph of Bi-RADS Assessment Score and Age for benign and malignant tumours
    """

    num_label = label_counts(labels, 2)
    split_feature_arrays = np.split(features, num_label[:1])

    avg_birads_list = []
    avg_age_list = []
    for array in split_feature_arrays:
        birads_list = []
        age_list = []
        for row in array:
            birads_list.append(row[0])
            age_list.append(row[1])
        avg_birads_list.append(np.mean(birads_list))
        avg_age_list.append(np.mean(age_list))

    n = 2

    ind = np.arange(n)  # the x locations for the groups
    width = 0.25

    fig, ax1 = plt.subplots()
    birads_set = ax1.bar(ind, avg_birads_list, width, color='r')

    ax1.set_ylabel('Bi-RADS score (scale 1 - 6)', color='r')
    ax1.set_title('Features of Mammographic Masses')
    ax1.set_ylim(ymax=6)
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels(('Benign', 'Malignant'))
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    age_set = ax2.bar(ind + width, avg_age_list, width, color='b')
    ax2.set_ylabel('Age (years)', color='b')
    ax2.set_ylim(ymax=80)
    ax2.tick_params(axis='y', labelcolor='b')

    ax1.legend((birads_set[0], age_set[0]), ('Average Bi-RADS Assessment Score', 'Average Age'), loc=2)

    fig.tight_layout()
    plt.show()


def generate_pie_data(features, labels):
    """
    Generates data based on the Shape, Margin, and Density in the given data,
    and plots it in the form of pie charts.

    :param features: feature array of mammographic mass data
    :param labels: label array of mammographic mass data

    :return: plots two pie charts each for shape, margin, and density of benign and malignant tumours
    """
    num_label = label_counts(labels, 2)
    split_feature_arrays = np.split(features, num_label[:1])

    shape_counts = []
    margin_counts = []
    density_counts = []
    for array in split_feature_arrays:
        shape_list = []
        margin_list = []
        density_list = []
        for row in array:
            shape_list.append(row[2])
            margin_list.append(row[3])
            density_list.append(row[4])
        shape_count = []
        margin_count = []
        density_count = []
        for i in range(1, 5):
            shape_count.append(shape_list.count(i))
            density_count.append(density_list.count(i))
        for i in range(1, 6):
            margin_count.append(margin_list.count(i))
        shape_counts.append(shape_count)
        margin_counts.append(margin_count)
        density_counts.append(density_count)

    def set_up_labels(ax, patches, feature_names, x_offset=None, y_offset=None):
        if x_offset is None:
            x_offset = [1.4, 1.4, 1.4, 1.4, 1.4]
        if y_offset is None:
            y_offset = [1.2, 1.2, 1.2, 1.2, 1.2]
        normed_features = [str(round((float(number) / sum(feature_names)) * 100, 2)) + '%' for number in feature_names]
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
        for patch, p in enumerate(patches):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontal_alignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(
                normed_features[patch],
                xy=(x, y), xytext=(x_offset[patch] * np.sign(x), y_offset[patch] * y),
                horizontalalignment=horizontal_alignment, **kw)

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Features of Benign and Malignant Mammographic Masses')

    shape_patches1, shape_texts1 = axs[0, 0].pie(
        shape_counts[0],
        wedgeprops=dict(width=0.8),
        startangle=-30)
    axs[0, 0].set_title('Shapes of Benign Masses')
    set_up_labels(
        axs[0, 0],
        shape_patches1, shape_counts[0])
    shape_patches2, shape_texts2 = axs[0, 1].pie(
        shape_counts[1],
        wedgeprops=dict(width=0.8),
        startangle=-40)
    axs[0, 1].set_title('Shapes of Malignant Masses')
    set_up_labels(
        axs[0, 1],
        shape_patches2, shape_counts[1])
    fig.legend(
        shape_patches1,
        ['Round', 'Oval', 'Lobular', 'Irregular'],
        loc=9, bbox_to_anchor=(0.5, 0.83))

    margin_patches1, margin_tests1 = axs[1, 0].pie(
        margin_counts[0],
        wedgeprops=dict(width=0.8),
        startangle=70)
    axs[1, 0].set_title('Margins of Benign Masses')
    set_up_labels(
        axs[1, 0],
        margin_patches1, margin_counts[0])
    margin_patches2, margin_tests2 = axs[1, 1].pie(
        margin_counts[1],
        wedgeprops=dict(width=0.8),
        startangle=-50)
    axs[1, 1].set_title('Margins of Malignant Masses')
    set_up_labels(
        axs[1, 1],
        margin_patches2, margin_counts[1])
    fig.legend(
        margin_patches1,
        ['Circumscribed', 'Microlobulated', 'Obscured', 'Ill-defined', 'Spiculated'],
        loc=10, bbox_to_anchor=(0.5, 0.5))

    exploders = [0.15, 0.1, 0, 0.2]
    density_patches1, density_tests1 = axs[2, 0].pie(
        density_counts[0],
        wedgeprops=dict(width=0.8),
        startangle=-20,
        explode=exploders)
    axs[2, 0].set_title('Densities of Benign Masses')
    set_up_labels(
        axs[2, 0],
        density_patches1, density_counts[0],
        x_offset=[2, 2, 1.4, 2], y_offset=[1.5, 1.5, 1.2, 2])
    density_patches2, density_tests2 = axs[2, 1].pie(
        density_counts[1],
        wedgeprops=dict(width=0.8),
        startangle=-25,
        explode=exploders)
    axs[2, 1].set_title('Densities of Malignant Masses')
    set_up_labels(
        axs[2, 1],
        density_patches2, density_counts[1],
        x_offset=[2, 2, 1.4, 2], y_offset=[1.5, 1.4, 1.2, 2])
    fig.legend(
        margin_patches1,
        ['High', 'Iso', 'Low', 'Fat-containing'],
        loc=8, bbox_to_anchor=(0.5, 0.18))

    plt.show()


def load_data():
    """
    Loads the data as complete features and labels as well as split features and labels for training and testing.

    :return: data (list of data arrays)
    """
    features, labels = load_arrays_from_csv('./mammographic_masses/mammographic_masses.csv', 5)
    data = [features, labels]

    train_features, train_labels, test_features, test_labels = split_test_data(data[0], data[1], 2, 0.1)
    data.extend([train_features, train_labels, test_features, test_labels])

    return data


def viz1(data):
    """
    First visualization function to generate bar graph of Bi-RADS score and age.
    """
    generate_bar_data(data[0], data[1])


def viz2(data):
    """
    Second visualization function to generate series of pie charts of shape, margin, and density of masses.
    """
    generate_pie_data(data[0], data[1])


def learn1(data):
    """
    First machine learning function to compute an accuracy percentage for machine learning model.
    """
    ml_model = svm.SVC(gamma='auto')
    # OTHER POSSIBLE ALTERNATIVE MACHINE LEARNING MODELS (replace line above):
    # ml_model = svm.LinearSVC(max_iter=10000)
    # ml_model = svm.SVC(gamma='auto')
    # ml_model = nn.MLPClassifier(max_iter=1000)
    # ml_model = lm.SGDClassifier()
    # ml_model = nb.KNeighborsClassifier()
    # ml_model = gp.GaussianProcessClassifier()
    # ml_model = tree.DecisionTreeClassifier()
    ml_model.fit(data[2], data[3])
    result = judge_accuracy(data[4], data[5], ml_model, show_comparisons=True)

    print(result)


def learn2(data):
    """
    Second machine learning function to compute balanced accuracy score for machine learning model.
    """
    ml_model = nn.MLPClassifier(max_iter=1000)
    # OTHER POSSIBLE ALTERNATIVE MACHINE LEARNING MODELS (replace line above):
    # ml_model = svm.LinearSVC(max_iter=10000)
    # ml_model = svm.SVC(gamma='auto')
    # ml_model = nn.MLPClassifier(max_iter=1000)
    # ml_model = lm.SGDClassifier()
    # ml_model = nb.KNeighborsClassifier()
    # ml_model = gp.GaussianProcessClassifier()
    # ml_model = tree.DecisionTreeClassifier()
    ml_model.fit(data[2], data[3])
    result = compute_bac(data[4], data[5], ml_model)

    print(result)


mammographic_masses_data = load_data()
learn1(mammographic_masses_data)
