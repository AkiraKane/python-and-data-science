from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report

def get_data():
    """
    Make a sample classification dataset
    returns : Independent variable y, dependent variable x
    """
    x, y = make_classification(n_features=4)
    return x,y

def plot_data(x,y):
    """
    Plot a scatter plot for all variable combinations
    """
    subplot_start = 321
    col_numbers = range(0,4)
    col_pairs = itertools.combinations(col_numbers,2)

    for col_pair in col_pairs:
        plt.subplot(subplot_start)
        plt.scatter(x[:,col_pair[0]], x[:,col_pair[1]], c=y)
        title_string = str(col_pair[0]) + "-" + str(col_pair[1])
        plt.title(title_string)
        x_label = str(col_pair[0])
        y_label = str(col_pair[1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        subplot_start += 1

    plt.show()

def get_train_test(x,y):
    """
    Prepare a stratified train and test split"
    """
    train_size = 0.8
    test_size = 1-train_size
    input_dataset = np.column_stack([x,y])
    stratified_split = StratifiedShuffleSplit(input_dataset[:,-1],
            test_size=test_size, n_iter=1)

    for train_indx, test_indx in stratified_split:
        train_x = input_dataset[train_indx, :-1]
        train_y = input_dataset[train_indx, -1]
        test_x = input_dataset[test_indx, :-1]
        test_y = input_dataset[test_indx, -1]
    return train_x, train_y, test_x, test_y

def build_model(x,y,k=2):
    """
    Fit a nearest neighbor model
    """
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x,y)
    return knn

def test_model(x,y,knn_model):
    y_predicted = knn_model.predict(x)
    print classification_report(y,y_predicted)

if __name__ == "__main__":
    #Load data
    x,y = get_data()
    #Scatter plot the data
    plot_data(x,y)

    # split the data into train and test
    train_x, train_y, test_x, test_y = get_train_test(x,y)

    #build the model
    knn_model = build_model(train_x, train_y)

    # test the model
    print "\nModle evaluation on the training set"
    print "============================\n"

    # test the model
    print "\nModel evaluation on training set"
    print "=============================\n"
    test_model(train_x, train_y, knn_model)

    print "\nModel evaluation on test set"
    print "=============================\n"
    test_model(test_x, test_y, knn_model)



