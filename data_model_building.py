from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pdb

def get_iris_data():
    data = load_iris()
    # x is instances/records
    # y is class label
    x = data['data']
    y = data['target']

    # merge x and y to a column merge
    input_dataset = np.column_stack([x,y])

    #shuffle dataset to have randomly distributed dataset
    np.random.shuffle(input_dataset)
    return input_dataset

# we need an 80/20 split for training
# 80% of data is training
# 20% is for Test set
train_size = 0.8
test_size = 1-train_size

# get and split data
input_dataset = get_iris_data()
train, test = train_test_split(input_dataset, test_size=test_size)

print "Dataset size ", input_dataset.shape
print "Train size ", train.shape
print "Test size ", test.shape



def get_class_distribution(y):
    """
    given an array of class labels,
    return the class distribution
    """
    distribution = {}
    set_y = set(y)
    for y_label in set_y:
        no_elements = len(np.where(y == y_label)[0])
        distribution[y_label] = no_elements
    dist_percentage = {class_label: count/(1.0*sum(distribution.values())) for
            class_label, count in distribution.items()}
    return dist_percentage

def print_class_label_split(train, test):
    """
    print the class distribution
    in test and train dataset
    """

    y_train = train[:, -1]
    train_distribution = get_class_distribution(y_train)
    print "\nTrain data set class label distribution"
    print "==========================================\n"
    for k,v in train_distribution.items():
        print "class label =%d, percentage records =%.2f"%(k,v)

    y_test = test[:, -1]
    test_distribution = get_class_distribution(y_test)

    print "\nTest data set class label distribution"
    print "=======================================\n"
    for k,v in test_distribution.items():
        print "Class label =%d, percentage records =%.2f"%(k,v)

print_class_label_split(train, test)



