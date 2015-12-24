from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pprint



def get_data():
    data = load_iris()
    x = data['data']
    y = data['target']
    label_names = data['target_names']

    return x,y,label_names.tolist()

def get_train_test(x,y):
    """
    Prepare a stratified train and test split
    """
    train_size = 0.8
    test_size = 1 - train_size
    input_dataset = np.column_stack([x,y])
    stratified_split = StratifiedShuffleSplit(input_dataset[:,-1], test_size=test_size,
            n_iter=1, random_state=77)

    for train_indx, test_indx in stratified_split:
        train_x = input_dataset[train_indx,:-1]
        train_y = input_dataset[train_indx,-1]
        test_x = input_dataset[test_indx,:-1]
        test_y = input_dataset[test_indx,-1]

    return train_x, train_y, test_x, test_y

def build_model(x,y):
    """
    Fit the model for the given attrivute
    class label pairs
    """
    model = tree.DecisionTreeClassifier(criterion="entropy")
    model = model.fit(x,y)
    return model

def test_model(x,y,model,label_names):
    """
    Inspect the mdoel for accuracy
    """
    y_predicted = model.predict(x)
    print "MOdel accuracy = %0.2f"%(accuracy_score(y,y_predicted)*100) + "%\n"
    print "\nConfusion Matrix"
    print "==================="
    print pprint.pprint(confusion_matrix(y,y_predicted))
    print "\nClassification Report"
    print "======================="
    print classification_report(y,y_predicted,target_names=label_names)

def get_feature_names():
    data = load_iris()
    return data['feature_names']

def probe_model(x,y,model, label_names):
    feature_names = get_feature_names()
    feature_importance = model.feature_importances_
    print "\nFeature Importance \n"
    print "=====================\n"
    for i, feature_name in enumerate(feature_names):
        print "%s = %0.3f"%(feature_name, feature_importance[i])

        tree.export_graphviz(model, out_file='tree.dot')




if __name__ == "__main__":
    x,y,label_names = get_data()

    train_x, train_y, test_x, test_y = get_train_test(x,y)

    model = build_model(train_x, train_y)

    probe_model(x,y,model,label_names)

    test_model(train_x, train_y, model, label_names)

    test_model(test_x,test_y,model,label_names)



