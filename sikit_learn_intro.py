from sklearn.datasets import load_iris, load_boston, make_classification, make_circles, make_moons
import pdb
import numpy as np
import matplotlib.pyplot as plt

# Iris dataset
data = load_iris()
x = data['data']
y = data['target']
y_labels = data['target_names']
x_labels = data['feature_names']

print
print x.shape
print y.shape
print x_labels
print y_labels

# Boston dataset
data = load_boston()
x = data['data']
y = data['target']
x_labels = data['feature_names']

print
print x.shape
print y.shape
print x_labels


# make some classification dataset
x, y = make_classification(n_samples=50,  n_features=5, n_classes=2)
print
print x.shape
print y.shape
print x[1, :]
print y[1]

# Some non linear dataset
x, y = make_circles()
plt.close('all')
plt.figure(1)
plt.scatter(x[:,0], x[:,1], c=y)

x, y = make_moons()
plt.figure(2)
plt.scatter(x[:,0], x[:,1], c=y)

# plt.show()

from sklearn.preprocessing import PolynomialFeatures
# Data Preprocessing routines
x = np.asmatrix([[1,2],[2,4]])
poly = PolynomialFeatures(degree = 2)
poly.fit(x)
x_poly = poly.transform(x)
print "Original x variable shape", x.shape
print x
print
print "Transformed x variables", x_poly.shape
print x_poly

# alternatively
x_poly = poly.fit_transform(x)

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

data = load_iris()
x = data['data']
y = data['target']
estimator = DecisionTreeClassifier()
estimator.fit(x,y)
predicted_y = estimator.predict(x)
predicted_y_prob = estimator.predict_proba(x)
predicted_y_lprob = estimator.predict_log_proba(x)

from sklearn.pipeline import Pipeline
poly = PolynomialFeatures(3)
tree_estimator = DecisionTreeClassifier()

steps = [('poly',poly), ('tree',tree_estimator)]
estimator = Pipeline(steps=steps)
estimator.fit(x,y)
predicted_y = estimator.predict(x)
pdb.set_trace()

