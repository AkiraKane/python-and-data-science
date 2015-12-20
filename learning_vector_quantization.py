# Learning Vector Quantization is a model-free method for clustering data
# points. LVQ can be used for classification tasks. LVQ is an online learning
# algorithm where the data points are processed one at a time. It makes a very
# simple intuition.


from sklearn.datasets import load_iris
import warnings
import numpy as np
from sklearn.metrics import euclidean_distances
import pdb
warnings.simplefilter("ignore")

data = load_iris()
x = data['data']
y = data['target']

# scale the variables
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
x = minmax.fit_transform(x)

# First declare the parameters for LVQ
R = 2
n_classes = 3
epsilon = 0.9
epsilon_dec_factor = 0.001

# Define the class the hold the prototype vectors:
class prototype(object):
    def __init__(self, class_id, p_vector, epsilon):
        self.class_id = class_id
        self.p_vector = p_vector
        self.epsilon = epsilon

    def update(self, u_vector, increment=True):
        if increment:
            # move the prototype vector closer to the input vector
            self.p_vector = self.p_vector + self.epsilon*(u_vector -
                    self.p_vector)
        else:
            # move the prototype vector away from the input vector
            self.p_vector = self.p_vector - self.epsilon*(u_vector -
                    self.p_vector)

# this function finds the closest prototype vector to the given vector
def find_closest(in_vector, proto_vectors):
    closest = None
    closest_distance = 99999
    for p_v in proto_vectors:
        distance = euclidean_distances(in_vector, p_v.p_vector)
        if distance < closest_distance:
            closest_distance = distance
            closest = p_v
    return closest

def find_class_id(test_vector, p_vectors):
    return find_closest(test_vector, p_vectors).class_id

# choose R initial prototypes for each class
p_vectors = []
for i in range(n_classes):
    # select a class
    y_subset = np.where(y == i)
    # select tuples for choosen class
    x_subset = x[y_subset]
    # get R random indices between 0 and 50
    samples = np.random.randint(0,len(x_subset),R)
    # select p_vectors
    for sample in samples:
        s = x_subset[sample]
        p = prototype(i, s, epsilon)
        p_vectors.append(p)

print "class id \t Initial prototype vector\n"
for p_v in p_vectors:
    print p_v.class_id,'\t',p_v.p_vector
    print

while epsilon >= 0.01:
    # sample a training instance randompy
    rnd_i = np.random.randint(0,149)
    rnd_s = x[rnd_i]
    target_y = y[rnd_i]

    #decrement epsilon value for next iteration
    epsilon = epsilon - epsilon_dec_factor
    # find closest prototype vector to given point
    closest_pvector = find_closest(rnd_s, p_vectors)

    #update the closest prototype vector
    if target_y == closest_pvector.class_id:
        closest_pvector.update(rnd_s)
    else:
        closest_pvector.update(rnd_s, False)
    closest_pvector.epsilon = epsilon

print "class id \t Final prototype Vector\n"
for p_vector in p_vectors:
    print p_vector.class_id, '\t', p_vector.p_vector

# small test to verify the correctness of the methods
predicted_y = [find_class_id(instance, p_vectors) for instance in x]

from sklearn.metrics import classification_report
print
print classification_report(y, predicted_y, target_names=['Iris-Setosa',
    'Iris-Versicolour', 'Iris-Virginica'])


