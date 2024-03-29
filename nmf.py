# Non-negative matrix factorization (NMF)
# Some notes
# Non-negative Matrix Factorization(NFM) is used extensively in recommendation
# systems using a collaborative filtering algorithm
# Say that our input  matrix A is of a dimension m x n. NMF factorizes the input
# matrix into two matrices, A_dash and H:
# A = A_dash * H
# Lets say we want to reduce the dimension of the A matrix to d, that is, we
# want the original m x n matric to be decomposed into m x d, where d << n
# The A_dash matrix is of a size m x d and the H matrix is of a size d x m. NMF
# solves this as an optimization problem, that is, minimizing the function:
# |A-A_dash * H|^2       ^2 = squared
# This was used to solve the famous Netflix challenge


import numpy as np
from collections import defaultdict
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pdb

# ratings matrix in python
ratings = [
    [5.0,5.0,4.5,4.5,5.0,3.0,2.0,2.0,0.0,0.0],
    [4.2,4.7,5.0,3.7,3.5,0.0,2.7,2.0,1.9,0.0],
    [2.5,0.0,3.3,3.4,2.2,4.6,4.0,4.7,4.2,3.6],
    [3.8,4.1,4.6,4.5,4.7,2.2,3.5,3.0,2.2,0.0],
    [2.1,2.6,0.0,2.1,0.0,3.8,4.8,4.1,4.3,4.7],
    [4.7,4.5,0.0,4.4,4.1,3.5,3.1,3.4,3.1,2.5],
    [2.8,2.4,2.1,3.3,3.4,3.8,4.4,4.9,4.0,4.3],
    [4.5,4.7,4.7,4.5,4.9,0.0,2.9,2.9,2.5,2.1],
    [0.0,3.3,2.9,3.6,3.1,4.0,4.2,0.0,4.5,4.6],
    [4.1,3.6,3.7,4.6,4.0,2.6,1.9,3.0,3.6,0.0]
]

movie_dict = {
    1:"Star Wars",
    2:"Matrix",
    3:"Inception",
    4:"Harry Potter",
    5:"The Hobbit",
    6:"Guns of Navarone",
    7:"Saving Private Ryan",
    8:"Enemy at the gates",
    9:"Where eagles dare",
    10:"Great Escape"
}

A = np.asmatrix(ratings, dtype=float)

# preform non-negative matrix transformations on the data.
max_components = 2
reconstruction_error = []
nmf = None
nmf = NMF(n_components = max_components, random_state=1)
A_dash = nmf.fit_transform(A)

# Examine the reduced matrix for i in range(A_dash.shape([0]):
for i in range(A_dash.shape[0]):
    print "User id = %d, comp1 score = %0.2f, comp 2 score = %0.2f"%(i+1,
            A_dash[i][0], A_dash[i][1])

plt.figure(1)
plt.title("User Concept Mapping")
x = A_dash[:,0]
y = A_dash[:,1]
plt.scatter(x,y)
plt.xlabel("Components 1 Score")
plt.ylabel("Components 2 Score")

# Let us examine our component MAtrix F.
F = nmf.components_
plt.figure(2)
plt.title("Movie Concept Mapping")
x = F[0,:]
y = F[1,:]
plt.scatter(x,y)
plt.xlabel("Component 1 score")
plt.ylabel("Component 2 score")
for i in range(F[0,:].shape[0]):
    plt.annotate(movie_dict[i+1], (F[0,:][i], F[1,:][i]))
plt.show()

