# Importing numpy
import pdb
import numpy as np

a_list = [1,2,3]
an_array = np.array(a_list)
# specify datatype
an_array = np.array(a_list, dtype=float)


# creating matrices
a_listoflist = [[1,2,3],[5,6,7],[8,9,10]]
a_matrix = np.matrix(a_listoflist, dtype=float)

def display_shape(a):
    print
    print a
    print
    print "Number of elements in a = %d"%(a.size)
    print "NUmber of dimensions in a = %d"%(a.ndim)
    print "Rows and Columns in a ",a.shape
    print
    print
    print
    print

display_shape(a_matrix)
# pdb.set_trace()

# array using np.linspace
created_array = np.linspace(1,10)
display_shape(created_array)


# array using np.logspace
created_array = np.logspace(1,10, base=10.0)
display_shape(created_array)

# Specify step size in arange while creating
# an array. This is where it is different
# from np.linspace
created_array = np.arange(1,10,2,dtype=int)
display_shape(created_array)

# a matrix withe all elements as 1
ones_matrix = np.ones((3,3))
display_shape(ones_matrix)
# a matrix with all elements as 0
zero_matrix = np.zeros((3,3))
display_shape(zero_matrix)


# identity matrix
# k parameter controls the index of 1
# if k = 0, (0,0), (1,1), (2,2) cell values
# are set to 1 in a 3x3 matrix
identity_matrix = np.eye(N=3, M=3, k=0)
display_shape(identity_matrix)
identity_matrix = np.eye(N=3, k=1)
display_shape(identity_matrix)


print
print "an arranged matrix"
a_matrix = np.arange(9)
display_shape(a_matrix)
print "then we reshape it to (3,3)"
a_matrix = a_matrix.reshape(3,3)
display_shape(a_matrix)
print "displaying matrix with [::-1]"
display_shape(a_matrix[::-1])


# random numbers
gen_random_num = np.random.randint(1,100,size=10)
print gen_random_num
uniform_rnd_numbers = np.random.normal(loc=0.2, scale=0.2, size=(3,3))
print uniform_rnd_numbers
