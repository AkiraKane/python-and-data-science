import numpy as np
import pdb

def euclidean_distance(x,y):
    if len(x) == len(y):
        return np.sqrt(np.sum(np.power((x-y),2)))
    else:
        print "Input should be of equal length"
    return None

def lrNorm_distance(x,y,power):
    if len(x) == len(y):
        return np.power(np.sum(np.power((x-y),power)),(1/(1.0*power)))
    else:
        print "Input should be of equal length"
    return None

def cosine_distance(x,y):
    if len(x) == len(y):
        return np.dot(x,y) / np.sqrt(np.dot(x,x) * np.dot(y,y))
    else: 
        print "Input should be of equal length"
    return None

def jaccard_distance(x,y):
    set_x = set(x)
    set_y = set(y)
    return 1 - len(set_x.intersection(set_y)) / len(set_x.union(set_y))

def hamming_distance(x,y):
    diff = 0
    if len(x) == len(y):
        for char1, char2 in zip(x,y):
            if char1 != char2:
                diff += 1
        return diff
    else:
        print "Input should be of equal length"
    return None


if __name__ == "__main__":
    # sample data, 2 vectors of dimension 3
    x = np.asarray([1,2,3])
    y = np.asarray([1,2,3])
    print
    print "This is Euclidean Distance"
    print euclidean_distance(x,y)
    print
    print "euclidean by invoking lr norm with r value of 2:"
    print lrNorm_distance(x,y,2)
    print "euclidean by invoking lr norm with r value of 1:"
    print lrNorm_distance(x,y,1)
    print

    # sample data for cosine distance
    x = [1,1]
    y = [1,0]
    print 'cosine distance'
    print cosine_distance(x,y)

    # sample data for jaccard distance
    x = [1,2,3]
    y = [1,2,3]
    print 'jaccard distance'
    print jaccard_distance(x,y)

    # sample data for hamming distance
    x = [11001]
    y = [11011]
    print "hamming distance"
    print hamming_distance(x,y)







 

