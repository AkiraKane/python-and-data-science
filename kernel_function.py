import numpy as np
# Simple example to illustrate Kernel Function concept.

# Find a mapping function to transform this space
# phi(x1,x2,x3) = (x1x2, x1x3, x2x3, x1x1, x2x2, x3x3)
# this will trasnform the input space into 6 dimensions

def mapping_function(x):
    output_list = []
    for i in range(len(x)):
        output_list.append(x[i]*x[i])

    output_list.append(x[0]*x[1])
    output_list.append(x[0]*x[2])
    output_list.append(x[1]*x[0])
    output_list.append(x[1]*x[2])
    output_list.append(x[2]*x[1])
    output_list.append(x[2]*x[0])
    return np.array(output_list)


if __name__ == "__main__":
    # 3 Dimensional input space
    x = np.array([10,20,30])
    y = np.array([8,9,10])

    # apply the mapping function
    tranf_x = mapping_function(x)
    tranf_y = mapping_function(y)
    print "output of mapping function"
    print "x:", x
    print "mapped x(tranf_x): ", tranf_x
    print "y: ", y
    print "mapped y(tranf_y): ", tranf_y
    print
    print "dot on tranf_x and tranf_y"
    print np.dot(tranf_x, tranf_y)

    # 
    print "the equivalent kernel functions transformation output"
    output = np.power((np.dot(x,y)),2)
    print output
