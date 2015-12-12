import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict
from matplotlib.pylab import frange
import pdb

# load data nd handle missing values
fill_data = lambda x : int(x.strip() or 0)
data = np.genfromtxt('president.txt', dtype=(int,int),
        converters={1:fill_data}, delimiter=",")
x = data[:, 0]
y = data[:, 1]


# group data using frequency, count of individual data points
# given a set of points, Counter() returns a dictionary, where key is a data
# point and a value is the frequency of data point in data set
x_freq = Counter(y)
x_ = np.array(x_freq.keys())
y_ = np.array(x_freq.values())

# Group data by range of years
x_group = OrderedDict()
group = 5
group_count = 1
keys = []
values = []
for i, xx in enumerate(x):
    ## individual data points is appended to list of keys
    keys.append(xx)
    values.append(y[i])
    #if we have procedded five data points (ie 5 years)
    if group_count == group:
        # convert the lsit of keys to a tuple
        # user the new tuple as the key to x_group dict
        x_group[tuple(keys)] = values
        keys = []
        values = []
        group_count = 1

    group_count += 1

# Accomodate the last batch of keys and values
x_group[tuple(keys)] = values

print x_group
# Plot the grouped data as dot plot
plt.subplot(311)
plt.title("Dot Plot by frequency")
plt.plot(y_, x_, 'ro')
plt.xlabel('Count')
plt.ylabel('# PResidential Request')
# min and max limits
plt.xlim(min(y_)-1, max(y_)+1)

plt.subplot(312)
plt.title("Simple dot plot")
plt.xlabel("# Presedential Request")
plt.ylabel("Frequency")

# prepare data for dot plot
for key, value in x_freq.items():
    x__ = np.repeat(key, value)
    y__ = frange(0.1, (value/10.0), 0.1)
    try:
        plt.plot(x__, y__, 'go')
    except ValueError:
        print x__.shape, y__.shape
    # set min and max
    plt.ylim(0.0, 0.4)
    plt.xlim(xmin=-1)

plt.xticks(x_freq.keys())
plt.subplot(313)
x_vals = []
x_labels = []
y_vals = []
x_tick = 1
for k,v in x_group.items():
    for i in range(len(k)):
        x_vals.append(x_tick)
        x_label = '-'.join([str(kk) if not i else str(kk) [-2:] for i, kk in
            enumerate(k)])
        x_labels.append(x_label)
    y_vals.extend(list(v))
    x_tick +=1

plt.title("Dot Plot by Year Grouping")
plt.xlabel("Year Group")
plt.ylabel("No presedential request")
try:
    plt.plot(x_vals, y_vals, 'ro')
except ValueError:
    print len(x_vals), len(y_vals)

plt.xticks(x_vals, x_labels, rotation=-35)
plt.show()




