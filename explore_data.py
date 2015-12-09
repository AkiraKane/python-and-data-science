import numpy as np
from matplotlib.pylab import frange
import matplotlib.pyplot as plt

fill_data = lambda x : int(x.strip() or 0)
data = np.genfromtxt('president.txt', dtype=(int, int),
        converters = {1:fill_data}, delimiter=  ',')
x = data[:,0]
y = data[:,1]


# Plot the data to look for trends or values
plt.close('all')
plt.figure(1)
plt.title("All Data")
plt.plot(x,y,'ro')
plt.xlabel('year')
plt.ylabel('No presidential request')

# calculate the percentile values (25th, 50th, 75th) for the data ot understand
# the data distribution
perc_25 = np.percentile(y,25)
perc_50 = np.percentile(y,50)
perc_75 = np.percentile(y,75)
print
print "25th percentile = %0.2f"%(perc_25)
print "50th percentile = %0.2f"%(perc_50)
print "75th percentile = %0.2f"%(perc_75)
print

# plot these percentile values as reference in the plot we generated in previous
# step
plt.axhline(perc_25, label='25th perc', c='r')
plt.axhline(perc_50, label='50th perc', c='g')
plt.axhline(perc_75, label='75th perc', c='m')
plt.legend(loc='best')

# look for outliers if any in the data by visual inspection
# remove outliers using a mask function
# remove outliers
y_masked = np.ma.masked_where(y==0, y)
# remove point 54 (an outlier)
y_masked = np.ma.masked_where(y_masked==54, y_masked)

# plot data again
plt.figure(2)
plt.title("Masked Data")
plt.plot(x, y_masked, 'ro')
plt.xlabel('year')
plt.ylabel('No Presendtial Request')
plt.ylim(0,60)

# Draw horizontal lines at 25, 50, and 75th percentile
plt.axhline(perc_25, label='25th perc', c='r')
plt.axhline(perc_50, label='50th perc', c='g')
plt.axhline(perc_75, label='75th perc', c='m')
plt.legend(loc='best')
plt.show()
