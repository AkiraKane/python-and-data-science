import numpy as np
import matplotlib.pyplot as plt
import pdb

def simple_line_plot(x,y,figure_no):
    plt.figure(figure_no)
    plt.plot(x,y)
    plt.xlabel('xvalues')
    plt.ylabel('yvalues')
    plt.title('Simple Line')

def simple_dots(x,y,figure_no):
    plt.figure(figure_no)
    plt.plot(x,y,'or')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Simple Dots')

def simple_scatter(x,y,figure_no):
    plt.figure(figure_no)
    plt.scatter(x,y)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Simple Scatter')

def scatter_with_color(x,y,labels,figure_no):
    plt.figure(figure_no)
    plt.scatter(x,y,c=labels)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Scatter with color')

plt.close('all')
# Sample x y data for line and simple dot plots
x = np.arange(1,100, dtype=float)
y = np.array([np.power(b,2) for b in x])

figure_no = 1
simple_line_plot(x, y, figure_no)
figure_no += 1
simple_dots(x, y, figure_no)

# Sample x,y data for scatter plot
x = np.random.uniform(size=100)
y = np.random.uniform(size=100)

figure_no += 1
simple_scatter(x, y, figure_no)
figure_no += 1
label = np.random.randint(2, size=100)
scatter_with_color(x, y, label, figure_no)


# Gernerating Heat Maps
def x_y_axis_labeling(x, y, x_labels, y_labels, figure_no):
    plt.figure(figure_no)
    plt.plot(x, y, "+r")
    plt.margins(0.2)
    plt.xticks(x, x_labels, rotation='vertical')
    plt.yticks(y, y_labels)

def plot_heat_map(x, figure_no):
    plt.figure(figure_no)
    plt.pcolor(x)
    plt.colorbar()

plt.close('all')
x = np.array(range(1,6))
y = np.array(range(100, 600, 100))
x_label = ['element 1', 'element 2', 'element 3', 'element 4', 'element 5']
y_label = ['weight1', 'weight2', 'weight3', 'weight4', 'weight5']

x_y_axis_labeling(x, y, x_label, y_label, 1)
x = np.random.normal(loc=0.5, scale=0.2, size=(10,10))
plot_heat_map(x, 2)

plt.show()
