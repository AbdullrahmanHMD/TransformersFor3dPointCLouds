import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_pcu(vector):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xdata = vector[:, 0]
    ydata = vector[:, 1]
    zdata = vector[:, 2]
    
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()
    

def plot_pcu_(vectors):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i, vector in enumerate(vectors):
        xdata = vector[:, 0]
        ydata = vector[:, 1]
        zdata = vector[:, 2]
        
        ax = fig.add_subplot(130 + (i + 1))
        ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
        
        
    plt.show()