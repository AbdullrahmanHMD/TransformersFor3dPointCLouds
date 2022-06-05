from dataset_normalized import NormalizedModelNet40
import os

from torch.utils.data import DataLoader
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# -------------------------------------------------------------------------------------------

def plot_pcu(vector):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xdata = vector[:, 0]
    ydata = vector[:, 1]
    zdata = vector[:, 2]
    
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()
    
path = 'C:\\Users\\abooo\\Desktop\\new_dataset\\modelnet40_ply_hdf5_2048'
dataset = NormalizedModelNet40(path, sample_size=1500, sampling_method='fps')
    

x, y, y_txt = dataset[8]


print(dataset.class_distribution(), flush=True)

print(y, flush=True)
print(y_txt, flush=True)

print()
plot_pcu(x)

