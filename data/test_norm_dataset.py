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
dataset = NormalizedModelNet40(path, sample_size=512, sampling_method='fps')
    
    

print(dataset.augmentation_count())
print(dataset.class_distribution())
print(dataset.adaptive_class_weights())

# class_num = 32
# class_indicies = dataset.class_indicies(class_num)

# random_ind = np.random.choice(class_indicies, 5)

# for ind in random_ind:
#     x, y, y_txt = dataset[ind]
#     print(y_txt, flush=True)
#     plot_pcu(x)
# x, y, y_txt = dataset[423]
# for ind in class_indicies:
    # x, y, y_txt = dataset[ind]
    # print(f'class: {y_txt} | class label: {y}')

# print(dataset.class_distribution(), flush=True)

# print(y, flush=True)
# print(y_txt, flush=True)

# print()
# plot_pcu(x)

