from data_loader import ModelNet40
import os

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


modelnet40_path = os.path.join(os.getcwd(), 'dataset_path.txt')
with open(modelnet40_path, 'r') as file:
    modelnet40_path = file.readline()


train_loader_fps = ModelNet40(dataset_path=modelnet40_path, test=False, sample='fps')
train_loader_uniform = ModelNet40(dataset_path=modelnet40_path, test=False, sample='uni')


data, label, label_txt = train_loader_fps[0]

print(f'The shape of the point cloud: {data.shape}')
print(f"Datapoint label: {label} | txt: {label_txt}")

plot_pcu(data)

data, _, _ = train_loader_uniform[0]

plot_pcu(data)


# NOTE: The below code is meant to test if all the images in the ModelNet40
# are working fine.


# errors = 0
# corrupted_data = []
# corrupted_data_indecies = []


# for i in range(len(train_loader)):
# # for i in range(50):
#     try:
#         train_loader[i]
#     except Exception as e:
#         errors += 1
#         print(f'WARNING: {train_loader.data_points_paths[i][0]} has errored out')
#         corrupted_data.append(train_loader.data_points_paths[i][0])
#         corrupted_data_indecies.append(i)


# print(f'Number of corrupted datapoints {errors}')

# file_name = "corrupted_images.txt"
# file_path = os.path.join(os.getcwd(), file_name)
# with open(file_path, 'w') as file:
#     for point in corrupted_data:
#         file.write(point)
#         file.write('\n')
        
# file_name = "corrupted_images_indecies.txt"
# file_path = os.path.join(os.getcwd(), file_name)
# with open(file_path, 'w') as file:
#     for point in corrupted_data_indecies:
#         file.write(str(point))
#         file.write('\n')