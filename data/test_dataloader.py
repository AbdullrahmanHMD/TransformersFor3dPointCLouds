from dataset import ModelNet40, collate_fn, arr
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

    
modelnet40_path = os.path.join(os.getcwd(), 'dataset_path.txt')
with open(modelnet40_path, 'r') as file:
    modelnet40_path = file.readline()


model_net = ModelNet40(dataset_path=modelnet40_path, test=False, sample_size=512, sampling='fps')
train_loader_fps = DataLoader(model_net, batch_size=1, shuffle=True, collate_fn=collate_fn )

# for x, y, _ in train_loader_fps:
    # print(y)
    
# for a in arr:
    # print(a)
# train_loader_uniform = ModelNet40(dataset_path=modelnet40_path, test=False, sampling='uni-sph')

index = model_net.class_indicies_distribution()[0][0]

data, _, _ = model_net[index]


print(data.size(), flush=True)
plot_pcu(data)


# for x, y, _ in train_loader_fps:
    # print(y)

# print(train_loader_fps[:10])


# data, label, label_txt = train_loader_fps[0]

# print(f'The shape of the point cloud: {data.shape}')
# print(f"Datapoint label: {label} | txt: {label_txt}")

# plot_pcu(data)

# data, _, _ = train_loader_uniform[0]

# plot_pcu(data)


# NOTE: The below code is meant to test if all the images in the ModelNet40
# are working fine.


# errors = 0
# corrupted_data = []
# corrupted_data_indecies = []


# for i in range(len(train_loader_fps)):
# # for i in range(50):
#     try:
#         train_loader_fps[i]
#     except Exception as e:
#         errors += 1
#         print(f'WARNING: {train_loader_fps.data_points_paths[i][0]} has errored out')
#         corrupted_data.append(train_loader_fps.data_points_paths[i][0])
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