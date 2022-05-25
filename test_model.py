from model import *
from data.dataset import ModelNet40
import torch
import os


in_features = 3
feature_dim = 128
out_featuers = 1024
k_size = 128
num_classes = 40

modelnet40_path = os.path.join(os.getcwd(), os.path.join('data', 'dataset_path.txt'))
with open(modelnet40_path, 'r') as file:
    modelnet40_path = file.readline()
    
data_loader = ModelNet40(dataset_path=modelnet40_path, test=False, sampling='fps')

model = PointCloudClassifier(in_features, feature_dim, out_featuers, k_size, num_classes)

data, label, label_txt = data_loader[0]

forward = model(data.float())
print(forward.shape)
print(torch.sum(forward).item())
print(forward)