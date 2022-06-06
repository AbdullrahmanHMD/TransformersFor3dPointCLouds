import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
import os

def plot_pcu(vector):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xdata = vector[:, 0]
    ydata = vector[:, 1]
    zdata = vector[:, 2]
    
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()


def get_y_pred_truth(model, data_loader):
    device = get_device()

    y_pred = []
    y_truth = []

    model.eval()
    for point in data_loader:
        if point == None:
            continue
        
        x, y, _ = point
        x = x.view(1, x.shape[0], x.shape[1])
        x = x.to(device)
        y = y.to(device)
        
        yhat = model(x.float())
        
        _, label = torch.max(yhat, 1)
        y_pred.append(label.item())
        y_truth.append(y.item())

    return y_pred, y_truth

def get_device():
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    return device


def save_model(model, path):
    with open(path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)


def load_model_state_dict(path):
    path = os.path.join(path)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    
    return model_state_dict