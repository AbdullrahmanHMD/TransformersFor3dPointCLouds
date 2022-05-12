from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm 
import torch
import numpy as np
import os

default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def train(model, optimizer, scheduler, train_loader, criterion, epochs, save_params=False, verbose=False):
    device = get_device()
    total_loss = []
    
    steps = len(train_loader)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for point in train_loader:
            if point == None:
                continue
            
            x, y, _ = point
            optimizer.zero_grad()
            
            yhat = model(x.float())
            yhat = yhat.view(1, -1)
            y = y.view(-1)
            
            loss = criterion(yhat, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose:
                print(f'learning rate: {scheduler.get_lr()[0]}')
            
            
        total_loss.append(epoch_loss)
        if save_params:
            export_parameters(model, f'param_epoch_{epoch}')
        
        if verbose:
            print(f'epoch: {epoch} | loss: {epoch_loss}')
            
        scheduler = CosineAnnealingLR(optimizer, steps)
    
    if verbose:
        print(f'total loss: {total_loss}')
    
    return np.array(total_loss)


def get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def export_parameters(model, param_name, path=default_path):
    path = os.path.join(path, param_name)
    with open(path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)
