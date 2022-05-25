from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm 
import torch
import numpy as np
import os
import time

default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def train(model, optimizer, scheduler, train_loader, criterion, epochs, save_params=False, verbose=False, load_model=False):
    device = get_device()
    total_loss = []
    
    # Loading the model from checkpoint:
    if load_model:
        parameters = os.listdir(default_path)
        if len(parameters) > 0:
            last_param = parameters[-1]
            model.load_state_dict(get_model_state_dict(last_param))
            epoch_start = epochs - (epochs - len(parameters))
        else:
            epoch_start = 0
    else:
        epoch_start = 0
    
    steps = len(train_loader)
    model.train()

    for epoch in range(epoch_start, epochs):
        epoch_loss = 0
        epoch_tic = time.time()
        for point in tqdm(train_loader):
            if point == None:
                continue
            
            x, y, _ = point
            optimizer.zero_grad()
            
            x.to(device=device)
            y.to(device=device)

            yhat = model(x.float())
 
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
        
        # if verbose:
        print(f'epoch: {epoch} | loss: {epoch_loss}')
        epoch_toc = time.time()

        print(f'Epoch time: {epoch_toc - epoch_tic}')

        # scheduler = CosineAnnealingLR(optimizer, steps)
    if verbose:
        print(f'total loss: {total_loss}')

    print(f'Final loss {total_loss[-1]}')
    
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


def get_model_state_dict(param_name, path=default_path):
    path = os.path.join(path, param_name)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    return model_state_dict