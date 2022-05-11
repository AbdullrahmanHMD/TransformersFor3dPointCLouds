from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm 
import torch
import numpy as np

def train(model, optimizer, scheduler, train_loader, criterion, epochs, verbose=False):
    
    total_loss = []
    steps = len(train_loader)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y, _ in train_loader:
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