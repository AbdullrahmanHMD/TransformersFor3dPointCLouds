from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm 
import torch
import numpy as np
import os
import time
from evaluate import eval_2

default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def train(model, optimizer, scheduler, train_loader, validation_loader, criterion, epochs,
          save_params=False, verbose=False, load_model=False):
            
    device = get_device()
    print(f'Current device: {device}')
    
    epoch_times = []
    total_loss = []    
    model.to(device)
    model.train()
    
    accuracies_validation = []
    accuracies_train = []
    steps = 10
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_tic = time.time()
        num_correct = 0
        
        for point in tqdm(train_loader):
            
            if point == None:
                continue

            optimizer.zero_grad()
            x, y, _ = point
            
            x = x.to(device)
            y = y.to(device)
            
            yhat = model(x.float())
            
            loss = criterion(yhat, y)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
    
            _, label = torch.max(yhat, 1)
            num_correct += (y == label).sum().item()
            
        total_loss.append(epoch_loss)
        
        mean_loss = sum(total_loss) / len(total_loss)
        scheduler.step(mean_loss)      
        
        print('Evaluating epoch...', flush=True)
        train_accuracy = 100 * num_correct / (len(train_loader) * train_loader.batch_size)
        test_accuracy = eval_2(model, validation_loader)

        # if scheduler != None:
            # lr = optimizer.param_groups[0]['lr']
            # print(f'Learning rate: {lr}')
            # scheduler.step()

        accuracies_train.append(train_accuracy)
        accuracies_validation.append(test_accuracy)
        
        epoch_toc = time.time()
        epoch_time = epoch_toc - epoch_tic
        epoch_times.append(epoch_time)
        epoch_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch} | Loss: {epoch_loss:.2f} | Train acc: {train_accuracy:.2f}' \
              f' | Val acc: {test_accuracy:.2f} | lr: {epoch_lr} | Runtime: {epoch_time:.2f} seconds')
    
    return total_loss, epoch_times, accuracies_train, accuracies_validation
    

def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
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