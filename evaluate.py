import torch
import os
from tqdm import tqdm

# The path for the saved parameters.
default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')


def evaluate(model, test_loader):
    
    device = get_device()
    model.eval()
    
    accuracy = 0
    num_correct = 0
    model = model.to(device)
    for x, y, _ in test_loader:
            
        x = x.to(device)
        y = y.to(device)
            
        yhat = model(x)
        _, label = torch.max(yhat, 1)
        num_correct += (y == label).sum().item()

    accuracy = 100 * num_correct / (len(test_loader) * test_loader.batch_size)
        
    return accuracy


# def evaluate(model, validation_loader, epochs, verbose=False):
#     model.eval()
#     total_accuracy = []
#     for epoch in range(1, epochs + 1):
#         model.load_state_dict(get_model_state_dict(f'param_epoch_{epoch}'))
#         correct = 0

#         for point in validation_loader:
#             if point == None:
#                 continue
            
#             x, y, _ = point
#             x = torch.squeeze(x, 0)
#             yhat = model(x.float()).view(1, -1)
            
#             _, label = torch.max(yhat, 1)
#             correct += (y == label).sum().item()

#         # accuracy = 100 * correct / (len(validation_loader) * validation_loader.batch_size)
#         accuracy = 100 * correct / len(validation_loader)
#         total_accuracy.append(accuracy)
#         if verbose:
#             print(f'Accuracy: {accuracy:.2f}%\t |\tEpoch: {epoch}')
#     return total_accuracy

def eval_2(model, test_loader):
    
    device = get_device()
    model.eval()
    total_accuracy = []

    accuracy = 0
    num_correct = 0
    for x, y, _ in tqdm(test_loader):
            
        x.to(device)
        y.to(device)
            
        yhat = model(x.float())
        _, label = torch.max(yhat, 1)
        num_correct += (y == label).sum().item()

        accuracy = 100 * num_correct / (len(test_loader) * test_loader.batch_size)
        total_accuracy.append(accuracy)
        
    return accuracy

def get_model_state_dict(param_name, path=default_path):
    path = os.path.join(path, param_name)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    return model_state_dict

def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device
