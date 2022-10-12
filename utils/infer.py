import torch
import numpy as np
import pandas as pd

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

def force_magnitude_error(actual, pred):
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)
    
    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), torch.div(pred.T, p).T.view(-1, 3,1 )).view(-1)), torch.pi)

def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f=[]
    for size in data.size.numpy()*3:
        f.append(np.abs(forces[start:start+size].numpy()).max())
        start += size
    return f


def infer(loader, net):
    f_actual = []
    e_actual = []
    e_pred = []
    f_pred = []
    size = []

    for data in loader:

        # extract ground truth values
        e_actual += data.y.tolist()
        f_actual += get_max_forces(data, data.f)
        size += data.size.tolist()
        # get predicted values
        data.pos.requires_grad = True
        e = net(data)
        
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        e_pred += e.tolist()
        f_pred += get_max_forces(data, f)
        

    # return as dataframe
    return pd.DataFrame({'e_actual': e_actual, 'f_actual': f_actual, 'e_pred': e_pred, 'f_pred': f_pred, 'cluster_size': size})

