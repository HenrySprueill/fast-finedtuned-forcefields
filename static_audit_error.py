import os
import os.path as op
import numpy as np
import pandas as pd
import json
import argparse
import torch
from torch_geometric.data import DataLoader
from datetime import datetime

from utils.infer import force_magnitude_error, force_angular_error
from utils.audit import load_static_model
from utils.water_dataset import PrepackedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Path to trained model to analyze.')
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset to analyze.')
parser.add_argument('--split', required=True, type=str, help='Path to split file to analyze.')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save results.')
input_args = parser.parse_args()

# get trained model
net = load_static_model(input_args.model, device='cpu')
modeldir = '_'.join(input_args.model.split('/')[:-2])

# load dataset
dataset = PrepackedDataset(None, 
                           input_args.split, 
                           input_args.dataset.split('/')[-1].replace('_data.hdf5',''), 
                           directory='/'.join(input_args.dataset.split('/')[:-1]))


test_data=dataset.load_data('test')
print('{len(test_data)} items in dataset')

loader = DataLoader(test_data, batch_size=256, shuffle=False)



fme_all = np.array(())
fae_all = np.array(())

df = pd.DataFrame()
print(datetime.now().isoformat(timespec='seconds'))
for data in loader:
    # get predicted values
    #data.pos.requires_grad=True
    e = net(data)
    
    f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
    
    # force errors 
    fme  = force_magnitude_error(data.f, f).numpy()
    fae  = force_angular_error(data.f, f).numpy()
    
    fme_all = np.concatenate((fme_all, fme))
    fme_all = np.concatenate((fae_all, fae))
    
    # get means of individual samples
    start = 0
    fme_individual = []
    fae_individual = []
    for size in data.size.numpy()*3:
        fme_individual.append(np.mean(np.abs(fme[start:start+size])))
        fae_individual.append(np.mean(np.abs(fae[start:start+size])))
        start += size
    
    tmp = pd.DataFrame({'size': size, 
                        'e_actual': data.y.numpy(), 'e_pred': e.detach().numpy(),
                        'fme_mae':fme_individual, 'fae_mae':fae_individual})
    df = pd.concat([df, tmp])

print(datetime.now().isoformat(timespec='seconds'))

df['e_diff']=df['e_actual']-df['e_pred']
df['e_mae']=df['e_diff'].apply(lambda x: np.mean(np.abs(x)))
df['e_water'] = df['e_actual']/df['size']

df.to_csv(op.join(input_args.savedir,f"{modeldir}-error.csv"), index=False)
print(f"saved to {input_args.savedir}/{modeldir}-error.csv")

