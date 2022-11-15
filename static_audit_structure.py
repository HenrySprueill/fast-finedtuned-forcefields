import os.path as op
import os
import pandas as pd
import numpy as np
import h5py
from ase import Atoms 
import argparse
from multiprocessing import Pool
from utils.graph import get_structure_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Path to trained model to analyze.')
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset to analyze.')
parser.add_argument('--split', required=True, type=str, help='Path to split file to analyze.')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save results.')
input_args = parser.parse_args()

# set up save name 
modeldir = '_'.join(input_args.model.split('/')[:-2])

# load dataset
dataset = h5py.File(input_args.dataset, "r")

# load test split
S = np.load(input_args.split)
mode_idx = S[f'test_idx']

def run_metrics(index, dataset=dataset):
    cluster_size = dataset["size"][index][0]

    z = dataset["z"][index][:cluster_size*3]
    pos = dataset["pos"][index][:cluster_size*3]

    ## make into ASE Atoms object
    cluster = Atoms(numbers=z, positions=pos)

    ## get structure metrics
    dat = get_structure_metrics(cluster)

    return dat

def get_means(dat):
    # return means
    return dat.groupby('type').mean().T.reset_index(drop=True)

def get_std(dat):
    # return means
    return dat.groupby('type').std().T.reset_index(drop=True)

def get_max(dat):
    # return means
    return dat.groupby('type').max().T.reset_index(drop=True)

def get_min(dat):
    # return means
    return dat.groupby('type').min().T.reset_index(drop=True)
    

with Pool(12) as p:
    d = p.map(run_metrics, mode_idx)
    d_means = p.map(get_means, d)
    d_std = p.map(get_std, d)
    d_max = p.map(get_max, d)
    d_min = p.map(get_min, d)


label = ['mean','std','max','min']
for i,d in enumerate([d_means, d_std, d_max, d_min]):
    df = pd.concat(d, sort=False, ignore_index=True)
    df.to_csv(op.join(input_args.savedir,f"{modeldir}-structure_{label[i]}.csv"), index=False)
    print(f"saved to {input_args.savedir}/{modeldir}-structure_{label[i]}.csv")





