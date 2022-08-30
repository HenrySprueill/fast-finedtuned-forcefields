#Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import torch
#import poptorch
import numpy as np
from copy import deepcopy as copy
from pathlib import Path
from tqdm import tqdm
#import popdist.poptorch
from torch_geometric.data import DataLoader
import h5py

class PrepackedDataset(torch.utils.data.Dataset):
    def __init__(self, loader, opt, shuffle=True, mode="train", directory="data/cached_dataset/"):
        self.dataset = []
        self.num_workers = opt.num_workers
        self.shuffle = shuffle
        self.directory = directory
        self.mode = mode

        if not os.path.exists(directory):
            os.makedirs(directory)

        if loader is None:
            self.load_data()
        else:
            self.create_container(loader)
            print("Finishing processing...")
            bar = tqdm(enumerate(loader), total=len(loader))
            for idx, data in bar:
                self.z[idx] = copy(data[0]).to(torch.uint8)
                self.edge_weight[idx] = copy(data[1])
                self.edge_index[idx] = copy(data[2]).to(torch.int32)
                self.batch[idx] = copy(data[3]).to(torch.uint8)
                self.energy_target[idx] = copy(data[4])

            self.save_data()

    def create_container(self, loader):
        tmp = next(iter(loader))
        n_elements = len(loader)
        z, e_w, e_i, b, e_t = tmp
        z_size, e_w_size = list(z.size()), list(e_w.size())
        e_i_size, b_size = list(e_i.size()), list(b.size())
        e_t_size = list(e_t.size())

        z_size.insert(0, n_elements)
        e_w_size.insert(0, n_elements)
        e_i_size.insert(0, n_elements)
        b_size.insert(0, n_elements)
        e_t_size.insert(0, n_elements)

        self.z = np.zeros(z_size, dtype=np.uint8)
        self.edge_weight = np.zeros(e_w_size)
        self.edge_index = np.zeros(e_i_size, dtype=np.int32)
        self.batch = np.zeros(b_size, dtype=np.uint8)
        self.energy_target = np.zeros(e_t_size)


    def save_data(self):
        print("Saving cached data in disk...")
        dataset = h5py.File(f"{self.directory}{self.mode}_data.hdf5", "w")
        dataset.create_dataset("z", dtype=np.uint8, data=self.z)
        dataset.create_dataset("edge_weight", dtype=np.float32, data=self.edge_weight)
        dataset.create_dataset("edge_index", dtype=np.int32, data=self.edge_index)
        dataset.create_dataset("batch", dtype=np.uint8, data=self.batch)
        dataset.create_dataset("energy_target", dtype=np.float32, data=self.energy_target)
        dataset.close()

    def load_data(self):
        print("Loading cached data from disk...")
        dataset = h5py.File(f"{self.directory}{self.mode}_data.hdf5", "r")

        if False: #popdist.isPopdistEnvSet():
            self.dataset_size = len(dataset["z"]) // int(popdist.getNumTotalReplicas())
            self.read_offset = popdist.getInstanceIndex() * self.dataset_size
        else:
            self.read_offset = 0
            self.dataset_size = len(dataset["z"])

        self.z, self.edge_weight, self.edge_index, self.batch, self.energy_target = [], [], [], [], []
        for i in range(self.read_offset, self.read_offset+self.dataset_size):
            self.z.append(torch.from_numpy(dataset["z"][i]))
            self.edge_weight.append(torch.from_numpy(dataset["edge_weight"][i]))
            self.edge_index.append(torch.from_numpy(dataset["edge_index"][i]))
            self.batch.append(torch.from_numpy(dataset["batch"][i]))
            self.energy_target.append(torch.from_numpy(dataset["energy_target"][i]))

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return self.z[index], self.edge_weight[index], self.edge_index[index], self.batch[index], self.energy_target[index]

    def get_dataloader(self, options):

        """
        # Not sure we actually need to use this loader
        return poptorch.DataLoader(options,self,
                                batch_size=1,
                                num_workers=self.num_workers,
                                shuffle=self.shuffle,
                                mode=poptorch.DataLoaderMode.Sync,
                                auto_distributed_partitioning=False)
        """
        # TODO: if we do need this, change "data" to the actual dataset
        return DataLoader(data, batch_size=1, shuffle=self.shuffle)
