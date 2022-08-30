# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import torch
#import poptorch
from copy import deepcopy as copy
from pathlib import Path

class PrepackedDataset(torch.utils.data.Dataset):
    def __init__(self, loader, opt, cached_data, shuffle=True):
        self.dataset = []
        self.cached_data = cached_data
        self.num_workers = opt.num_workers
        self.shuffle = shuffle
        self.water_dataset = opt.water_dataset
        self.test_data = []
        directory = f"data/cached_dataset/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.file_name_conf = f"{directory}saved_data_{opt.batch_size}_{opt.device_iterations}_{opt.replication_factor}_{opt.gradient_accumulation}_"
        self.water_dataset = opt.water_dataset
        if not self.cached_data:
            print("Data saved in disk mode")
            if Path(self.file_name_conf + "0_0.pt").exists():
                items_per_batch = 5
                for idx_batch in range(len(loader)):
                    batch = []
                    for idx in range(items_per_batch):
                        file_name = f"{self.file_name_conf}{idx}_{idx_batch}.pt"
                        batch.append(file_name)
                    self.dataset.append(batch)
            else:
                for idx_batch, data in enumerate(loader):
                    batch = []
                    for idx, data_piece in enumerate(data):
                        file_name = f"{self.file_name_conf}{idx}_{idx_batch}.pt"
                        print("Saving file " + file_name + "...")
                        torch.save(data_piece, file_name)
                        print("File " + file_name + " SAVED!")
                        batch.append(file_name)
                    self.dataset.append(batch)
        else:
            file_name_z = f"{self.file_name_conf}_z.pt"
            file_name_edge_weight = f"{self.file_name_conf}_edge_weight.pt"
            file_name_edge_index = f"{self.file_name_conf}_edge_index.pt"
            file_name_batch = f"{self.file_name_conf}_batch.pt"
            file_name_energy_target = f"{self.file_name_conf}_energy_target.pt"
            if Path(file_name_z).exists():
                self.z = torch.load(file_name_z)
                self.edge_weight = torch.load(file_name_edge_weight)
                self.edge_index = torch.load(file_name_edge_index)
                self.batch = torch.load(file_name_batch)
                self.energy_target = torch.load(file_name_energy_target)
            else:
                z, edge_weight, edge_index, batch, energy_target = [], [], [], [], []
                for data in loader:
                    if len(z) < 10:
                        self.test_data.append(copy(data))
                    z.append(copy(data[0]).to(torch.uint8))
                    edge_weight.append(copy(data[1]))
                    edge_index.append(copy(data[2]).to(torch.int32))
                    batch.append(copy(data[3]).to(torch.uint8))
                    energy_target.append(copy(data[4]))
                self.z = torch.stack(z, dim=0)
                self.edge_weight = torch.stack(edge_weight, dim=0)
                self.edge_index = torch.stack(edge_index, dim=0)
                self.batch = torch.stack(batch, dim=0)
                self.energy_target = torch.stack(energy_target, dim=0)

                print("Saving file " + file_name_z + "...")
                torch.save(self.z, file_name_z)
                print("File " + file_name_z + " SAVED!")

                print("Saving file " + file_name_edge_weight + "...")
                torch.save(self.edge_weight, file_name_edge_weight)
                print("File " + file_name_edge_index + " SAVED!")

                print("Saving file " + file_name_edge_index + "...")
                torch.save(self.edge_index, file_name_edge_index)
                print("File " + file_name_edge_index + " SAVED!")

                print("Saving file " + file_name_batch + "...")
                torch.save(self.batch, file_name_batch)
                print("File " + file_name_batch + " SAVED!")

                print("Saving file " + file_name_energy_target + "...")
                torch.save(self.energy_target, file_name_energy_target)
                print("File " + file_name_energy_target + " SAVED!")

    def __len__(self):
        if self.cached_data:
            return len(self.z)
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        if self.cached_data:
            return self.z[index], self.edge_weight[index], self.edge_index[index], self.batch[index], self.energy_target[index]
        else:
            item = [torch.load(x) for x in self.dataset[index]]
            return item

    def get_dataloader(self):
        persistent_workers = True
        num_workers = self.num_workers
        mode = poptorch.DataLoaderMode.Async
        
        async_options = {
            "sharing_strategy": poptorch.SharingStrategy.SharedMemory,
            "early_preload": True,
            "buffer_size": self.num_workers,
            "load_indefinitely": True,
            "miss_sleep_time_in_ms": 0
        }
        
        if 'all' in self.water_dataset:
            persistent_workers = False
            num_workers = 0
            mode = poptorch.DataLoader.Sync
            async_options = None
        


        return poptorch.DataLoader(options=poptorch.Options(),
                                dataset=self,
                                shuffle=self.shuffle,
                                num_workers=num_workers,
                                persistent_workers= persistent_workers,
                                mode=mode,
                                async_options=async_options)
