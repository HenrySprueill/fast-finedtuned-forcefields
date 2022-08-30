# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import torch
import torch.nn.functional as F
#import poptorch

from torch_geometric.data import Data, Batch, Dataset
from typing import Optional, List, Tuple, Union
from data_utils import data_keys
from packing import PackingStrategy
from itertools import chain


class TupleCollater(object):
    """
    Collate a PyG Batch as a tuple of tensors
    """
    def __init__(self,
                 include_keys: Optional[Union[List[str], Tuple[str]]] = None):
        """
        :param include_keys (optional): Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the tuple. By default will
            extract the keys required for training the SchNet model.
        """
        super().__init__()
        self.include_keys = include_keys

        if self.include_keys is None:
            # Use the defaults + the "batch" vector with the same ordering as
            # the forward method of the SchNet model.
            keys = list(data_keys())
            keys.insert(-1, "batch")
            self.include_keys = tuple(keys)

        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "__call__")

    def __call__(self, data_list):
        with poptorch.profiling.Channel("Batch").tracepoint("from_data_list"):
            batch = Batch.from_data_list(data_list)

        assert all(
            [hasattr(batch, k) for k in self.include_keys]
        ), f"Batch is missing a required key: include_keys='{self.include_keys}'"
        return tuple(getattr(batch, k) for k in self.include_keys)


class CombinedBatchingCollater(object):
    """ Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """
    def __init__(self,
                 mini_batch_size: int,
                 is_packed: bool = False,
                 graphs_per_package: Optional[int] = None):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        :param is_packed (bool): Flag to configure the collater to unpack a
            packed dataset (default: False)
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.is_packed = is_packed
        self.graphs_per_package = graphs_per_package
        self.tuple_collate = TupleCollater()
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "_prepare_package", "__call__")

    def _prepare_package(self, packages):
        num_packages = len(packages)
        total_num_graphs = num_packages * self.graphs_per_package
        graphs = list(chain.from_iterable(packages))
        num_graphs = len(graphs)

        if num_graphs < total_num_graphs:
            last = graphs[-1]
            last.y = F.pad(last.y, (0, total_num_graphs - num_graphs))
            graphs[-1] = last

        return graphs

    def __call__(self, batch):
        num_items = len(batch)
        assert num_items % self.mini_batch_size == 0, "Invalid batch size. " \
            f"Got {num_items} graphs and mini_batch_size={self.mini_batch_size}."

        num_mini_batches = num_items // self.mini_batch_size
        batches = [None] * num_mini_batches
        start = 0
        stride = self.mini_batch_size

        for i in range(num_mini_batches):
            slices = batch[start:start + stride]

            if self.is_packed:
                slices = self._prepare_package(slices)

            batches[i] = self.tuple_collate(slices)
            start += stride

        num_outputs = len(batches[0])
        outputs = [None] * num_outputs

        for i in range(num_outputs):
            outputs[i] = torch.stack(tuple(item[i] for item in batches))

        return tuple(outputs)


class PackingCollater(object):
    """
    Collater object for packed datasets

    This collater is expected to be used along with the k-nearest neighbors
    method for determining the graph edges.
    """
    def __init__(self, package_batch_size: int, strategy: PackingStrategy,
                 k: int):
        """
        :param package_batch_size (int): The number of packages to be processed
            in a single step.
        :param strategy (PackingStrategy): The packing strategy used to pack the
            dataset.
        :param k (int): The number of nearest neighbors used in building the
            graphs.
        """
        super().__init__()
        self.package_batch_size = package_batch_size
        self.strategy = strategy
        self.k = k
        self.collater = CombinedBatchingCollater(
            self.package_batch_size,
            is_packed=True,
            graphs_per_package=strategy.max_num_graphs())

        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "key_slice", "_prepare_package", "__call__")

    @staticmethod
    def key_slice(data, keys=data_keys()):
        """Slices the Data instance to only include the provided keys"""
        if "num_nodes" not in keys:
            # Include num_nodes to prevent warnings with async loader
            keys = keys + ("num_nodes", )
        values = [getattr(data, k) for k in keys]
        kwargs = dict([*zip(keys, values)])
        return Data(**kwargs)

    def _prepare_package(self, package):
        """Prepares each package by padding any incomplete data tensors"""
        num_nodes = 0

        for i, graph in enumerate(package):
            num_nodes += graph.num_nodes
            graph = self.key_slice(graph)
            graph.y = graph.y.view(-1)
            package[i] = graph

        assert num_nodes <= self.strategy.max_num_nodes, \
            f"Too many nodes in package. Package contains {num_nodes} nodes "\
            f"and maximum is {self.strategy.max_num_nodes}."

        num_filler_nodes = self.strategy.max_num_nodes - num_nodes

        if num_filler_nodes > 0:
            filler_mol = create_packing_molecule(num_filler_nodes, self.k)
            filler_mol.y = filler_mol.y.view(-1)
            package.append(filler_mol)

        num_graphs = len(package)
        max_num_graphs = self.strategy.max_num_graphs()
        assert num_graphs <= max_num_graphs, \
            f"Too many graphs in package. Package contains {num_graphs} "\
            f"graphs and maximum is {max_num_graphs}."

        return package

    def __call__(self, data_list):
        num_packages = len(data_list)
        assert num_packages % self.package_batch_size == 0
        data_list = [self._prepare_package(p) for p in data_list]
        assert all(
            sum(g.num_nodes for g in p) == self.strategy.max_num_nodes
            for p in data_list)
        return self.collater(data_list)


def create_dataloader(dataset: Dataset,
                      ipu_opts: Optional[poptorch.Options] = None,
                      use_packing: bool = False,
                      batch_size: int = 1,
                      shuffle: bool = False,
                      k: Optional[int] = None,
                      num_workers: int = 0):
    """
    Creates a data loader for graph datasets

    Applies the mini-batching method of concatenating multiple graphs into a
    single graph with multiple disconnected subgraphs. See:

    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html
    """
    if ipu_opts is None:
        ipu_opts = poptorch.Options()

    if use_packing:
        collater = PackingCollater(batch_size, dataset.strategy, k)
    else:
        collater = CombinedBatchingCollater(batch_size)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               persistent_workers=True,
                               mode=poptorch.DataLoaderMode.Sync,
                               collate_fn=collater)


def create_packing_molecule(num_atoms: int, k: int = 8):
    """
    Creates a packing molecule

    A non-interacting molecule that is used to fill incomplete batches when
    using packed datasets.

    :param num_atoms (int): The number of atoms in the packing molecule.
    :param k (int): Number of neighbors used to build the edges (default: 8).

    :returns: Data instance
    """
    z = torch.zeros(num_atoms, dtype=torch.int64)
    num_edges = k * num_atoms
    edge_attr = torch.zeros(num_edges)
    edge_index = torch.zeros(2, num_edges, dtype=torch.int64)
    y = torch.tensor(0.0)
    return Data(z=z,
                edge_attr=edge_attr,
                edge_index=edge_index,
                y=y,
                num_nodes=num_atoms)
