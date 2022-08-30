# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os.path as osp
from typing import Optional, Tuple

#import poptorch
import torch
from torch_geometric.datasets import QM9
from tqdm import tqdm
#from jsonargparse import namespace_to_dict
from data_utils import dataroot
from dataloader import create_dataloader
import packing
from transforms import create_transform
import water_dataset


def calculate_splits(splits: Tuple[float], num_examples: int):
    """
    Converts the tuple describing dataset splits as a fraction of the full
    dataset into the number of examples to include in each split.

    :param splits (tuple of float): A tuple describing the ratios to
        divide the full dataset into training, validation, and test splits.
    :param num_examples (int): The total number of examples in the full dataset
    """
    assert isinstance(splits, (list, tuple)) and \
        all(isinstance(s, float) for s in splits) and \
        len(splits) == 3 and sum(splits) == 1., \
        f"Invalid splits {splits}. Must be a tuple or list containing " \
        "exactly three floats that add up to 1.0."

    splits = torch.tensor(splits)
    splits = torch.round(splits * num_examples).long()
    splits = torch.cumsum(splits, 0)
    return tuple(splits.tolist())


class DatasetFactory:
    """
    DatasetFactory

    Abstract interface for managing the reproducible application of dataset
    transforms and randomized splits. Sub-classes must implement the following:

        * create_dataset: materialize the full dataset
        * strategy: the pre-computed dataset packing strategy

    Typical usage:

        factory = ConcreteFactory(...)
        loader = factory.dataloader(split="train", num_workers=4)
        ...
    """
    def __init__(self,
                 root: Optional[str] = "data",
                 splits: Optional[Tuple[float,...]] = None,
                 batch_size: int = 1,
                 use_packing: bool = False,
                 k: int = 8,
                 cutoff: float = 30.0,
                 max_num_atoms: int = 32,
                 use_qm9_energy: bool = True,
                 use_standardized_energy: bool = True,
                 options: Optional[poptorch.Options] = None,
                 ):
        """
        Dataset Factory parameters.

        :param root: The root folder name for storing the dataset.
        :param splits: A tuple describing the ratios
            to divide the full dataset into training, validation, and test
            datasets (default: (0.8, 0.1, 0.1)).
        :param batch_size: The data loader batchsize (default: 1)
        :param use_packing: Use packing to minimise padding values
            (default: False).
        :param k: Number of neighbors to use for building the k-nearest
            neighbors graph when using packing (default: 8).
        :param cutoff: Cutoff distance of interatomc interactions in
            Angstroms (default: 30.0)
        :param max_num_atoms: The maximum number of atoms used by the
            PadMolecule transform (default: 32).
        :param options: Instance of
            poptorch.Options
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.model_batch_size = self.batch_size
        self.splits = (0.8, 0.1, 0.1) if splits is None else splits
        self.use_packing = use_packing
        mode = "knn" if self.use_packing else "radius"
        self.transform = create_transform(mode=mode,
                                          k=k,
                                          cutoff=cutoff,
                                          max_num_atoms=max_num_atoms,
                                          use_padding=True,
                                          use_qm9_energy=use_qm9_energy,
                                          use_standardized_energy=use_standardized_energy)
        self.k = k

        if self.use_packing:
            self.model_batch_size = self.strategy.max_num_graphs(
            ) * self.batch_size

        self.options = options
        self.reset()

        # Cache the RNG state for reproducible random shuffle
        self._rng_state = torch.get_rng_state()
        dataset = self.create_dataset()

        if not self.use_packing:
            return

        # Cache the num_nodes tensor for each example in the dataset for packing
        self._num_nodes_file = osp.join(dataset.processed_dir, "num_nodes.pt")

        if osp.exists(self._num_nodes_file):
            return

        bar = tqdm(dataset, desc="Calculating number of nodes")
        num_nodes = [g.num_nodes for g in bar]
        num_nodes = torch.tensor(num_nodes)
        torch.save(num_nodes, self._num_nodes_file)

    def reset(self):
        """
        resets the state of the factory.

        This method is necessary to ensure that the factory does not maintain a
        persistent reference to a fully materialized dataset.
        """
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def create_dataset(self):
        """
        Abstract method for creating the full dataset used by the factory.
        """
        raise NotImplementedError

    @property
    def strategy(self):
        """
        Abstract property for storing the packing strategy used by the factory.
        """
        raise NotImplementedError

    def setup_splits(self):
        """
        Setup dataset splits by randomly assigning examples to the training,
        validation, or test splits.
        """
        # Restore the RNG state for reproducible random shuffle
        torch.set_rng_state(self._rng_state)
        dataset = self.create_dataset()
        dataset, perm = dataset.shuffle(return_perm=True)

        if self.use_packing:
            num_nodes = torch.load(self._num_nodes_file)
            num_nodes = num_nodes[perm]
            dataset = packing.PackedDataset(dataset,
                                    self.strategy,
                                    num_nodes,
                                    shuffle=True)
        splits = calculate_splits(self.splits, len(dataset))
        self._train_dataset = dataset[0:splits[0]]
        self._val_dataset = dataset[splits[0]:splits[1]]
        self._test_dataset = dataset[splits[1]:]

    @property
    def train_dataset(self):
        """
        Training split of the dataset
        """
        if self._train_dataset is None:
            self.setup_splits()
        return self._train_dataset

    @property
    def val_dataset(self):
        """
        Validation split of the dataset
        """
        if self._val_dataset is None:
            self.setup_splits()
        return self._val_dataset

    @property
    def test_dataset(self):
        """
        Test split of the dataset
        """
        if self._test_dataset is None:
            self.setup_splits()
        return self._test_dataset

    def dataloader(self, split: str = "train", num_workers: int = 4):
        """
        Create a dataloader for the desired split of the dataset.

        :param split (str): The desired dataset split to load (default: "train")
        :param num_workers (int): number of asynchronous workers used by the
            dataloader (default: 4).
        """
        dataset = LazyDataset(self, split)
        shuffle = True if split == "train" else False
        # calls create_dataloader in dataloader.py
        return create_dataloader(dataset,
                                 self.options,
                                 use_packing=self.use_packing,
                                 batch_size=self.batch_size,
                                 shuffle=shuffle,
                                 k=self.k,
                                 num_workers=num_workers)

class TestWaterDatasetFactory(DatasetFactory):
    def create_dataset(self):
        return water_dataset.TestWaterDataSet(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        return packing.water_test_packing_strategy()

    @staticmethod
    def create(cfg):
        suffix = f"packed_k{cfg.k}" if cfg.use_packing else f"padded"
        if cfg.root is None:
            cfg.root = dataroot(f"water_test_{suffix}")
        return TestWaterDatasetFactory(**namespace_to_dict(cfg))


class TrainWaterDatasetFactory(DatasetFactory):
    def create_dataset(self):
        return water_dataset.TrainWaterDataSet(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        """
        The pre-computed packing strategy for the QM9 dataset.
        """
        return packing.water_train_packing_strategy()

    @staticmethod
    def create(cfg):
        suffix = f"packed_k{cfg.k}" if cfg.use_packing else f"padded"
        if cfg.root is None:
            cfg.root = dataroot(f"water_train_{suffix}")
        return TrainWaterDatasetFactory(**namespace_to_dict(cfg))


class AllWaterDatasetFactory(DatasetFactory):
    def create_dataset(self):
        return water_dataset.AllWaterDataSet(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        """
        The pre-computed packing strategy for the QM9 dataset.
        """
        return packing.all_water_packing_strategy()

    @staticmethod
    def create(cfg):
        suffix = f"packed_k{cfg.k}" if cfg.use_packing else f"padded"
        if cfg.root is None:
            cfg.root = dataroot(f"water_all_{suffix}")
        return AllWaterDatasetFactory(**namespace_to_dict(cfg))

class DebugWaterDatasetFactory(DatasetFactory):
    def create_dataset(self):
        return water_dataset.DebugWaterDataSet(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        return packing.water_debug_packing_strategy()

    @staticmethod
    def create(cfg):
        suffix = f"packed_k{cfg.k}" if cfg.use_packing else "padded"
        if cfg.root is None:
            cfg.root = dataroot(f"water_debug_{suffix}")
        return DebugWaterDatasetFactory(**namespace_to_dict(cfg))

class QM9DatasetFactory(DatasetFactory):
    def create_dataset(self):
        """
        Create the QM9 dataset. Downloads to the root location and applies the
        dataset transform to a pre-processed version saved to disk.
        """
        return QM9(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        """
        The pre-computed packing strategy for the QM9 dataset.
        """
        return packing.qm9_packing_strategy()

    @staticmethod
    def create(cfg):
        """
        Static factory method for creating this factory with the root location
        customised for packing/padding. This can help save time downloading and
        processing the dataset for repeated runs.
        """
        suffix = "packed" if cfg.use_packing else "padded"
        if cfg.root is None:
            cfg.root = dataroot(f"qm9_{suffix}")
        return QM9DatasetFactory(**namespace_to_dict(cfg))


class LazyDataset(torch.utils.data.Dataset):
    """
    Lazy initialized dataset.

    A lightweight decorator for lazy-initializing a map-style dataset. This
    approach can save the cost of serializing & deserializing the entire dataset
    when using asynchronous data loading.

    This class is intended to be used in combination with the DatasetFactory and
    is not intended to be called directly. Performance critical methods of this
    class are instrumented for profiling with PopVision System Analyser.
    """
    def __init__(self, dataset_factory: DatasetFactory, split: str = "train"):
        """
        :param dataset_factory (DatasetFactory): The dataset factory instance
        :param split (str): The desired dataset split to load (default: "train")
        """
        super().__init__()
        self._dataset = None
        self._factory = dataset_factory
        assert split in ("train", "val", "test"), \
          f"Invalid split = {split}. Must be 'train', 'val', or 'test'."
        self._attr = split + "_dataset"

    def _reset(self):
        """
        resets the dataset so that both this class and the factory instance are
        not holding a reference to a materialized dataset.
        """
        self._dataset = None
        self._factory.reset()

    def __len__(self):
        """
        Dataset length

        This method is called on the main process so we load the dataset, get
        the length, and then reset to avoid the cost of serializing the entire
        dataset for asynchronous workers.
        """
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        with channel.tracepoint("__len__"):
            self._load()
            L = len(self._dataset)
            self._reset()
            return L

    def _load(self):
        """
        Loads the specified dataset split using the factory.
        """
        if self._dataset is None:
            channel = poptorch.profiling.Channel(self.__class__.__name__)
            with channel.tracepoint("load"):
                self._dataset = getattr(self._factory, self._attr)

    def __getitem__(self, idx):
        """
        Get an example from the map-style dataset

        Lazy initializes the dataset on asynchronous workers.
        """
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        with channel.tracepoint("__getitem__"):
            self._load()
            return self._dataset.__getitem__(idx)

    @property
    def strategy(self):
        """
        Forwards the packing strategy property to the dataset factory.
        """
        return self._factory.strategy
