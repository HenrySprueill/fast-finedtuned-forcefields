# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F

from torch_geometric.transforms import Compose, Distance, KNNGraph, RadiusGraph
from typing import Optional


# BaseTransforms taken from pytorch_geometric
from abc import ABC
from typing import Any
class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`~torch_geometric.data.Data` objects, either by implicitly passing
    them as an argument to a :class:`~torch_geometric.data.Dataset`, or by
    applying them explicitly to individual :class:`~torch_geometric.data.Data`
    objects.

    .. code-block:: python

        import torch_geometric.transforms as T
        from torch_geometric.datasets import TUDataset

        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

        dataset = TUDataset(path, name='MUTAG', transform=transform)
        data = dataset[0]  # Implicitly transform data on every access.

        data = TUDataset(path, name='MUTAG')[0]
        data = transform(data)  # Explicitly transform data.
    """
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class QM9EnergyTarget(BaseTransform):
    """
    Data transform to extract the energy target from the PyG QM9 dataset.

    The QM9 dataset consists of a total of 19 regression targets. This transform
    updates the regression targets stored in data.y to only include the internal
    energy at 0K (eV) to use as the target for training.

    Expected input:
        data.y is a vector with shape [1, 19]

    Transformed output:
        data.y is as scalar with shape torch.Size([])
    """
    ENERGY_TARGET = 7

    def __init__(self, debug: bool = False):
        self.debug = debug

    def validate(self, data):
        assert hasattr(data, "y") \
          and isinstance(data.y, torch.Tensor) \
          and data.y.shape == (1, 19),\
          "Invalid data input. Expected data.y == Tensor with shape [1, 19]"

    def __call__(self, data):
        self.validate(data)
        data.y = data.y[0, self.ENERGY_TARGET]
        return data

class StandardizeEnergy(BaseTransform):
    """
    Data transform for energy standardization

    This transform rescales the molecular total energy to an energy per-atom
    that does not include the single-atom energies. This is done by looking up a
    reference value for the single-atom energy for each atom. These values are
    added up over all atoms in the molecule and subtracted from the molecular
    total energy. Finally this difference is rescaled by the number of atoms in
    the sample. More succinctly:

    E_per_atom = (E_molecule - sum(E_atomref)) / num_atoms

    This transform is motivated by training a SchNet network with a dataset that
    contains molecules with a high variation in the number of atoms. This will
    be directly correlated with a large variation in the total energy regression
    target. Applying this transform is expected to both accelerate and
    stabilize the training process.
    """
    def __init__(self):
        super().__init__()
        self.atomref = self.energy_atomref()

    @staticmethod
    def energy_atomref():
        """
        Single-atom reference energies for atomic elements in QM9 data. See:

            torch_geometric/datasets/qm9.py: https://git.io/JP6iU

        Limited to H, C, N, O, and F elements which are the only atoms present
        in the QM9 dataset.
        """
        refs = [
            -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
            -2713.48485589
        ]

        out = torch.zeros(100)
        out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(refs)
        return out

    def _sum_atomrefs(self, z):
        refs = self.atomref[z]
        mask = refs == 0.
        assert ~torch.any(mask), f"Invalid element type(s): {z[mask].tolist()}"
        return refs.sum()

    def __call__(self, data):
        num_atoms = data.z.numel()
        sum_atomrefs = self._sum_atomrefs(data.z)
        data.y = (data.y - sum_atomrefs) / num_atoms
        return data

    def inverse(self, z, y):
        """
        Performs the inverse of the standardize energy transform.

        :param z (Tensor [num_atoms]): The atomic numbers for the molecule
        :param y (Tensor): The standardized energy to invert.
        """
        return y * z.numel() + self._sum_atomrefs(z)


class PadMolecule(BaseTransform):
    """
    Data transform that applies padding to enforce consistent tensor shapes.

    Padding atoms are defined as have atomic charge of zero. Padding edges are
    defined as having a length equal to the cutoff used in the Schnet network.
    """
    def __init__(self,
                 mode: str = "radius",
                 max_num_atoms: Optional[int] = None,
                 max_num_edges: Optional[int] = None,
                 k: Optional[int] = None,
                 cutoff: float = 6.0):
        """
        :param mode (str): Mode used to calculate the graph edges which impacts
            the necessary padding as:
            * "radius": pads atomic numbers up to max_num_atoms, and graph edges
                are padded up to max_num_edges.
            * "knn": pads graph edges up to k * num_atoms for molecules where
                num_atoms < k.
        :param max_num_atoms (int, optional): The maximum number of atoms to pad
            the atom numbers up to. Must be provided when using "radius" mode.
        :param max_num_edges (int, optional): The maximum number of edges used
            to pad the edge_index and edge_attr up to. This can be inferred as
            max_num_atoms * (max_num_atoms-1) but note that this may introduce
            more padding than necessary.
        :param k (int, optional): The number of nearest neighbors used by the
            KNN graph. Used to pad edges up to k * num_atoms for molecules that
            are smaller than k.  Must be provided when using "knn" mode.
        :param cutoff (float): The cutoff in Angstroms used in the SchNet model
            (default: 6.0).
        """
        super().__init__()
        self.mode = mode
        self.max_num_atoms = max_num_atoms
        self.max_num_edges = max_num_edges
        self.k = k
        self.cutoff = cutoff

        if self.mode == "radius":
            assert self.max_num_atoms is not None, "Incompatible options: "\
            "'max_num_atoms' must be provided when mode='radius'."

            if self.max_num_edges is None:
                # Assume fully connected graph between all atoms
                self.max_num_edges = max_num_atoms * (max_num_atoms - 1)

        elif self.mode == "knn":
            assert self.k is not None, "Incompatible options: "\
            "'k' must be provided when mode='knn'."

        else:
            raise ValueError(f"Invalid mode='{mode}'. "\
                "Supported modes are 'knn' or 'radius'")

    def validate(self, data):
        """
        Validates that the input molecule does not exceed the constraints that:

          * the number of atoms must be <= max_num_atoms
          * the number of edges must be <= max_num_edges

        :returns: Tuple containing the number atoms and the number of edges
        """
        num_atoms = data.z.numel()
        num_edges = data.edge_index.shape[1]

        if self.mode == "radius":
            assert num_atoms <= self.max_num_atoms, \
            f"Too many atoms. Molecule has {num_atoms} atoms "\
            f"and max_num_atoms is {self.max_num_atoms}."

            assert num_edges <= self.max_num_edges, \
            f"Too many edges. Molecule has {num_edges} edges defined "\
            f"and max_num_edges is {self.max_num_edges}."

        return num_atoms, num_edges

    def __call__(self, data):
        num_atoms, num_edges = self.validate(data)

        if self.mode == "radius":
            num_fake_atoms = self.max_num_atoms - num_atoms
            data.z = F.pad(data.z, (0, num_fake_atoms))
            data.num_nodes = self.max_num_atoms

        if self.mode == "radius":
            num_fake_edges = self.max_num_edges - num_edges
        else:
            num_fake_edges = self.k * num_atoms - num_edges

        # Fake edges are self-loops on the first atom.
        data.edge_index = F.pad(data.edge_index, (0, num_fake_edges, 0, 0),
                                value=0)
        data.edge_attr = F.pad(data.edge_attr.squeeze(1), (0, num_fake_edges),
                               value=self.cutoff)
        return data

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(mode={self.mode}, "

        if self.max_num_atoms is not None:
            s += f"max_num_atoms={self.max_num_atoms}, "

        if self.max_num_edges is not None:
            s += f"max_num_edges={self.max_num_edges}, "

        if self.k is not None:
            s += f"k={self.k}, "

        s += f"cutoff={self.cutoff})"
        return s


class ClampedDistance(BaseTransform):
    """
    Clamps the distances to within the specified cutoff.

    This transform is intended to be used in combination with the k-NN graph
    method. This ensures that there are no interatomic interactions outside of
    the cutoff radius for the SchNet model.
    """
    def __init__(self, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, data):
        data.edge_attr.clamp_max_(self.cutoff)
        return data

    def __repr__(self) -> str:
        return '{}(cutoff={})'.format(self.__class__.__name__, self.cutoff)


def create_transform(mode: str = "radius",
                     k: int = 8,
                     cutoff: float = 6.0,
                     use_padding: bool = True,
                     max_num_atoms: int = 32,
                     max_num_edges: int = None,
                     use_qm9_energy: bool = False,
                     use_standardized_energy: bool = False):
    """
    Creates a sequence of transforms defining a data pre-processing pipeline

    :param mode (str): Mode for calculating graph edges. Can either be:
        * "radius": creates edges based on node positions within the cutoff.
            This is the default mode.
        * "knn": creates edges from the k-nearest neighbors. Any neighbors that
            fall outside of the cutoff are clamped to the cutoff value.
    :param k (int): Number of neighbors used when mode is "knn" (default: 8).
    :param cutoff (float): Cutoff distance for interatomic interactions in
        Angstroms (default: 6.0).
    :param use_padding (bool): Use the PadMolecule transform (default: True)
    :param max_num_atoms (int): The maximum number of atoms used by the
        PadMolecule transform (default: 32).
    :param max_num_edges (int, optional):  The maximum number of edges used by
        the PadMolecule transform. When not provided, this will be inferred as:
        max_num_atoms * (max_num_atoms-1) but may introduce more padding than
        necessary (default: None)
    :param use_qm9_energy (bool): Use the QM9EnergyTarget transform
        (default: False).
    :param use_standardized_energy (bool): Use the StandardizeEnergy transform
        (default: False).

    :returns: A composite transform
    """
    if mode == "knn":
        transforms = [KNNGraph(k=k, force_undirected=False, loop=False)]

    elif mode == "radius":
        transforms = [RadiusGraph(cutoff)]
        if max_num_edges is None:
            max_num_edges = max_num_atoms * (max_num_atoms - 1)

    else:
        raise ValueError(
            f"Invalid mode='{mode}'. Supported modes are 'knn' or 'radius'")

    transforms.append(Distance(norm=False, cat=False))

    if mode == "knn":
        transforms.append(ClampedDistance(cutoff=cutoff))

    if use_qm9_energy:
        transforms.append(QM9EnergyTarget())

    if use_standardized_energy:
        transforms.append(StandardizeEnergy())

    # Padding transform is applied last since it will introduce padding atoms
    # that are incompatible with the StandardizeEnergy transform (if used).
    if use_padding:
        transforms.append(
            PadMolecule(mode=mode,
                        k=k,
                        cutoff=cutoff,
                        max_num_atoms=max_num_atoms,
                        max_num_edges=max_num_edges))

    return Compose(transforms)
