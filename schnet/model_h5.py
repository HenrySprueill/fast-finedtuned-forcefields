# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

from torch_geometric.nn.models.schnet import GaussianSmearing, \
    InteractionBlock, ShiftedSoftplus
from torch_scatter.scatter import scatter_add
#from poptorch import identity_loss
#from torch_geometric.transforms import Compose, Distance, KNNGraph
#import transforms

from torch_geometric.nn import knn_graph

class SchNet(nn.Module):
    """IPU implementation of SchNet

    This implementation is adapted from the PyTorch Geometric implementation of
    SchNet. The main differences are: 

        * the interaction graph must be pre-computed as part of the dataset 
          pre-processing.
        * all inputs of the forward method are padded to meet the static tensor
          shape requirements of the IPU.
        * dataset standardization is not handled by the model. Instead this is
          expected to be handled in the dataset pre-processing when necessary.

    Padding atoms are defined as having atomic charge of zero and are 
    non-interacting. Padding edges are defined as having a length equal to the
    cutoff to ensure they are excluded from the radial basis expansion.
    """
    def __init__(self,
                 num_features: int = 100,
                 num_interactions: int = 4,
                 num_gaussians: int = 25,
                 cutoff: float = 6.0,
                 batch_size: Optional[int] = None,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        """
        :param num_features (int): The number of hidden features used by both
            the atomic embedding and the convolutional filters (default: 128).
        :param num_interactions (int): The number of interaction blocks
            (default: 2).
        :param num_gaussians (int): The number of gaussians used in the radial
            basis expansion (default: 50).
        :param cutoff (float): Cutoff distance for interatomic interactions
            which must match the one used to build the radius graphs
            (default: 6.0).
        :param batch_size (int, optional): The number of molecules in the batch.
            This can be inferred from the batch input when not supplied.
        :param mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        :param std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        :param atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
        """
        super().__init__()
        self.num_features = num_features
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        self.atom_embedding = nn.Embedding(100,
                                           self.num_features,
                                           padding_idx=0)
        self.basis_expansion = GaussianSmearing(0.0, self.cutoff,
                                                self.num_gaussians)

        self.interactions = nn.ModuleList()

        for _ in range(self.num_interactions):
            block = InteractionBlock(self.num_features, self.num_gaussians,
                                     self.num_features, self.cutoff)
            self.interactions.append(block)

        self.lin1 = nn.Linear(self.num_features, self.num_features // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(self.num_features // 2, 1)
        
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def hyperparameters(self):
        """
        hyperparameters for the SchNet model.

        :returns: Dictionary of hyperparamters.
        """
        return {
            "num_features": self.num_features,
            "num_interactions": self.num_interactions,
            "num_gaussians": self.num_gaussians,
            "cutoff": self.cutoff,
            "batch_size": self.batch_size
        }

    def extra_repr(self) -> str:
        """
        extra representation for the SchNet model.

        :returns: comma-separated string of the model hyperparameters.
        """
        s = []
        for key, value in self.hyperparameters().items():
            s.append(f"{key}={value}")

        return ", ".join(s)

    def reset_parameters(self):
        """
        Initialize learnable parameters used in training the SchNet model.
        """
        self.atom_embedding.reset_parameters()

        for interaction in self.interactions:
            interaction.reset_parameters()

        xavier_uniform_(self.lin1.weight)
        zeros_(self.lin1.bias)
        xavier_uniform_(self.lin2.weight)
        zeros_(self.lin2.bias)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch):#edge_weight, edge_index, batch, energy_target=None):
        """
        Forward pass of the SchNet model

        :param z: Tensor containing the atomic numbers for each atom in the
            batch. Vector with size [num_atoms].
        :param edge_weight: Tensor containing the interatomic distances for each
            interacting pair of atoms in the batch. Vector with size [num_edges]
        :param edge_index: Tensor containing the indices defining the
            interacting pairs of atoms in the batch. Matrix with size
            [2, num_edges]
        :param batch: Tensor assigning each atom within a batch to a molecule.
            This is used to perform per-molecule aggregation to calculate the
            predicted energy. Vector with size [num_atoms]
        :param energy_target (optional): Tensor containing the energy target to
            use for evaluating the mean-squared-error loss when training.
        """
        # Collapse any leading batching dimensions
        batch = batch.long()
        edge_index = knn_graph(
            pos,
            28,
            batch,
            loop=False,
        )
        
        row, col = edge_index

        edge_weight = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        
        z = z.view(-1).long()
        #edge_weight = data.edge_attr.view(-1)
        edge_index = edge_index.view(2, -1).long()
        #batch = batch.long()

        h = self.atom_embedding(z)
        edge_attr = self.basis_expansion(edge_weight)

        #h = self.atom_embedding(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        
        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean
            
        if self.atomref is not None:
            h = h + self.atomref(z)

        mask = (z == 0).view(-1, 1)
        h = h.masked_fill(mask.expand_as(h), 0.)

        batch = batch.view(-1)
        out = scatter_add(h, batch, dim=0, dim_size=self.batch_size).view(-1)

        if not self.training:
            return out

        return out

    @staticmethod
    def loss(input, target):
        """
        Calculates the mean squared error

        This loss assumes that zeros are used as padding on the target so that
        the count can be derived from the number of non-zero elements.
        """
        loss = F.mse_loss(input, target, reduction="sum")
        N = (target != 0.0).to(loss.dtype).sum()
        loss = loss / N
        return identity_loss(loss, reduction="none")
