# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import torch
from tqdm import tqdm
from torch_geometric.datasets import QM9
from data_utils import dataroot


@pytest.fixture
def pyg_qm9():
    return QM9(root=dataroot("pyg_qm9"))


@pytest.fixture
def qm9_num_nodes(pyg_qm9, request):
    """
    Fixture for calculating the number of nodes per graph for the QM9 dataset

    Uses pytest caching to store result to speed up testing. Cleared with:

        % pytest --cache-clear
    """
    num_nodes = request.config.cache.get("pyg_qm9/num_nodes", None)

    if num_nodes is None:
        print(
            "Calculating the number of nodes for each graph in the QM9 dataset."
        )
        num_nodes = [d.num_nodes for d in tqdm(pyg_qm9)]
        request.config.cache.set("pyg_qm9/num_nodes", num_nodes)

    return torch.tensor(num_nodes)
