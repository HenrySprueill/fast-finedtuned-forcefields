# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

from dataloader import *
from data_utils import dataroot
from helpers import assert_equal


@pytest.fixture
def molecule(pyg_qm9):
    # The index of the largest molecule in the QM9 dataset
    # Data(edge_attr=[56, 4], edge_index=[2, 56], idx=[1], name="gdb_57518",
    #      pos=[29, 3], x=[29, 11], y=[1, 19], z=[29])
    max_index = 55967
    return pyg_qm9[max_index]


@pytest.mark.parametrize("folder", ["", 1.])
def test_invalid_dataroot(folder):
    with pytest.raises(AssertionError, match="Invalid folder"):
        dataroot(folder)


def test_tuple_collater(molecule):
    keys = ("x", "y", "z")
    collate_fn = TupleCollater(include_keys=keys)
    batch = collate_fn([molecule])

    assert isinstance(batch, tuple)
    assert len(batch) == len(keys)

    for i, k in enumerate(keys):
        assert_equal(actual=batch[i], expected=getattr(molecule, k))


def test_tuple_collater_invalid_keys(molecule):
    keys = ("x", "y", "z", "v")
    collate_fn = TupleCollater(include_keys=keys)

    with pytest.raises(AssertionError,
                       match="Batch is missing a required key"):
        collate_fn([molecule])


@pytest.mark.parametrize("mini_batch_size", [1, 16])
def test_combined_batching_collater(molecule, mini_batch_size):
    # Simulates 4 replicas
    num_replicas = 4
    combined_batch_size = num_replicas * mini_batch_size
    data_list = [molecule] * combined_batch_size
    collate_fn = CombinedBatchingCollater(mini_batch_size=mini_batch_size)
    batch = collate_fn(data_list)

    for v in batch:
        assert v.shape[0] == num_replicas


def test_combined_batching_collater_invalid(molecule):
    collate_fn = CombinedBatchingCollater(mini_batch_size=8)

    with pytest.raises(AssertionError, match="Invalid batch size"):
        collate_fn([molecule] * 9)
