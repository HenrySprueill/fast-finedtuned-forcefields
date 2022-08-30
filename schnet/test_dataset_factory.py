# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest

from dataset_factory import QM9DatasetFactory
from torch.testing import assert_close
from helpers import assert_equal
from tqdm import tqdm


def assert_datasets_equal(first_loader, second_loader):
    loaders = zip(first_loader, second_loader)
    bar = tqdm(loaders, desc="assert_datasets_equal")
    for a, b in bar:
        for u, v in zip(a, b):
            assert_equal(u, v)


@pytest.mark.parametrize("use_packing", [True, False])
def test_qm9_datasetfactory(use_packing):
    torch.manual_seed(0)
    factory = QM9DatasetFactory.create(batch_size=1024,
                                       use_packing=use_packing)

    num_train = len(factory.train_dataset)
    num_val = len(factory.val_dataset)
    num_test = len(factory.test_dataset)

    if use_packing:
        num_examples = factory.strategy.packed_len()
    else:
        num_examples = factory.strategy.unpacked_len()

    assert num_train + num_val + num_test == num_examples

    # Check the default split of (0.8, 0.1, 0.1) is satisfied
    assert_close(num_train, round(0.8 * num_examples), atol=1, rtol=1)
    assert_close(num_val, round(0.1 * num_examples), atol=1, rtol=1)
    assert_close(num_test, round(0.1 * num_examples), atol=1, rtol=1)

    # Check that randomized splits are reproducible
    for split in ("train", "val", "test"):
        first_loader = factory.dataloader(split=split, num_workers=16)
        factory.reset()
        second_loader = factory.dataloader(split=split, num_workers=16)
        assert_datasets_equal(first_loader, second_loader)