# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

from data_utils import *
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