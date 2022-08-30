# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

from transforms import *
from helpers import assert_equal


@pytest.fixture
def molecule(pyg_qm9):
    # The index of the largest molecule in the QM9 dataset
    # Data(edge_attr=[56, 4], edge_index=[2, 56], idx=[1], name="gdb_57518",
    #      pos=[29, 3], x=[29, 11], y=[1, 19], z=[29])
    max_index = 55967
    return pyg_qm9[max_index]


def test_qm9_energy_target(molecule):
    transform = QM9EnergyTarget()
    assert molecule.y.shape == (1, 19)
    molecule = transform(molecule)
    assert molecule.y.shape == torch.Size([])


def test_standardize_energy(molecule):
    # First apply the QM9EnergyTarget so that y has just the energy
    transform = QM9EnergyTarget()
    molecule = transform(molecule)
    expected = molecule.y

    # Check that a round-trip of applying the standardize energy transform and
    # the inverse produces the same value.
    transform = StandardizeEnergy()
    molecule = transform(molecule)
    actual = transform.inverse(molecule.z, molecule.y)
    assert_equal(actual, expected)

    # Insert an unsupported element type and check that the transform fails
    molecule.z[0] = 2
    with pytest.raises(AssertionError, match="Invalid element type"):
        transform(molecule)


def test_pad_molecule_radius(molecule):
    transform = create_transform(mode="radius",
                                 cutoff=3.0,
                                 max_num_atoms=36,
                                 max_num_edges=360)

    molecule = transform(molecule)
    assert molecule.num_nodes == 36
    assert molecule.edge_attr.numel() == 360


@pytest.mark.parametrize("k", [20, 40])
def test_pad_molecule_knn(molecule, k):
    transform = create_transform(mode="knn", cutoff=3.0, k=k)

    num_atoms = molecule.num_nodes
    molecule = transform(molecule)
    assert molecule.z.numel() == num_atoms
    assert molecule.edge_attr.numel() == k * num_atoms


def test_pad_molecule_invalid_atoms(molecule):
    transform = create_transform(cutoff=3.0, max_num_atoms=20)

    with pytest.raises(AssertionError, match="Too many atoms"):
        transform(molecule)


def test_pad_molecule_invalid_edges(molecule):
    transform = create_transform(cutoff=3.0, max_num_edges=100)

    with pytest.raises(AssertionError, match="Too many edges"):
        transform(molecule)


def test_clamped_distance(molecule):
    # Use a radius graph method with a large cutoff and no padding to calculate
    # the full all-all interatomic distances
    all_distances = create_transform(mode="radius",
                                     cutoff=30.,
                                     use_padding=False)

    molecule = all_distances(molecule)
    distances = molecule.edge_attr
    clamped_cutoff = 3.0
    mask = distances > clamped_cutoff
    assert torch.any(
        mask), f"Expected some inter-atomic distances > {clamped_cutoff}"

    transform = ClampedDistance(cutoff=clamped_cutoff)
    molecule = transform(molecule)
    assert torch.all(molecule.edge_attr[mask] == clamped_cutoff)
    assert_equal(actual=molecule.edge_attr[~mask], expected=distances[~mask])


@pytest.mark.parametrize("k", [1, 4, 8, 36])
def test_knn_mode(k, molecule):
    transform = create_transform(mode="knn", k=k, use_padding=True)
    molecule = transform(molecule)
    num_atoms = molecule.z.numel()
    expected_num_edges = k * num_atoms

    assert molecule.edge_attr.numel() == expected_num_edges
    assert molecule.edge_index.shape[1] == expected_num_edges


@pytest.mark.parametrize("mode", ["radius", "knn"])
@pytest.mark.parametrize("use_padding", [True, False])
@pytest.mark.parametrize("use_qm9_energy", [True, False])
@pytest.mark.parametrize("use_standardized_energy", [True, False])
def test_create_transform(molecule, mode, use_padding, use_qm9_energy,
                          use_standardized_energy):
    transform = create_transform(
        mode=mode,
        use_padding=use_padding,
        use_qm9_energy=use_qm9_energy,
        use_standardized_energy=use_standardized_energy)

    molecule = transform(molecule)

    # Sanity test that all the required attributes are present
    assert hasattr(molecule, "z")
    assert hasattr(molecule, "y")
    assert hasattr(molecule, "edge_attr")
    assert hasattr(molecule, "edge_index")
