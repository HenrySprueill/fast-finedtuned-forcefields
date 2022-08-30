# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
from torch.nn.functional import mse_loss
import poptorch
import pytest
import torch_geometric.nn.models.schnet as gschnet

from copy import deepcopy
from torch_geometric.data import Batch
from dataloader import create_packing_molecule
from transforms import create_transform
from model import SchNet
from torch.testing import assert_close
from helpers import assert_equal
from itertools import product

CUTOFF = 6.0
K = 10
batch_size = [1, 8]
mode = ["radius", "knn"]
params = list(product(batch_size, mode))
ids = (f"{m}_{b}" for b, m in params)


@pytest.fixture(params=params, ids=ids)
def batch(pyg_qm9, request):
    batch_size, mode = request.param

    pyg_qm9.transform = create_transform(mode=mode,
                                         cutoff=CUTOFF,
                                         k=K,
                                         use_qm9_energy=True,
                                         use_standardized_energy=True,
                                         use_padding=True)
    data_list = list(pyg_qm9[0:batch_size])
    return Batch.from_data_list(data_list)


class InferenceHarness(object):
    def __init__(self,
                 batch_size,
                 mol_padding=0,
                 num_features=32,
                 num_gaussians=25,
                 num_interactions=2):
        super().__init__()
        self.seed = 0
        self.batch_size = batch_size
        self.mol_padding = mol_padding
        self.create_model(num_features, num_gaussians, num_interactions)
        self.create_reference_model(num_features, num_gaussians,
                                    num_interactions)

    def create_model(self, num_features, num_gaussians, num_interactions):
        # Set seed before creating the model to ensure all parameters are
        # initialized to the same values as the PyG reference implementation.
        torch.manual_seed(self.seed)
        self.model = SchNet(num_features=num_features,
                            num_gaussians=num_gaussians,
                            num_interactions=num_interactions,
                            cutoff=CUTOFF,
                            batch_size=self.batch_size)
        self.model.eval()

    def create_reference_model(self, num_features, num_gaussians,
                               num_interactions):
        # Use PyG implementation as a reference implementation
        torch.manual_seed(0)
        self.ref_model = gschnet.SchNet(hidden_channels=num_features,
                                        num_filters=num_features,
                                        num_gaussians=num_gaussians,
                                        num_interactions=num_interactions,
                                        cutoff=CUTOFF)

        self.ref_model.eval()

    def reference_output(self, batch):
        # Mask out fake atom data added as padding.
        real_atoms = batch.z > 0.

        if torch.all(~real_atoms):
            # All padding atoms case
            return torch.zeros(torch.max(batch.batch) + 1)

        out = self.ref_model(batch.z[real_atoms], batch.pos,
                             batch.batch[real_atoms])
        return out.view(-1)

    def compare(self, actual, batch):
        expected = self.reference_output(batch)

        if self.mol_padding == 0:
            assert_close(actual, expected)
            return

        pad_output = torch.zeros(self.mol_padding)
        assert_equal(actual[-self.mol_padding:], pad_output)
        assert_close(actual[:-self.mol_padding], expected)

    def test_cpu_padded(self, batch):
        # Run padded model on CPU and check the output agrees with the reference
        actual = self.model(batch.z, batch.edge_attr, batch.edge_index,
                            batch.batch)
        self.compare(actual, batch)

    def test_ipu(self, batch):
        pop_model = poptorch.inferenceModel(self.model)
        actual = pop_model(batch.z, batch.edge_attr, batch.edge_index,
                           batch.batch)
        self.compare(actual, batch)


def test_inference(batch):
    mol_padding = 2
    batch_size = torch.max(batch.batch).item() + 1 + mol_padding
    harness = InferenceHarness(batch_size, mol_padding)
    harness.test_cpu_padded(batch)
    harness.test_ipu(batch)


def test_inference_packing():
    packing_molecule = create_packing_molecule(num_atoms=7, k=4)
    batch = Batch.from_data_list([packing_molecule])
    harness = InferenceHarness(batch_size=1)
    harness.test_cpu_padded(batch)
    harness.test_ipu(batch)


class Stepper(object):
    def __init__(self, model, lr=0.001, optimizer=poptorch.optim.Adam):
        super().__init__()
        model.train()
        self.lr = lr
        self.setup_cpu(model, optimizer)
        self.setup_ipu(model, optimizer)
        self.check_parameters()

    def setup_cpu(self, model, optimizer):
        self.cpu_model = deepcopy(model)
        self.optimizer = optimizer(self.cpu_model.parameters(), lr=self.lr)

    def setup_ipu(self, model, optimizer):
        self.ipu_model = deepcopy(model)
        ipu_optimizer = optimizer(self.ipu_model.parameters(), lr=self.lr)
        options = poptorch.Options()
        options.Precision.enableFloatingPointExceptions(True)
        self.training_model = poptorch.trainingModel(self.ipu_model,
                                                     optimizer=ipu_optimizer,
                                                     options=options)

    def check_parameters(self):
        for cpu, ipu in zip(self.cpu_model.named_parameters(),
                            self.ipu_model.named_parameters()):
            name, cpu = cpu
            ipu = ipu[1]
            message = f"Parameter {name} was not equal"
            assert_close(ipu, cpu, msg=message)

    def cpu_step(self, batch):
        self.optimizer.zero_grad()
        out, loss = self.cpu_model(*batch)
        loss.backward()
        self.optimizer.step()
        return out, loss

    def ipu_step(self, batch):
        out, loss = self.training_model(*batch)
        self.training_model.copyWeightsToHost()
        return out, loss


def test_training(batch):
    torch.manual_seed(0)
    model = SchNet(num_features=32, num_gaussians=25, num_interactions=2)
    model.train()
    stepper = Stepper(model)

    num_steps = 40
    cpu_loss = torch.empty(num_steps)
    ipu_loss = torch.empty(num_steps)
    batch = (batch.z, batch.edge_attr, batch.edge_index, batch.batch, batch.y)

    for i in range(num_steps):
        cpu_out, cpu_loss[i] = stepper.cpu_step(batch)
        ipu_out, ipu_loss[i] = stepper.ipu_step(batch)
        assert_close(actual=ipu_out, expected=cpu_out)
        stepper.check_parameters()
        print(f"{cpu_loss[i]:0.03f}   {ipu_loss[i]:0.03f}")

    assert_close(actual=ipu_loss, expected=cpu_loss, atol=1e-4, rtol=1e-5)


def test_loss():
    # Check that loss agrees with mse_loss implementation
    torch.manual_seed(0)
    input = torch.randn(10)
    target = torch.randn(10)
    actual = SchNet.loss(input, target)
    expected = mse_loss(input, target)
    assert_equal(actual, expected)

    # Insert random "padding" zeros.
    mask = torch.randn_like(input) > 0.6
    input[mask] = 0.0
    target[mask] = 0.0
    actual = SchNet.loss(input, target)
    expected = mse_loss(input[~mask], target[~mask])
    assert_equal(actual, expected)