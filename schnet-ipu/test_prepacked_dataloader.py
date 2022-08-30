# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
from schnet.cached_dataset import PrepackedDataset
import poptorch
from dataset_factory import QM9DatasetFactory


class Config:
    def __init__(self):
        self.num_workers = 10
        self.batch_size = 8
        self.device_iterations = 16
        self.replication_factor = 8
        self.gradient_accumulation = 1
        self.synthetic_data = False
        self.half_precision = False

def poptorch_options(cfg_bench):
    options = poptorch.Options()
    options.deviceIterations(cfg_bench.device_iterations)
    options.replicationFactor(cfg_bench.replication_factor)
    options.enableSyntheticData(cfg_bench.synthetic_data)
    options.Training.gradientAccumulation(cfg_bench.gradient_accumulation)
    options._Popart.set("defaultPrefetchBufferingDepth", 3)
    options._Popart.set("rearrangeAnchorsOnHost", False)

    if cfg_bench.half_precision:
        options.Precision.setPartialsType(torch.float16)

    return options

def test_tuple_collater():
    cfg = Config()
    options = poptorch_options(cfg)
    factory = QM9DatasetFactory(batch_size = cfg.batch_size,
                                options = options)
    factory_loader = factory.dataloader(split="val", num_workers=cfg.num_workers)
    wr_dataset = PrepackedDataset(factory_loader, opt=cfg, cached_data=True, shuffle=False)
    loader = wr_dataset.get_dataloader()

    loader_iter = iter(loader)
    idx = 0
    for data in loader.dataset.test_data:
        prepacked_data = next(loader_iter)
        assert torch.all(data[0] == prepacked_data[0])
        assert torch.all(data[1] == prepacked_data[1])
        assert torch.all(data[2] == prepacked_data[2])
        assert torch.all(data[3] == prepacked_data[3])
        assert torch.all(data[4] == prepacked_data[4])
        idx += 1
    assert idx == 10
