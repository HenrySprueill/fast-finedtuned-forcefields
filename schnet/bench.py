# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import poptorch
import torch
import time
import wandb
from numpy import round
from functools import partial
from tqdm import tqdm
from model import SchNet
import dataset_factory
from jsonargparse import ArgumentParser, ActionConfigFile, namespace_to_dict
from cached_dataset import PrepackedDataset
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR

class Timer:
    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.elapsed = time.perf_counter() - self.tic


class Logger:
    def __init__(self, cfg):
        self.use_wandb = cfg.bench.use_wandb
        if self.use_wandb:
            wandb.init(project=cfg.project,
                       name=cfg.run_name,
                       settings=wandb.Settings(console="wrap"),
                       config=namespace_to_dict(cfg))
            # wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py") or path.endswith(".yml"))

    def model(self, model):
        hparams = model.hyperparameters()
        if self.use_wandb:
            wandb.config.update(hparams, allow_val_change=True)

    def log(self, data):
        if self.use_wandb:
            wandb.log(data)

def poptorch_options(cfg_bench):
    options = poptorch.Options()
    options.deviceIterations(cfg_bench.device_iterations)
    options.replicationFactor(cfg_bench.replication_factor)
    options.enableSyntheticData(cfg_bench.synthetic_data)
    options.Training.gradientAccumulation(cfg_bench.gradient_accumulation)
    options._Popart.set("replicatedCollectivesSettings.prepareScheduleForMergingCollectives", True)
    options._Popart.set("replicatedCollectivesSettings.mergeAllReduceCollectives", True)
    options._Popart.set("accumulateOuterFragmentSettings.schedule", int(popart.AccumulateOuterFragmentSchedule.OverlapCycleOptimized))
    
    if cfg_bench.half_precision:
        options.Precision.setPartialsType(torch.float16)

    if cfg_bench.profile_dir:
        options.enableProfiling(cfg_bench.profile_dir)

    return options


def schnet_model(batch_size, cutoff):
    model = SchNet(batch_size=batch_size, cutoff=cutoff)
    model.train()
    return model


def synthetic_epoch(model, loader, data):
    num_steps = len(loader)
    for _ in tqdm(range(num_steps)):
        _, _ = model(*[x[0] for x in data])


def epoch_ipu(model, loader, logger, epoch=0):
    losses = []
    bar = tqdm(loader)
    N = 0

    for data in bar:
        _, loss = model(*[x[0] for x in data])
        N += (data[-1] != 0).sum().item()
        bar.set_description(f"{N} {epoch} loss: {loss.mean().item():0.6f}")
        losses.append(loss)

    mean_loss = torch.stack(losses).mean()
    print(f" Epoch {epoch},  mean epoch loss: {mean_loss}")
    logger.log({'train/mean epoch loss': mean_loss, 'Epoch': epoch,
                'train/min_loss': torch.stack(losses).min(),
                'train/max_loss': torch.stack(losses).max(),
                'train/std_loss': torch.stack(losses).std()})

def epoch_cpu(model, loader, optimizer, logger, epoch=0):
    losses = []
    bar = tqdm(loader)
    N = 0
    for data in bar:
        optimizer.zero_grad()
        _, loss = model(*[x[0] for x in data])
        N += (data[-1] != 0).sum().item()
        bar.set_description(f"{N} {epoch} loss: {loss.mean().item():0.6f}")
        losses.append(loss)
        loss.backward()
        optimizer.step()
    mean_loss = torch.stack(losses).mean()
    print(f" Epoch {epoch},  mean epoch loss: {mean_loss}")
    logger.log({'train/mean epoch loss': mean_loss, 'Epoch': epoch,
                'train/min_loss': torch.stack(losses).min(),
                'train/max_loss': torch.stack(losses).max(),
                'train/std_loss': torch.stack(losses).std()})


def bench(use_packing: bool = True,
          synthetic_data: bool = False,
          batch_size: int = 12,
          replication_factor: int = 16,
          device_iterations: int = 6,
          gradient_accumulation: int = 2,
          learning_rate: float = 1e-3,
          num_epochs: int = 10,
          profile_dir: Optional[str] = None,
          factory: Optional[str] = 'qm9',
          water_dataset: Optional[str] = None,
          use_standardized_energy=True,
          half_precision: bool = False,
          num_workers: int =4,
          use_wandb: bool = False,
          cfg = None):


    """
    IPU Benchmark of SchNet model using the QM9 Dataset.

    :param cfg: jsonargparse configuration file
    :param use_packing: Apply graph packing to the QM9 dataset.
    :param synthetic_data: Use synthetic data on the device to disable host I/O.
    :param batch_size: The batch size used by the data loader.
    :param replication_factor: The number of data parallel replicas.
    :param device_iterations: The number of device iterations to use.
    :param gradient_accumulation: The number of mini-batches to accumulate for
        the gradient calculation.
    :param learning_rate: The learning rate used by the optimiser.
    :param num_epochs: The number of epochs to benchmark.
    :param profile_dir: Run a single training step with profiling enabled and
        saves the profiling report to the provided location.
    :param factory: String used to indicate whether to run the water or QM9 dataset. The default is QM9
    :param use_standardized_energy: Use energy standardization for the energy target
    :param use_wandb: Use Weights and Biases to log benchmark results.
    """
    torch.manual_seed(0)
    logger = Logger(cfg)

    # If the replication factor is set to a negative value, use CPU only
    # poptorch options will be None in this case
    if cfg.bench.replication_factor < 0 :
        cfg.dataset_factory.options = None
    else:
        cfg.dataset_factory.options = poptorch_options(cfg.bench)

    # These options are passed to the init of the dataset_factory class
    factory = get_factory(cfg)

    channel = poptorch.profiling.Channel("dataloader")
    timer = Timer()
    with channel.tracepoint("construction"), timer:
        loader = factory.dataloader(split="train", num_workers=cfg.bench.num_workers)
        wr_dataset = PrepackedDataset(loader, opt=cfg.bench, cached_data=True)
        loader = wr_dataset.get_dataloader()

    logger.log({"dataloader/construction_time": timer.elapsed})

    if cfg.time_dataloader:
        time_dataloader(cfg, logger, channel, timer, loader)

    # Create the model here, so that even if only doing validation or testing can just load weights
    model = schnet_model(factory.model_batch_size, cfg.dataset_factory.cutoff)
    optimizer = poptorch.optim.Adam(model.parameters(), lr=cfg.bench.learning_rate)
    if cfg.bench.half_precision:
        model.half()

    logger.model(model)

    # Disable, using 2.7 million data loader will fail
    if 'all' not in cfg.bench.water_dataset :
        data = next(iter(loader))
    else:
        data = None
    
    # if replication factor > 0, the run is on IPU
    if cfg.bench.replication_factor > 0:
        training_model = poptorch.trainingModel(model=model, optimizer=optimizer, options=cfg.dataset_factory.options)
        # compile the model if its on ipu
        with timer:
            if data is not None:
                training_model.compile(*[x[0] for x in data])

        logger.log({"train/compile_time": timer.elapsed})
        # will only use synthetic data if running on ipu, not cpu. Don't need logic for
        # replication < 0 and synthetic
        if cfg.bench.synthetic_data:
            if data is not None:
                epoch_fn = partial(synthetic_epoch, data=data)
        else:
            epoch_fn = partial(epoch_ipu, logger=logger)
    # If running on cpu use the schnet model without wrapping in poptorch data loader
    else:
        training_model = model
        epoch_fn = partial(epoch_cpu, logger=logger, optimizer=optimizer)

    scheduler = CosineAnnealingLR(optimizer, cfg.bench.num_epochs, eta_min=0.00001)
    # Time to train timer
    train_start_time = time.time()

    for epoch in range(cfg.bench.num_epochs):
        with channel.tracepoint(f"epoch_{epoch}"), timer:
            epoch_fn(model=training_model, loader=loader, epoch=epoch)
            scheduler.step()
            training_model.setOptimizer(optimizer)
            logger.log({"epoch": epoch, "train/epoch_time": timer.elapsed})

    time_to_train= round((time.time() - train_start_time) / 60)
    logger.log({'time to train': time_to_train})

    if cfg.bench.replication_factor > 0:
        if training_model.isAttachedToDevice():
            training_model.detachFromDevice()

    if cfg.bench.synthetic_data:
        return

    if cfg.validation:
        # Use poptorch default options (single IPU, 1 device iteration, etc)
        # Trained weights are implicitly copied to the inferenceModel instance.
        factory.options = poptorch.Options()
        model.eval()
         # If replication factor < 0, run on CPU
        if cfg.bench.replication_factor < 0:
            val_model = model
        else:
            val_model = poptorch.inferenceModel(model)

        val_loader = factory.dataloader(split="val", num_workers=cfg.bench.num_workers)
        bar = tqdm(val_loader, desc="Validation")
        losses = []

        for data in bar:
            z, edge_attr, edge_index, batch, y = data
            prediction = val_model(z, edge_attr, edge_index, batch)
            loss = val_model.loss(prediction, y.view(-1))

            losses.append(loss)
            bar.set_description(f"loss: {loss.mean().item():0.6f}")
        losses = torch.stack(losses)
        mean = losses.mean().item()
        std = losses.std().item()
        min = losses.min().item()
        max = losses.max().item()

        unit_string = '(Kcal/mol)' if 'water' in cfg.bench.factory else '(eV/atom)'
        print(f"Mean validation loss: {mean:0.6f} +/- {std:0.6f} {unit_string}")
        print(f"               Range: [{min:0.6f}, {max:0.6f}]\n")
        logger.log({"validation/mean loss": mean})
        logger.log({"validation/std": std})
        logger.log({"validation/min": min})
        logger.log({"validation/max": max})

def get_factory(cfg):
    if 'water' in cfg.bench.factory:
        if 'all' in cfg.bench.water_dataset:
            factory= dataset_factory.AllWaterDatasetFactory.create(cfg.dataset_factory)
        elif 'test' in cfg.bench.water_dataset:
            factory= dataset_factory.TestWaterDatasetFactory.create(cfg.dataset_factory)
        elif 'debug' in cfg.bench.water_dataset:
            factory =dataset_factory.DebugWaterDatasetFactory.create(cfg.dataset_factory)
        else:
            factory= dataset_factory.TrainWaterDatasetFactory.create(cfg.dataset_factory)
    else:
        factory = dataset_factory.QM9DatasetFactory.create(cfg.dataset_factory)
    return factory

def time_dataloader(cfg, logger, channel, timer, loader):
    for step in range(cfg.bench.num_epochs):
        N = 0

        with channel.tracepoint(f"epoch_{step+1}"), timer:
            for data in tqdm(loader):
                energy_target = data[-1]
                N += (energy_target != 0).sum().item()

        logger.log({
                "epoch": step,
                "dataloader/epoch_time": timer.elapsed,
                "num_processed": N
            })

            # Simulates other work happening between epochs
        with channel.tracepoint("other"):
            time.sleep(1)


if __name__ == '__main__':
    # Creating an explicit argument parser that takes as input the attributes of the
    # dataset_factory class and bench function, plus some additional arguments
    # This is so that we can check what's coming into the code by examing the cfg
    # namespace, and ensure our changes are honored
    # Arguments that are linked only need to be provided once in the config file to
    # avoid clobering intended values through code repetition
    parser = ArgumentParser()
    parser.add_class_arguments(dataset_factory.DatasetFactory, "dataset_factory")
    parser.add_function_arguments(bench, "bench")

    # The parser will look for and use the yaml file provided after the --config flag
    # when it encounters the --config flag
    parser.add_argument('--config', action=ActionConfigFile)

    #The following arguments are linked, the first value in the call is used to populate the second
    parser.link_arguments('dataset_factory.use_packing', 'bench.use_packing')
    parser.link_arguments('dataset_factory.batch_size', 'bench.batch_size')
    parser.link_arguments('dataset_factory.use_standardized_energy', 'bench.use_standardized_energy')
    parser.add_argument('--time_dataloader', type=bool, default=False)
    parser.add_argument('--project', type=str, default='schnet_bench')
    parser.add_argument('--run_name', type=Optional[str], default=None)
    parser.add_argument('--validation', type=bool, default=False)
    
    cfg = parser.parse_args()
    cfg.poptorch_version = poptorch.__version__
    bench(cfg=cfg)
