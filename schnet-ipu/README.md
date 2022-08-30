# IPU implementation of SchNet Rev 0.1

The implementation of Schnet that includes the water dataset now uses a single config file. This config file can be used to run both the QM9 dataset and the Water dataset. The config file is named `bench_config.yml` and is in the `configs` directory.

The `bench_config.yml` file is annotated, containing an explanation of what each parameter does in comments located above the parameter. The default values used for the QM9 dataset are listed there as well.

The run instructions are the same as below, with the name of the config file being the only change.

```:bash
python3 bench.py --config configs/bench_config.yml
```

As pointed out below, each parameter in the config file can be individually set with a flag at the command line.
All possible parameters that can be passed to the command line are listed below.

```:bash
usage: bench.py [-h] [--dataset_factory.root ROOT]
                [--dataset_factory.splits SPLITS]
                [--dataset_factory.batch_size BATCH_SIZE]
                [--dataset_factory.use_packing {true,false}]
                [--dataset_factory.k K] [--dataset_factory.cutoff CUTOFF]
                [--dataset_factory.max_num_atoms MAX_NUM_ATOMS]
                [--dataset_factory.use_qm9_energy {true,false}]
                [--dataset_factory.use_standardized_energy {true,false}]
                [--dataset_factory.options.help CLASS]
                [--dataset_factory.options OPTIONS]
                [--bench.synthetic_data {true,false}]
                [--bench.replication_factor REPLICATION_FACTOR]
                [--bench.device_iterations DEVICE_ITERATIONS]
                [--bench.gradient_accumulation GRADIENT_ACCUMULATION]
                [--bench.learning_rate LEARNING_RATE]
                [--bench.num_epochs NUM_EPOCHS]
                [--bench.profile_dir PROFILE_DIR] [--bench.factory FACTORY]
                [--bench.water_dataset WATER_DATASET]
                [--bench.half_precision {true,false}]
                [--bench.num_workers NUM_WORKERS]
                [--bench.use_wandb {true,false}] [--config CONFIG]
                [--print_config [={comments,skip_null}+]]
                [--time_dataloader {true,false}] [--project PROJECT]
                [--run_name RUN_NAME] [--validation {true,false}]
                [--num_workers NUM_WORKERS] [--checkpoint {true,false}]
                [--checkpoint_epochs CHECKPOINT_EPOCHS]
                [--model_name MODEL_NAME]
```

A more detailed help message for each parameter can be obtained by running

```bash
python3 bench.py --help
```
# IPU Implementation of SchNet

## Overview

This implementation is adapted from the PyTorch Geometric implementation of [SchNet](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py). The main differences are: 

* the interaction graph is pre-computed as part of the dataset pre-processing.
* all inputs of the forward method are padded to provide static tensor shapes to
  the IPU.
* dataset standardization is not handled by the model. Instead this is expected
  to be handled in the dataset pre-processing when necessary.

This project implements two techniques for producing static tensor shapes:

* Padding: this is the simplest approach where each molecule is padded to have
  the same number of atoms and edges as the largest molecule in the dataset.
* Packing: by analysing the distribution of the number of atoms we can minimise
  padding by combining molecules of different sizes to reach a target batch
  size (in terms of the total number of atoms).

## Datasets

* [QM9 from PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/qm9.py)

## Running
`bench.py` is the main script for running a comprehensive performance experiment
on the IPU. The full usage is replicated below for reference.  This script
accepts a `--config` argument to specify a `yml` file containing the options to
use for the experiment. For example, to run the benchmark with graph packing:
```:bash
python3 bench.py --config configs/qm9_packed.yml
```

Similarly, to run the benchmark with padding applied to every molecule:
```:bash
python3 bench.py --config configs/qm9_padded.yml
```

Configurations set in a `yml` file can be used in combination with command-line
arguments:
```:bash
python3 bench.py --config configs/qm9_packed.yml --use_wandb true
```
The above command will run the packing benchmark with the logging to 
[Weights and Biases](https://docs.wandb.ai/) enabled.

### Complete Usage
Below are the supported command-line arguments for `bench.py`.  

```:bash
usage: bench.py [-h] [--config CONFIG]
                [--print_config [={comments,skip_null}+]]
                [--use_packing {true,false}] [--synthetic_data {true,false}]
                [--batch_size BATCH_SIZE]
                [--replication_factor REPLICATION_FACTOR]
                [--device_iterations DEVICE_ITERATIONS]
                [--gradient_accumulation GRADIENT_ACCUMULATION]
                [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
                [--profile_dir PROFILE_DIR] [--use_wandb {true,false}]

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file in json or yaml format.
  --print_config [={comments,skip_null}+]
                        Print configuration and exit.

IPU Benchmark of SchNet model using the QM9 Dataset:
  --use_packing {true,false}
                        Apply graph packing to the QM9 dataset. (type: bool,
                        default: True)
  --synthetic_data {true,false}
                        Use synthetic data on the device to disable host I/O.
                        (type: bool, default: False)
  --batch_size BATCH_SIZE
                        The batch size used by the data loader. (type: int,
                        default: 12)
  --replication_factor REPLICATION_FACTOR
                        The number of data parallel replicas. (type: int,
                        default: 16)
  --device_iterations DEVICE_ITERATIONS
                        The number of device iterations to use. (type: int,
                        default: 6)
  --gradient_accumulation GRADIENT_ACCUMULATION
                        The number of mini-batches to accumulate for the
                        gradient calculation. (type: int, default: 2)
  --learning_rate LEARNING_RATE
                        The learning rate used by the optimiser. (type: float,
                        default: 0.001)
  --num_epochs NUM_EPOCHS
                        The number of epochs to benchmark. (type: int,
                        default: 10)
  --profile_dir PROFILE_DIR
                        Run a single training step with profiling enabled and
                        saves the profiling report to the provided location.
                        (type: Union[str, null], default: null)
  --use_wandb {true,false}
                        Use Weights and Biases to log benchmark results.
                        (type: bool, default: True)
```

## Testing
The following command will run all the tests in this directory:
```:bash
pytest -s .
```

## Profiling
Graphcore have developed two profiling tools that can help identify bottlenecks:

* [PopVision Graph Analyser](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/)
* [PopVision System Analyser](https://docs.graphcore.ai/projects/system-analyser-userguide/en/latest/)

For example, the following command will collect a graph analyser report for a
single epoch with dataset packing:
```:bash
python3 bench.py --config configs/qm9_packed.yml --num_epochs 1 --profile_dir graph_profile
```
This technique is used to understand the execution of the model on the
IPU.  The system analyser complements this by providing insight into the
execution of the data pipeline on the host.  Set the following environment
variable to capture a system analyser profile:
```:bash
export PVTI_OPTIONS='{"enable":"true", "directory":"system_profile"}'
```
Running your model with this environment variable set will enable 
instrumentation in the Poplar SDK to capture timing information for the process
of orchestrating data movement from the host to the IPU. Additional timing
information can be collected by using the
[poptorch.profiling.Channel](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.profiling.Channel)
context
manager.

