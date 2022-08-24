"""Run clusters with different basis sets and store results to an ASE database

It is presently hard-coded to run on Cori KNL nodes. You will need to change the Parsl configuration,
 the NWChem executable path, and the NWChem memory/core settings to adapt to a different system
"""

from argparse import ArgumentParser
from typing import Iterable, Tuple, Optional
from concurrent.futures import as_completed, Future

from parsl import ThreadPoolExecutor
from parsl.dataflow.futures import AppFuture
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import SlurmProvider
from ase.calculators.nwchem import NWChem
from ase.db import connect
from ase.io import read
from io import TextIOWrapper
from tqdm import tqdm
import zipfile
import ase

import parsl

# Compute node information (settings for Cori KNL)
cores_per_node = 68
memory_per_node = 90  # In GB
scratch_path = '/global/cscratch1/sd/wardlt/nwchem-bench/'


@parsl.python_app
def run_nwchem(atoms: ase.Atoms, calc: NWChem, temp_path: Optional[str] = None) -> Tuple[ase.Atoms, float]:
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        calc: NWChem calculator to use
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation
    """
    from tempfile import TemporaryDirectory
    import time

    with TemporaryDirectory(dir=temp_path, prefix='nwc') as dir:
        # Update the scratch directory
        calc.scratch = str(dir)
        calc.perm = str(dir)

        # Run the calculation
        start_time = time.perf_counter()
        atoms.set_calculator(calc)
        atoms.get_forces()
        run_time = time.perf_counter() - start_time

        return atoms, run_time


def generate_structures() -> Iterable[Tuple[str, ase.Atoms]]:
    """Iterate over all structures in Henry's ZIP file

    Yields:
        Tuple of (filename, ase.Atoms) object
    """

    with zipfile.ZipFile('data/initial_MP2.zip') as zp:
        for info in zp.infolist():
            if info.filename.endswith(".xyz"):
                with zp.open(info, mode='r') as fp:
                    yield info.filename, read(TextIOWrapper(fp), format='xyz')


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--ranks-per-node', default=64, help='Number of ranks per node', type=int)
    parser.add_argument('--num-nodes', default=1, help='Number of nodes on which to run', type=int)
    parser.add_argument('--basis', default='aug-cc-pvdz', help='Basis set to use for all atoms')
    parser.add_argument('--max-to-run', default=None, type=int, help='Maximum number of tasks to run')
    args = parser.parse_args()

    # Make a generator over structures to read
    strc_iter = generate_structures()

    # Make the NWChem calculator
    ranks_per_node = cores_per_node
    calc = nwchem = NWChem(
        memory=f'{memory_per_node / ranks_per_node:.1f} gb',
        basis={'*': args.basis},
        set={
            'lindep:n_dep': 0,
            'cphf:maxsub': 95,
            'mp2:aotol2e fock': '1d-14',
            'mp2:aotol2e': '1d-14',
            'mp2:backtol': '1d-14',
            'cphf:maxiter': 999,
            'cphf:thresh': '6.49d-5',
            'int:acc_std': '1d-16'
        },
        scf={'maxiter': 99, 'tol2e': '1d-15'},
        mp2={'freeze': 'atomic'},
        initwfc={
            'dft': {
                'xc': 'hfexch',
                # 'convergence': {  # TODO (wardlt): Explore if there is a better value for the convergence
                #     'energy': '1d-12',
                #     'gradient': '5d-19'
                # },
                # 'tolerances': {'acccoul': 15},
                'maxiter': 50,
            },
            'set': {
                'quickguess': 't',
                'fock:densityscreen': 'f',
                'lindep:n_dep': 0,
            }
        },
        # Note: Parsl sets --ntasks-per-node=1 in #SBATCH. For some reason, --ntasks in srun overrides it
        command=(f'srun -N {args.num_nodes} '
                 f'--ntasks={ranks_per_node * cores_per_node} '
                 f'--export=ALL,OMP_NUM_THREADS={1} '
                 '--cpu-bind=cores shifter nwchem PREFIX.nwi > PREFIX.nwo'),
    )

    # Make the Parsl configuration
    config = parsl.Config(
        app_cache=False,  # No caching needed
        retries=1,  # Will restart a job if it fails for any reason
        executors=[HighThroughputExecutor(
            label='launch_from_mpi_nodes',
            max_workers=1,
            provider=SlurmProvider(
                partition='regular',
                account='m1513',
                launcher=SimpleLauncher(),
                walltime='36:00:00',
                nodes_per_block=args.num_nodes,  # Number of nodes per job
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,  # Maximum number of jobs
                scheduler_options='#SBATCH --image=ghcr.io/nwchemgit/nwchem-702.mpipr.nersc:latest\n#SBATCH -C knl',
                worker_init=f'''
module load python
conda activate /global/project/projectdirs/m1513/lward/hydronet/env

module swap craype-{{${{CRAY_CPU_TARGET}},mic-knl}}
export OMP_NUM_THREADS={68 // args.ranks_per_node}
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131026
export MPICH_GNI_NUM_BUFS=80
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_RDMA_THRESHOLD=65536
export COMEX_MAX_NB_OUTSTANDING=6

which python
hostname
pwd
                    ''',
                cmd_timeout=120,
            ),
        )]
    )
    parsl.load(config)

    # Open the ASE database
    print(f'Submitting from database...')
    with connect('initial.db', type='db') as db:
        # Submit structures to Pars
        futures = []
        for filename, atoms in generate_structures():
            # Store some tracking information
            atoms.info['filename'] = filename  # Pass the file name along with the calculation
            atoms.info['basis'] = args.basis

            # Skip if this structure is already in the database
            if db.count(filename=filename, basis=args.basis) > 0:
                continue

            # Submit the calculation to run
            future = run_nwchem(atoms, calc, temp_path=None)
            futures.append(future)

            # Check if the total
            if args.max_to_run is not None and len(futures) >= args.max_to_run:
                break
        print(f'Submitted {len(futures)}')

        # Loop over the futures and store them if the complete
        n_failures = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc='completed'):
            # Get the result
            future: AppFuture = future
            future.exception()
            if future.exception() is not None:
                print(f'Failure for {future.task_def["args"].info["filename"]}')
                n_failures += 1
                continue
            atoms, runtime = future.result()

            # Store it
            db.write(atoms, **atoms.info, runtime=runtime)
        if n_failures > 0:
            print(f'Total failure count {n_failures}')
