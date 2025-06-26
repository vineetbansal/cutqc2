# Explaining the Reconstruction Example Script

This document explains the basic environment setup for distributed reconstruction, along with how to run both `cut_and_eval_example.py` and `reconstruction_example.py`.

I assume SLURM is used to execute Python scripts; however, running without SLURM is not much different, as certain environment variables will need to be manually set. More details on this are shown below.

## 1 - Setup

#### Required Environment Variables

Prior to distributed reconstruction, the following [local environment variables](https://pytorch.org/tutorials//intermediate/dist_tuto.html?highlight=init_process_group#:~:text=MASTER_PORT%3A%20A%20free,or%20a%20worker.) must be set:

- MASTER_PORT: A free port on the machine that will host the process with rank 0
- MASTER_ADDR: IP address of the machine that will host the process with rank 0
- WORLD_SIZE: The total number of processes, so that the master knows how many workers to wait for
- RANK: Rank of each process, so they will know whether it is the master or a worker


#### SLURM

In the given example, [SLURM](https://slurm.schedmd.com/), a compute cluster scheduler, is used to automate the process of finding a free port for the master and setting the respective environment variables.

When executed, the `reconstruction_example.slurm` SLURM script implicitly sets rank and GPUs per node environment variables as:

- SLURM_GPUS_ON_NODE
- SLURM_PROCID

> (note this is on each respective machine)

Moreover, the address and port information is explicitly set in 'reconstruct.slurm' with:

  ```console
    # Each node is set the following ENV VARS
    export MASTER_PORT=$(get_free_port)  # Get a free Port
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  
    export MASTER_ADDR=$master_addr
  ```

### Without Slurm 

The example can be run without SLURM by manually setting the environment variables and executing an instance of the example on each machine. Note that this is true only for distributed reconstruction, i.e., cut and evaluation should be done on a single, separate, isolated process.

## 2 - Running 'cut_and_eval_example.py'

Before reconstruction can be performed, any given circuit must first be cut and evaluated, and the results saved to a `.pkl` file. This can be done using `cutqc.save_cutqc_obj(filename)` as seen in the `cut_and_eval_example.py` file.

Run `cut_and_eval.py` using the corresponding SLURM script:

```
    sbatch cut_and_eval.slurm
```

Running the SLURM script should produce an output file containing (or printed to console, if running without SLURM):

```
    --- Cut --- 
    Set parameter GURO_PAR_SPECIAL
    Set parameter TokenServer to value "license.rc.princeton.edu"
    --- Evaluate ---
    --- Dumping CutQC Object into adder_example.pkl ---
    Completed
```

## 3 - Running 'reconstruction_example.py'

Once the example adder circuit is cut and the CutQC instance is saved as a pickle file, distributed reconstruction can be performed.

### Breakdown of example

```{.Python .numberLines .lineAnchors}
import os
from cutqc.main import CutQC 

# Environment variables
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK    = int(os.environ["SLURM_PROCID"])
WORLD_SIZE    = int(os.environ["WORLD_SIZE"])

if __name__ == "__main__":
    full_path       = 'adder_example.pkl'
    compute_backend = 'GPU'
    comm_backend    = 'nccl'
    
    # Load CutQC Instance from Pickle
    print(f'--- Running {full_path} ---')
    cutqc = CutQC(
        pytorch_distributed = True,
        reconstruct_only = True,
        load_data        = full_path,
        compute_backend  = compute_backend,
        comm_backend     = comm_backend,
        gpus_per_node = GPUS_PER_NODE,
        world_rank    = WORLD_RANK,
        world_size    = WORLD_SIZE
    )

    # Initiate Reconstruct
    compute_time = cutqc.build(mem_limit=32, recursion_depth=1)
    approximation_error = cutqc.verify()

    print('--- Reconstruction Complete ---')    
    print("Total Reconstruction Time:\t{}".format(compute_time))
    print("Approximation Error:\t{}".format(approximation_error))
    cutqc.destroy_distributed()    
```

#### Lines 5 - 7:

  ```python
    GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
    WORLD_RANK    = int(os.environ["SLURM_PROCID"])
    WORLD_SIZE    = int(os.environ["WORLD_SIZE"])
  ```

As mentioned in the first section, distributed reconstruction requires process information. On initialization of a CutQC object, only the World rank, world size, and GPUs per machine are required to be passed.

#### Lines 10 - 12:

  ```python
      full_path       = 'adder_example.pkl'
      compute_backend = 'GPU'
      comm_backend    = 'nccl'
  ```

`full_path` should be the full path to the pickled CutQC object used to cut and evaluate the original target circuit.

`compute_backend` is the device backend used. In this case, GPU is used but CPU is also possible. In cases of memory-intensive subcircuit reconstruction problem instances, CPU may be necessary.

`comm_backend` is the [communication backend](https://pytorch.org/docs/stable/distributed.html), which facilitates the communication of data between nodes during computation/execution. In this case, the communication backend used is [NVIDIA's NCCL](https://developer.nvidia.com/nccl).

#### Lines 16 - 25:

  ```python
      cutqc = CutQC(
          pytorch_distributed = True,
          reconstruct_only = True,
          load_data        = full_path,
          compute_backend  = compute_backend,
          comm_backend     = comm_backend,
          gpus_per_node = GPUS_PER_NODE,
          world_rank    = WORLD_RANK,
          world_size    = WORLD_SIZE
      )
  ```

A new CutQC object is created for reconstruction. The previous CutQC instance, used to cut and evaluate, is loaded internally so that the master node can send the partitioned workloads to each worker.

Reconstruct only must be passed as `True` too ensure CutQC does not attempt to cut and instead initializes for reconstruction. The `pytorch_distributed` parameter indicates to CutQC which computaitonal framework too use (Tensorflow, or Pytorch);for multinode distributed reconstruction, it must be passed as True. 

#### Lines 28 - 29:

  ```python
      compute_time = cutqc.build(mem_limit=32, recursion_depth=1)
      approximation_error = cutqc.verify()
  ```

Once the CutQC object is instantiated, the reconstruction process can be initiated by calling build.

In addition to the explicitly passed memory limit, there is an implicit memory limit imposed by the compute device itself. In some cases, the partitioned workload may exceed the capacity of each respective GPU; if this occurs, then the distributed graph contractor will fail with the message 'Error 2006: Batch of size $M$, too large for GPU device of size N', where M is the size of batch and $N$ is memory capacity of a GPU. 

A simple solution is to increase the number of GPU nodes used; Alternatively, a compute device type, like CPU, with more memory can be used instead. 

Finally, the verify method is called in the example as a sanity check which computes the subcircuit reconstruction using TensorFlow on a single node. This can be removed, as it has no practical effect on reconstruction.

#### Lines 34: 

  ```python
      cutqc.destroy_distributed()    
  ```

When reconstruction is complete, resources can be freed by calling destroy_distributed.

### Executing and Output

Once the circuit is cut and the results are computed, you can run parallel reconstruction by calling the SLURM script:

```
    sbatch dist_driver.slurm
```

Running the SLURM script should produce an output file containing:

```
    MASTER_ADDR=adroit-h11g3
    MASTER_PORT=31179
    WORLD_SIZE=2
    --- Running adder_example.pkl ---
    self.parallel_reconstruction: True
    Worker 1, compute_device: cuda:1
    --- Running adder_example.pkl ---
    self.parallel_reconstruction: True
    Worker 0, compute_device: cuda:0
    subcircuit_entry_length: [32, 32]
    LEN(DATASET): 16
    NUMBER BATCHES: 1
    Compute Time: 1.5637431228533387
    Approximate Error: 1.2621774483536279e-29
    verify took 0.011
    --- Reconstruction Complete ---
    Total Reconstruction Time:	1.5637431228533387
    Approximation Error:	1.2621774483536279e-29
    DESTROYING NOW! 1.5637431228533387
    WORKER 1 DYING
```