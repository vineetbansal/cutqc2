## 1. Create conda environment

conda create --name CutQCSummer2025 python=3.11
conda activate CutQCSummer2025
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi
pip install numpy qiskit matplotlib pydot scipy tqdm pylatexenc scikit-learn tensorflow networkx torch qiskit-aer psutil

# Conda Load Adroit Commands

module load anaconda3/2024.6
conda activate CutQCSummer2025
module load gurobi/12.0.0

# Delete Conda Environement so fresh env can be made again

conda remove -n CutQCSummer2025 --all

## Allocate an interactive shell

salloc --nodes=1 --ntasks=4 --cpus-per-task=12 --mem=4G --time=01:20:00


## Export for quite tensorflow

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

## Torch Verbosity
export NCCL_DEBUG=$debug_level # Turn on nccl debuging outputs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_SUBSYS=ALLOC
