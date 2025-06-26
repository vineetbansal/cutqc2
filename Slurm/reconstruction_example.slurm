#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=4       # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                  # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4              # number of gpus per node
#SBATCH --time=00:00:50           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails

# Setup for Multi-node Workload
export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # Master node is first host in list of hostnames
export MASTER_ADDR=$master_addr

# Disable verbose tensorlfow output
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

# Sanity Print
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# Load Modules
module purge
module load anaconda3/2024.2
conda activate CutQCSummer2025
module load gurobi/12.0.0

srun python reconstruction_example.py

