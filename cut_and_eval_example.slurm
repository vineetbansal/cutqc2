#!/bin/bash
#SBATCH --output='cut_and_eval_example.out'
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=1       # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                 # memory per cpu-core (4G is default)
#SBATCH --time=00:00:55           # total run time limit (HH:MM:SS)

# Load Modules
module purge
module load anaconda3/2024.2
conda activate CutQCSummer2025
module load gurobi/12.0.0

python3 cut_and_eval_example.py 

