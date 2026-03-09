#!/bin/bash

#SBATCH --job-name=spc_run
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00


/fs/nexus-scratch/samern/conda/envs/sam_llm/bin/python run_experiment.py 
