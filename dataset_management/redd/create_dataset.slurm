#!/bin/bash
#SBATCH -J gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu2
#SBATCH -o out.%j
#SBATCH -e err.%j
cd  $SLURM_SUBMIT_DIR
srun python3 create_trainset_redd.py --data_dir ./low_freq --appliance_name washingmachine --aggregate_mean 522 --aggregate_std 814 --save_path ./washingmachine/ > washingmachine.log

