#!/bin/bash -l
#SBATCH -J AsLOV2ChiLifeGTNSample5000         # Job name
#SBATCH -p batch
#SBATCH -o /home/jsheppard/research/aslov2/Gd-sTPATCN/equilibrated/stride_1/output.o%j                  # stdout output file
#SBATCH -e /home/jsheppard/research/aslov2/Gd-sTPATCN/equilibrated/stride_1/error.e%j                   # stderr error file
#SBATCH -N 1                           # Total # of nodes (1 for serial)
#SBATCH -n 1                          # Total # of mpi tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 12:00:00                    # Run time (hh:mm:ss)
#SBATCH --mail-user=sheppard@ucsb.edu
#SBATCH --mail-type=all                # Send email at job start and job end

# Other commands follow #SBATCH directives...
module list
pwd
date
conda activate chilife_env

export MPLBACKEND=Agg

srun python -u compare_R1_rotlib_traj_GTNSample5000.py