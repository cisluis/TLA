#!/bin/bash

#SBATCH --job-name=TLA_setup
#SBATCH -N 1                         # number of compute nodes
#SBATCH -n 10                        # number of concurrently running jobs
#SBATCH --mem=256G                   # requests memory on the node
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1                    # Request one GPU
#SBATCH -t 0-0:05:00                 # upper bound time: d-hh:mm:ss
#SBATCH --export=NONE                # purge the job-submitting shell environment
#SBATCH --output=log/slurm-%A.%a.out # stdout file
#SBATCH --error=log/slurm-%A.%a.err  # stderr file
#SBATCH --mail-type=ALL              # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=lhcisner@asu.edu # send-to address

# Load the python interpreter
module load mamba/latest
source activate tlaenv

# index of sample in the sample table
I=$(($SLURM_ARRAY_TASK_ID - 1))

# run all setup steps in job array
python src/tla_setup_sample.py ${STUDY} $I ${GRAPH} ${REDO}

