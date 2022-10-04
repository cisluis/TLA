#!/bin/bash

#SBATCH --job-name=TLA_setup_array
#SBATCH --time=0-0:15:00             # upper bound time: d-hh:mm:ss
#SBATCH --partition=debug
#SBATCH --qos=wildfire
#SBATCH --ntasks=10                  # number of concurrently running jobs
#SBATCH --mem=0                      # requests all the memory on the node
#SBATCH --export=NONE                # purge the job-submitting shell environment
#SBATCH --output=log/slurm-%A.%a.out # stdout file
#SBATCH --error=log/slurm-%A.%a.err  # stderr file

# Load the python interpreter
module purge
module load anaconda/py3
source activate tlaenv

SRC=$( dirname "$0")

# index of sample in the sample table
I=$(($SLURM_ARRAY_TASK_ID - 1))

# run all setup steps in job array
python {$SRC}tla_setup_sample.py ${STUDY} $I ${GRAPH} ${REDO}

