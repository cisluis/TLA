#!/bin/bash -l

#SBATCH --job-name=TLA_setup_sumarize 
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks=1                   # number of tasks (default: allocates 1 core per task)
#SBATCH --partition=serial 
#SBATCH --qos=normal
#SBATCH --time=0-0:15:00             # upper bound time: d-hh:mm:ss
#SBATCH --output=log/slurm-%A.%a.out # stdout file
#SBATCH --error=log/slurm-%A.%a.err  # stderr file
#SBATCH --export=NONE                # Purge the job-submitting shell environment
#SBATCH --mem=0                      # Requests all the memory on the node

# Load the python interpreter
module purge
module load anaconda/py3
source activate tlaenv

# summarize study statistics 
python src/tla_sum.py ${STUDY}
