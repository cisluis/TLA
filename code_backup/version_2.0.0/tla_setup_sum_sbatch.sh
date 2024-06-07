#!/bin/bash

IFS=','

# function to count the number of lines in a file
# (accounts for files that don't end with a newline character, from Mac OS) 
numlines () {
    local eofn=$(($(tail -n 1 $1 | wc -l)))
    local nlines=$(($(wc -l < $1) - $eofn)) 
    echo $nlines
}

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
ncases=$(($(numlines $samples_files)))

echo "TLA_setup: Processing ($ncases) samples in study <$1>" 

# run the setup summary
mkdir -p log
sums=$(sbatch --parsable --export=STUDY=$1,CATEGORY=$2 src/tla_setup_sum.sh)
