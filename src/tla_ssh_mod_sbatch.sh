#!/bin/bash

IFS=','
mkdir -p log
SRC=$( dirname "$0")

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
ncases=$(($(wc -l < $samples_files) - 1))

echo "TLA_ssh: Processing ($ncases) samples in study <$1>" 

# run all samples in a slum array
steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1 {$SRC}tla_ssh_step.sh)

