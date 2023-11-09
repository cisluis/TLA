#!/bin/bash

IFS=','
mkdir -p log

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"

#rm $done_samples $sub_samples
ncases=$(($(wc -l < $samples_files) - 1))

echo "TLA_setup_sum: Processing ($ncases) samples in study <$1>" 

# run the setup summary
sums=$(sbatch --parsable --export=STUDY=$1 src/tla_setup_sum.sh)
