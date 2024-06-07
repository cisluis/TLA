#!/bin/bash

IFS=','
mkdir -p log

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"

ncases=$(($(wc -l < $samples_files) - 1))

echo "TLA_setup: Processing ($ncases) samples in study <$1>" 
# run all samples in a slum array
# >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH=$2,REDO=$3 src/tla_setup_sample.sh)

# run the setup summary
sums=$(sbatch --dependency=afterok:$steps --parsable --export=STUDY=$1 src/tla_setup_sum.sh)
