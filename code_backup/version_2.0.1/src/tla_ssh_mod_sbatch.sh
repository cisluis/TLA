#!/bin/bash

IFS=','
mkdir -p log

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files=$data_path/$name"_samples.csv"

ncases=$(($(wc -l < $samples_files) - 1))

echo "TLA SSH: Processing ($ncases) samples in study <$1>" 
# run all samples in a slum array
steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH='',REDO='' src/tla_ssh_sample.sh)


