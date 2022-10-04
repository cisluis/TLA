#!/bin/bash

SRC=$( dirname "$0")
IFS=','

# get samples_file name and determine number of samples to be processed
read name raw_path raw_samples_table rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
ncases=$(($(wc -l < $samples_files) - 1))

echo "TLA_setup: Processing ($ncases) samples in study <$1>" 

/opt/anaconda3/etc/profile.d/conda.sh
conda activate tlaenv

# run all samples in study
for (( i=0; i<$ncases; I++ ))
do
	python $SRC"tla_setup_sample.py" $1 $i $2 $3
done


# run the setup summary
python $SRC"tla_setup_sum.py" $1

