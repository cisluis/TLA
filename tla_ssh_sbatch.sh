#!/bin/bash

IFS=','
mkdir -p log

# get samples_file name and determine number of samples to be processed
# if some samples are already processed, they would be listed in a list 
# in the data folder with (static) name "done_samples.csv"
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files=$data_path/$name"_samples.csv"
done_samples="$data_path/ssh_done_samples.csv"
sub_samples="$data_path/ssh_sub_samples.csv"

if [ ! -f $done_samples ]
then
    ncases=$(($(wc -l < $samples_files) - 1))
else
    # generate a list of samples not yet processed
    grep -vFf $done_samples $samples_files > $sub_samples 
    ncases=$(($(wc -l < $sub_samples) - 1))
fi

if [ $ncases -eq 0 ]
then
    echo "TLA SSH: All samples already processed in study <$1>" 
else
    echo "TLA SSH: Processing ($ncases) samples in study <$1>" 

    # run all samples in a slum array
    steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH='',REDO='' src/tla_ssh_sample.sh)
fi

# remove temporary files
rm $sub_samples