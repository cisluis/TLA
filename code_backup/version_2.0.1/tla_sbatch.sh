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
# if some samples are already processed, they would be listed in a list 
# in the data folder with (static) name "done_samples.csv"
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files=$data_path/$name"_samples.csv"
done_samples="$data_path/tla_done_samples.csv"
sub_samples=$data_path/"tla_sub_samples.csv"

if [ ! -f $done_samples ]
then
    echo "Found NO previously processed samples..."
    ncases=$(($(numlines $samples_files)))
else
    echo "Found some previously processed samples..."
    # generate a list of samples not yet processed
    grep -w -vFf $done_samples $samples_files > $sub_samples 
    ncases=$(($(numlines $sub_samples)))
fi

if [ $ncases -eq 0 ]
then
    ncases=$(($(numlines $samples_files)))
    echo "TLA_setup: All ($ncases) samples are already processed in study <$1>"  
else
    echo "TLA: Processing remaining ($ncases) samples in study <$1>" 
    mkdir -p log
    # run all samples in a slum array
    # >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
    steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH=$2,REDO=$3 src/tla_sample.sh)
fi

# remove temporary files
rm $sub_samples