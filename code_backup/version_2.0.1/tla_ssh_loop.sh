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
done_samples="$data_path/ssh_done_samples.csv"
sub_samples="$data_path/ssh_sub_samples.csv"

if [ ! -f $done_samples ]
then
    echo "Found NO previously pre-processed samples..."
    ncases=$(($(numlines $samples_files)))
else
    # generate a list of samples not yet processed
    echo "Found some previously pre-processed samples..."
    grep -w -vFf $done_samples $samples_files > $sub_samples 
    ncases=$(($(numlines $sub_samples)))
fi

if [ $ncases -eq 0 ]
then
    ncases=$(($(numlines $samples_files)))
    echo "TLA_SSH: All ($ncases) samples are already pre-processed in study <$1>"
else
    echo "TLA_SSH: Processing remaining ($ncases) samples in study <$1>" 

    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate tlaenv

    # run all samples in study
    for (( I=0; I<$ncases; I++ ))
    do
	python src/tla_ssh_sample.py $1 $I $2 $3
    done
fi

# remove temporary files
rm $sub_samples







