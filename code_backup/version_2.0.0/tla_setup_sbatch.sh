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
# in the data folder with (static) name "setup_done_samples.csv"
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
done_samples="$data_path/setup_done_samples.csv"
sub_samples="$raw_path/setup_sub_samples.csv"

if [ ! -f $done_samples ]
then
    echo "Found NO previously pre-processed samples..."
    ncases=$(($(numlines $samples_files)))
    studtbl=$1

    echo "TLA_setup: Pre-processing ($ncases) samples in study <$1>" 
    mkdir -p log
    # run all samples in a slum array
    # >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
    steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH=$2,REDO=$3 src/tla_setup_sample.sh)

else
    # generate a list of samples not yet processed
    echo "Found some previously pre-processed samples..."
    grep -w -vFf $done_samples $samples_files > $sub_samples 
    ncases=$(($(numlines $sub_samples)))

    if [ $ncases -eq 0 ]
    then
        ncases=$(($(numlines $samples_files)))
        echo "TLA_setup: All ($ncases) samples are already pre-processed in study <$1>"     
    else
        echo "TLA_setup: Pre-processing remaining ($ncases) samples in study <$1>" 

	# create a temporary parameter file pointing to sub-sample list
        studtbl=${1/'.csv'/'_sub.csv'}    
        cp $1 $studtbl	
        sed -Ei "" "2 s/[^,]*/setup_sub_samples.csv/3" $studtbl

        # run all (undone) samples in a slum array
        mkdir -p log
        # >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
        steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$studtbl,GRAPH=$2,REDO=$3 src/tla_setup_sample.sh)

        # remove temporary file
        rm $studtbl
    fi
    
    # remove temporary file
    rm $sub_samples
fi

