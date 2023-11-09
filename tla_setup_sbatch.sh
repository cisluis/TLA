#!/bin/bash

IFS=','
mkdir -p log

# get samples_file name and determine number of samples to be processed
# if some samples are already processed, they would be listed in a list 
# in the data folder with (static) name "setup_done_samples.csv"
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
done_samples="$data_path/setup_done_samples.csv"
sub_samples="$raw_path/setup_sub_samples.csv"

if [ ! -f $done_samples ]
then
    echo "Found NO previously processed samples..."
    ncases=$(($(wc -l < $samples_files) - 1))
    studtbl=$1

    echo "TLA_setup: Processing ($ncases) samples in study <$1>" 
    # run all samples in a slum array
    # >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
    steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$1,GRAPH=$2,REDO=$3 src/tla_setup_sample.sh)

else
    echo "Found some previously processed samples..."
    # generate a list of samples not yet processed
    grep -vFf $done_samples $samples_files > $sub_samples 
   
    ncases=$(($(wc -l < $sub_samples) - 1))

    # create a temporary parameter file pointing to sub-sample list
    studtbl=${1/'.csv'/'_sub.csv'}    
    cp $1 $studtbl	
    sed -Ei "" "2 s/[^,]*/setup_sub_samples.csv/3" $studtbl

    if [ $ncases -eq 0 ]
    then
        echo "TLA_setup: All samples already processed in study <$1>"     
    else
        echo "TLA_setup: Processing remaining ($ncases) samples in study <$1>" 
        # run all (undone) samples in a slum array
        # >> use {$2, $3 = --graph, --redo} for optional graphing or redoing calculations 	
        steps=$(sbatch --array=1-$ncases --parsable --export=STUDY=$studtbl,GRAPH=$2,REDO=$3 src/tla_setup_sample.sh)
    fi

    # remove temporary files
    rm $sub_samples $studtbl
fi

