#!/bin/bash

IFS=','

# get samples_file name and determine number of samples to be processed
# if some samples are already processed, they would be listed in a list 
# in the data folder with (static) name "setup_done_samples.csv"
read name raw_path raw_samples_table raw_classes_table data_path rest < <(sed "1d" "$1")
samples_files="$raw_path/$raw_samples_table"
done_samples="$data_path/setup_done_samples.csv"
sub_samples="$data_path/setup_sub_samples.csv"

if [ ! -f $done_samples ]
then
    ncases=$(($(wc -l < $samples_files) - 1))
    studtbl=$1
else
    # generate a list of samples not yet processed
    grep -vFf $done_samples $samples_files > $sub_samples 
   
    ncases=$(($(wc -l < $sub_samples) - 1))

    # create a temporary parameter file pointing to sub-sample list
    studtbl=${1/'.csv'/'_sub.csv'}    
    cp $1 $studtbl	
    sed -Ei "" "2 s/[^,]*/setup_sub_samples.csv/3" $studtbl
fi

if [ $ncases -eq 0 ]
then
    echo "TLA Setup: All samples already processed in study <$1>" 
else
    echo "TLA Setup: Processing ($ncases) samples in study <$1>" 

    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate tlaenv

    # run all samples in study
    for (( I=0; I<$ncases; I++ ))
       do
	  python src/tla_setup_sample.py $studtbl $I $2 $3
       done
fi

# remove temporary files
rm $sub_samples $studtbl

