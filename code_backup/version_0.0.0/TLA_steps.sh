#!/bin/bash

for (( counter=0; counter<28; counter=counter+1 ))
do
    echo -n "$counter "
    export SETN="$counter"
    jupyter nbconvert --to notebook --execute --allow-errors TLA_scarce-Copy2.ipynb --output TLA_scarce_2_"$counter".ipynb

done
printf "\n"