#!/bin/bash

# $dim $perm $size_of_data_type $num sub tensor 
PARAMS=("2 2 275000 2 1 2 4 3 4 32 0.02"\
)

for ((i=0;i<${#PARAMS[@]};++i)) ; do
        echo "Case $i"
        ./Nd_estimation ${PARAMS[$i]}
    done
