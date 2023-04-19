#!/bin/bash

# $dim $perm $size_of_data_type $num_sub_tensor $alpha 
PARAMS=(    "2 2 275000 2 1 2 4 3 4 4 0.01 "\
              
)
for ((i=0;i<${#PARAMS[@]};++i)) ; do
        echo "Case $i"
        ./test_inplace ${PARAMS[$i]}
    done
