#!/bin/bash

# $dim $perm $size_of_data_type $num_sub_tensor $alpha 
PARAMS=(    "85937500 2 2 2 2 2 1 3 4 5 4 16 0.01 "\
              
)
for ((i=0;i<${#PARAMS[@]};++i)) ; do
        echo "Case $i"
        ./test_inplace ${PARAMS[$i]}
    done
