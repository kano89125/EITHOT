#!/bin/bash

# $dim $perm $size_of_data_type $num_sub_tensor $alpha 
PARAMS=(
        
)
filename='test_cases/'
case $1 in
    1) filename+='exp1/'
    ;;
    2) filename+='exp2/'
    ;;
    3) filename+='exp3/'
    ;;
    4) filename+='exp4/'
    ;;
    5) filename+='exp5/'
    ;;
    6) filename+='exp6/'
    ;;
esac 
filename+='*.txt'
# throughput='inplace_bench_throughput'
# total_time='inplace_bench'
# transpose_time='inplace_bench_wo_h2d2h'
for f in $filename
do
    echo $f
    n=1
    while read line; do 
    echo "Case $n"
    ./test_inplace $line
    # ./Nd_estimation $line
    n=$((n+1))
    echo ""
    done < $f

    # n=1
    # while read line; do 
    # echo "Case $n"
    
    # n=$((n+1))
    # echo ""
    # done < "proper_input.txt"
done 


# for f in $filename
# do
#     echo $f
# done
