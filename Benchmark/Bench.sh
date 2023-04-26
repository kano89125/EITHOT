#!/bin/bash

# $dim $perm $size_of_data_type $num_sub_tensor $alpha 


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
if $1 == 1
then
	for f in $filename
    do
        if test -f "proper_input.txt"; then
	    rm "proper_input.txt"
	fi
        echo $f
        n=1
        while read line; do 
        echo "Case $n"

        ./test_inplace $line
        n=$((n+1))
        echo ""
        done < $f
    done 
else
    for f in $filename
    do
    	if test -f "proper_input.txt"; then
	    rm "proper_input.txt"
	fi
        echo $f
        n=1
        while read line; do 
        echo "Case $n"

        ./Nd_estimation $line
        n=$((n+1))
        echo ""
        done < $f

        n=1
        while read line; do 
        echo "Case $n"
        ./test_inplace $line
        n=$((n+1))
        echo ""
        done < "proper_input.txt"
    done     
fi
# for f in $filename
# do
#     echo $f
# done
