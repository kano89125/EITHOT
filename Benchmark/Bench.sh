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
if test "$1" -eq 1;
then
	for f in $filename
    do
        if test -f "proper_input.txt"; then
	    rm "proper_input.txt"
	fi
        echo $f
        save_path=${f:0:-4}
        if test -d $save_path; 
	    then
		rm -r -f $save_path
	fi
	mkdir $save_path
        n=1
        while read line; do 
        echo "Case $n"

        ./test_inplace $line
        n=$((n+1))
        echo ""
        done < $f
        mv 'inplace_bench.txt' $save_path/
	mv 'inplace_bench_throughput.txt' $save_path/
	mv 'Statistic.txt' $save_path/
    done 
else
    for f in $filename
    do
    	if test -f "proper_input.txt"; then
	    rm "proper_input.txt"
	    fi
        echo $f
        save_path=${f:0:-4}
        if test -d $save_path; 
	    then
		rm -r -f $save_path
	fi
	mkdir $save_path
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
	python3 "Statistic.py"
	mv 'inplace_bench.txt' $save_path/
	mv 'inplace_bench_throughput.txt' $save_path/
	mv 'Statistic.txt' $save_path/
    done     
fi
# for f in $filename
# do
#     echo $f
# done
