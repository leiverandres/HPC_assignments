#! /bin/bash
g++ -o proc_mul.out multi_process_mat_mult.cpp
for j in {1..10}
do
for i in 10 100 200 400 600 800 1000 2000 3000 4000
do
./proc_mul.out $i >> proc_times.txt
done
done
cat proc_times.txt
