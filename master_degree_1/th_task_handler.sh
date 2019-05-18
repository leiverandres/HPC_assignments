#! /bin/bash
g++ -o th_mul.out -std=c++17 -pthread threads_mat_mul.cpp
for j in {1..10}
do
for i in 10 100 200 400 600 800 1000 2000 3000 4000
do
./th_mul.out $i >> th_times.txt
done
done
cat th_times.txt
