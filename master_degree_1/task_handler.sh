#! /bin/bash
g++ -o mat_mul.out mat_mul.cpp
for j in {1..10}
do
for i in 10 100 200 400 600 800 1000 2000 4000
do
./mat_mul.out $i >> times.txt
done
done
cat times.txt
