#! /bin/bash
g++ -o seq.out seq_mul_flatten.cpp
g++ -o optimized.out ./optimization/miss_cache_seq_mul_flatten.cpp
rm seq_times.csv opt_times_miss_cache.csv
echo "size, time" >> seq_times.csv
echo "size, time" >> opt_times_miss_cache.csv
for j in {1..10}
do
for i in 500 1000 1500 2000
do
./seq.out $i >> seq_times.csv
./optimized.out $i >> opt_times_miss_cache.csv
done
done
echo "DONE"
