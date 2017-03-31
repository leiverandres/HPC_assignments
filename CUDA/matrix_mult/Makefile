gpu_exec_name = ./matrix_mult.out
cpu_exec_name = ./seq_matrix_mult.out
shared_exec_name = ./shared_matrix_mult.out
num_tests = 7

build-files: cuda seq shared

cuda : matrix_mult.cu
	nvcc matrix_mult.cu -o $(gpu_exec_name)

seq : seq_matrix_mult.cpp
	g++ seq_matrix_mult.cpp -o $(cpu_exec_name)

shared: shared_matrix_mult.cu
	nvcc shared_matrix_mult.cu -o ${shared_exec_name}

test-gpu : test.js
	node test.js $(cpu_exec_name) $(gpu_exec_name) $(num_tests) cpu gpu

test-shared : test.js
	node test.js ${gpu_exec_name} ${shared_exec_name} ${num_tests} gpu shared_gpu

