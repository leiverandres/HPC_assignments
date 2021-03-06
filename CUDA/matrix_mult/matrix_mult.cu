#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#include "utils.hpp"

using namespace std;

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__ 
void Multiplication(float *A, float *B, float *C, int A_ROWS, int A_COLS, int B_COLS) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < A_ROWS && col < B_COLS) {
    float sum = 0.0;
    for (int i = 0; i < A_COLS; i++) {
      sum += A[row * A_COLS + i] * B[i * B_COLS + col];
    }
    C[row * B_COLS + col] = sum;
  }
}

void PrintUsage(string program) {
  cout << "Usage: " << program << " A_ROWS A_COLS B_COLS [OPTIONS]" << endl;
  cout << "* Is not needed to pass B_ROWS because B_ROWS must be equal to A_COLS";
  cout << endl << "OPTIONS: " << endl;
  cout << "-p: Print matrices" << endl;
  cout << endl;
}

int main(int argc, char *argv[]) {
  int A_ROWS, A_COLS, B_ROWS, B_COLS;
  if (argc < 4) {
    PrintUsage(argv[0]);
    return -1;
  } else {
    A_ROWS = atoi(argv[1]);
    A_COLS = atoi(argv[2]);
    B_ROWS = A_COLS;
    B_COLS = atoi(argv[3]);
  }
  clock_t start, end;
  double time_used;
  
  size_t size_a = A_ROWS * A_COLS * sizeof(float);
  size_t size_b = B_ROWS * B_COLS * sizeof(float);
  size_t size_c = A_ROWS * B_COLS * sizeof(float);

  float *h_a = (float *) malloc(size_a);
  float *h_b = (float *) malloc(size_b);
  float *h_c = (float *) malloc(size_c);
  float *d_a;
  float *d_b;
  float *d_c;

  // Init matrices
  utils::InitMat(h_a, A_ROWS, A_COLS);
  utils::InitMat(h_b, B_ROWS, B_COLS);

  cudaError_t err;
  err = cudaMalloc((void **) &d_a, size_a);
  checkCudaError(err);
  err = cudaMalloc((void **) &d_b, size_b);
  checkCudaError(err);
  err = cudaMalloc((void **) &d_c, size_c);
  checkCudaError(err);

  // Multiplication in GPU
  start = clock();
  
  err = cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  checkCudaError(err);
  err = cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  checkCudaError(err);

  int block_size = 32;
  dim3 block_dim(block_size, block_size, 1);
  dim3 grid_dim(ceil((float)B_COLS / block_size), ceil((float)A_ROWS / block_size), 1);

  Multiplication<<<grid_dim, block_dim>>>(d_a, d_b, d_c, A_ROWS, A_COLS, B_COLS);
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
  checkCudaError(err);

  end = clock();
  
  // Show matriced
  if (argv[4]) {
    utils::ShowMat("A", h_a, A_ROWS, A_COLS);
    utils::ShowMat("B", h_b, B_ROWS, B_COLS);  
    utils::ShowMat("P_C", h_c, A_ROWS, B_COLS);    
  }
 
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC; 
  printf("%.10f ", time_used);  // time in GPU 
    // printf("%.10f\n", cpu_time_used / gpu_time_used); // acceleration

  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
