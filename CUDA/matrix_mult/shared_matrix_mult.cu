#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#include "utils.hpp"

#define TILE_WIDTH 32

using namespace std;

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__
void matrixMult(float *matA, float *matB, float *matC, int n) {
  __shared__ float shared_matA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shared_matB[TILE_WIDTH][TILE_WIDTH];
  
  int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
  
  float sum = 0;
  for (int m = 0; m * TILE_WIDTH < n; m++) {
    if (i < n && m * TILE_WIDTH + threadIdx.x < n) {
      shared_matA[threadIdx.y][threadIdx.x] = matA[i * n + (m * TILE_WIDTH + threadIdx.x)];
    } else {
      shared_matA[threadIdx.y][threadIdx.x] = 0;
    }
    if (m * TILE_WIDTH + threadIdx.y < n && j < n) {
      shared_matB[threadIdx.y][threadIdx.x] = matB[(m * TILE_WIDTH + threadIdx.y) * n + j];
    } else {
      shared_matB[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      sum += shared_matA[threadIdx.y][k] * shared_matB[k][threadIdx.x];
    }
    __syncthreads();
  }
    
  if (i < n && j < n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    matC[row * n + col] = sum;
  }
}

int main(int argc, char **argv) {
 int A_ROWS, A_COLS, B_ROWS, B_COLS;  
  if (argc < 4) {
     cout << "Usage: " << argv[0] << " a_rows a_cols b_cols" << endl; 
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

  int block_size = TILE_WIDTH;
  dim3 block_dim(block_size, block_size, 1);
  dim3 grid_dim(ceil((float)B_COLS / block_size), ceil((float)A_ROWS / block_size), 1); 
   
  matrixMult<<<grid_dim, block_dim>>>(d_a, d_b, d_c, A_ROWS);

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
