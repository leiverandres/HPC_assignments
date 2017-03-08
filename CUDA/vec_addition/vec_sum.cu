#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#define N 1024

void seqSum(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i++)
    c[i] = a[i] + b[i];
}

void fillRandomVec(int *x, int size) {
  for (int i = 0; i < size; i++)
    x[i] = rand() % 50;
}

void showVec(int *x, int size) {
  for (int i = 0; i < size; i++)
    printf("%d ", x[i]);
  printf("\n");
}

void compareResults(int *ans1, int *ans2, int size) {
  for (int i = 0; i < size; i++) {
    if (ans1[i] != ans2[i]) {
       printf("Sum comparison failed at %d index\n", i);
       return;
    }
  }
  printf("Answers are the same");
}

__global__ void deviceAddVector(int *d_a, int *d_b, int *d_c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    d_c[i] = d_a[i] + d_b[i];
    //  printf("Tread %d make sum %d + %d = %d", i, d_a[i], d_b[i], d_c[i]);
  }
}

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 
}

int main(int argc, char *argv[]) {
  int *h_a, *h_b, *h_c1, *h_c2;
  int *d_a, *d_b, *d_c;
  size_t num_bytes = N * sizeof(int);
  
  h_a = (int *) malloc(num_bytes);
  h_b = (int *) malloc(num_bytes);
  h_c1 = (int *) malloc(num_bytes);
  h_c2 = (int *) malloc(num_bytes);
  
  fillRandomVec(h_a, N);
  fillRandomVec(h_b, N);
  
  cudaError_t err_a = cudaMalloc((void **) &d_a, num_bytes);
  cudaError_t err_b = cudaMalloc((void **) &d_b, num_bytes);
  cudaError_t err_c = cudaMalloc((void **) &d_c, num_bytes);  
  checkCudaError(err_a);
  checkCudaError(err_b);
  checkCudaError(err_c);

  cudaError_t err_cpy_a = cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
  cudaError_t err_cpy_b = cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);

  checkCudaError(err_cpy_a);
  checkCudaError(err_cpy_b);

  seqSum(h_a, h_b, h_c1, N);

  // showVec(h_a, N); showVec(h_b, N); showVec(h_c1, N);

  int block_size = min(256, N);
  int num_blocks = ceil(N / block_size);
  printf("%d blocks, %d per threads per block\n", num_blocks, block_size);
  deviceAddVector<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);

  cudaError_t err_cpy_c = cudaMemcpy(h_c2, d_c, num_bytes, cudaMemcpyDeviceToHost);
  
  checkCudaError(err_cpy_c);
  // showVec(h_c2, N);
  compareResults(h_c1, h_c2, N);
  
  free(h_a); free(h_b); free(h_c1); free(h_c2);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
