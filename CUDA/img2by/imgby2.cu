#include <iostream>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

__global__ void matByConst(unsigned char *img, unsigned char *result, int alpha, int cols, int rows) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    int idx = row * cols + col;
    result[idx] = img[idx] * alpha;
  }
}

void seqMatByConst(unsigned char *img, unsigned char *result, int alpha, int cols, int rows) {
  for (int i = 0; i < cols*rows; i++) {
    result[i] = img[i] * alpha;
  }
}

void checkCudaError(cudaError_t err) { 
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: img2grayscale.out <image_name>" << endl;
    return -1;
  }
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
  Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if (!img.data) {
    cout << "Error reading image, it may not exist" << endl;
  }

  int N = img.rows * img.cols;
  int num_bytes = N * sizeof(unsigned char);

  unsigned char *h_img =  img.data;
  unsigned char *h_result = (unsigned char *) malloc(num_bytes);
  unsigned char *d_img;
  unsigned char *d_result;
  unsigned char *h_result_test = (unsigned char *) malloc(num_bytes);

  cudaError_t err; 
  err = cudaMalloc((void **) &d_img, num_bytes);
  checkCudaError(err);
  err = cudaMalloc((void **) &d_result, num_bytes);
  checkCudaError(err);

  startGPU = clock();
  err = cudaMemcpy(d_img, h_img, num_bytes, cudaMemcpyHostToDevice);
    
  int block_size = 32;
  
  dim3 grid_dim(ceil(img.cols / float(block_size)), ceil(img.rows / float(block_size)), 1);
  dim3 block_dim(block_size, block_size, 1);
  matByConst<<<grid_dim, block_dim>>>(d_img, d_result, 2, img.cols, img.rows);
 
  err = cudaMemcpy(h_result, d_result, num_bytes, cudaMemcpyDeviceToHost);
  checkCudaError(err);
  endGPU = clock();

  startCPU = clock();
  seqMatByConst(h_img, h_result_test, 2, img.cols, img.rows);
  endCPU = clock();

  Mat result_gpu;
  result_gpu.create(img.size().height, img.size().width, img.type()); 
  result_gpu.data = h_result;
  
  Mat result_cpu;
  result_cpu.create(img.size().height, img.size().width, img.type());
  result_cpu.data = h_result_test;
  
  namedWindow("Image in grayscale", WINDOW_NORMAL);
  namedWindow("Image after miltiplication in GPU", WINDOW_NORMAL);
  namedWindow("Image after multiplication in CPU", WINDOW_NORMAL);
  imshow("Image in grayscale", img);
  imshow("Image after multiplication in CPU", result_cpu);
  imshow("Image after multiplication in GPU", result_gpu);
  waitKey(0);
  
  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Time in GPU: %.10f\n",gpu_time_used);
  cpu_time_used = ((double) (endCPU - startCPU)) / CLOCKS_PER_SEC;
  printf("Time in CPU: %.10f\n",cpu_time_used);
  printf("Acceleration: %.10fX\n", cpu_time_used / gpu_time_used);

  free(h_result); free(h_result_test);
  cudaFree(d_img); cudaFree(d_result);
  return 0;
}
