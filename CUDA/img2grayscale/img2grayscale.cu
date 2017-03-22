#include <stdio.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define BLUE 0
#define GREEN 1
#define RED 2
#define B_WEIGHT 0.1140
#define G_WEIGHT 0.5870
#define R_WEIGHT 0.2989

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__ 
void ImageToGrayscale(unsigned char *d_img, unsigned char *d_gray, int cols, int rows) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    int i = row * cols + col;
    d_gray[i] = B_WEIGHT * d_img[i * 3 + BLUE] +
                G_WEIGHT * d_img[i * 3 + GREEN] +
                R_WEIGHT * d_img[i * 3 + RED];
  }
}

void SeqImageToGrayscale(unsigned char *h_img, unsigned char *h_gray, int cols, int rows) {
  for (int i = 0; i < rows*cols; i++) {
    h_gray[i] = B_WEIGHT * h_img[i * 3 + BLUE] +
                G_WEIGHT * h_img[i * 3 + GREEN] +
                R_WEIGHT * h_img[i * 3 + RED];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: img2grayscale.out <image_name>" << endl;
    return -1;
  }
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;

  Mat img = imread(argv[1]);

  if (!img.data) {
    cout << "Error reading image, it may not exist" << endl;
  }

  int width = img.cols;
  int height = img.rows;
  int N = width * height;
  size_t size_gray = N * sizeof(unsigned char);
  size_t size_color = N * 3 * sizeof(unsigned char);

  Mat gray_gpu(height, width, CV_8UC1);
  Mat gray_cpu(height, width, CV_8UC1);

  unsigned char *h_img = img.data;
  unsigned char *h_gray_img = (unsigned char *) malloc(size_gray);
  unsigned char *gray_img = (unsigned char *) malloc(size_gray);
  unsigned char *d_img;
  unsigned char *d_gray_img;

  cudaError_t err;
  err = cudaMalloc((void **) &d_img, size_color);
  checkCudaError(err);
  err = cudaMalloc((void **) &d_gray_img, size_gray);
  checkCudaError(err);

  // Image to grayscale in GPU
  startGPU = clock();

  err = cudaMemcpy(d_img, h_img, size_color, cudaMemcpyHostToDevice);
  checkCudaError(err);

  int block_size = 32;
  dim3 block_dim(block_size, block_size, 1);
  dim3 grid_dim(ceil(width / float(block_size)), ceil(height / float(block_size)), 1);

  ImageToGrayscale<<<grid_dim, block_dim>>>(d_img, d_gray_img, width, height);
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_gray_img, d_gray_img, size_gray, cudaMemcpyDeviceToHost);
  checkCudaError(err);

  endGPU = clock();

  // Image to grayscale in CPU
  startCPU = clock();

  SeqImageToGrayscale(h_img, gray_img, width, height);

  endCPU = clock();

  // Generating result images
  gray_gpu.data = h_gray_img;
  gray_cpu.data = gray_img;

  // Show results
  namedWindow("Image", WINDOW_NORMAL);
  namedWindow("Image to grayscale in GPU", WINDOW_NORMAL);
  namedWindow("Image to grayscale in CPU", WINDOW_NORMAL);
  imshow("Image", img);
  imshow("Image to grayscale in CPU", gray_cpu);
  imshow("Image to grayscale in GPU", gray_gpu);
  waitKey(0);

  // Generating times
  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Time in GPU: %.10f\n",gpu_time_used);
  cpu_time_used = ((double) (endCPU - startCPU)) / CLOCKS_PER_SEC;
  printf("Time in CPU: %.10f\n",cpu_time_used);
  printf("Acceleration: %.10fX\n", cpu_time_used / gpu_time_used);

  free(h_gray_img); free(gray_img);
  cudaFree(d_img); cudaFree(d_gray_img);
  return 0;
}
