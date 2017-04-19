#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define checkCudaErr(err)                                                              \
  if ((err) != cudaSuccess) {                                                          \
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(EXIT_FAILURE);                                                                \
}

using namespace std;
using namespace cv;

__constant__ float kernel_x[9];
__constant__ float kernel_y[9];

__global__
void convolutionKernel(uchar *img, float *result, int width, int height, 
                       int kernel_size, bool use_x_kernel) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = kernel_size / 2;
  if (row < height && col < width) {
    float acum = 0.0;
    int start_i = row - offset;
    int start_j = col - offset;

    for (int k = 0; k < kernel_size; k++) {
      for (int l = 0; l < kernel_size; l++) {
        int kernel_row = start_i + k;
        int kernel_col = start_j + l;
        if (kernel_row >= 0 && kernel_col >= 0 && 
            kernel_row < height && kernel_col < width) {
          if (use_x_kernel) {
            acum += img[kernel_row * width + kernel_col] * 
                    kernel_x[k * kernel_size + l];
          } else {
            acum += img[kernel_row * width + kernel_col] * 
                    kernel_y[k * kernel_size + l];
          }
        }
      }
    }
    result[row * width + col] = acum;
  }
}

__global__
void gradientMagnitudeKernel(float *x_derivate, float *y_derivate, 
                             float *result, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < height && col < width) {
    int idx = row * width + col;
    float x = x_derivate[idx];
    float y = y_derivate[idx];
    result[idx] = sqrtf((x * x) + (y * y));
  } 
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "You must provide a image name" << endl;
    return -1;
  }

  cudaError_t err;
  
  Mat img = imread(argv[1], 0);
  
  int width = img.cols;
  int height = img.rows;
  size_t img_bytes_uc = width * height * sizeof(uchar);
  size_t img_bytes_f  = width * height * sizeof(float);

  uchar *h_img = img.data;
  float *h_sobel = (float *) malloc(img_bytes_f);
  uchar *d_img;
  float *d_sobel;
  float *d_gradient_x, *d_gradient_y;
  
  err = cudaMalloc((void **) &d_img, img_bytes_uc);
  checkCudaErr(err);
  err = cudaMalloc((void **) &d_sobel, img_bytes_f);
  checkCudaErr(err);
  err = cudaMalloc((void **) &d_gradient_x, img_bytes_f);
  checkCudaErr(err);
  err = cudaMalloc((void **) &d_gradient_y, img_bytes_f);
  checkCudaErr(err);
  
  err = cudaMemcpy(d_img, h_img, img_bytes_uc, cudaMemcpyHostToDevice);
  checkCudaErr(err); 

  int kernel_size = 3;
  size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
  float h_kernel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_kernel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  err = cudaMemcpyToSymbol(kernel_x, h_kernel_x, kernel_bytes);
  checkCudaErr(err);
  err = cudaMemcpyToSymbol(kernel_y, h_kernel_y, kernel_bytes);
  checkCudaErr(err);
  
  int block_size = 32;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);
  
  convolutionKernel<<<dim_grid, dim_block>>>
        (d_img, d_gradient_x, width, height, kernel_size, true);
  cudaDeviceSynchronize();
  
  convolutionKernel<<<dim_grid, dim_block>>>
        (d_img, d_gradient_y, width, height, kernel_size, false);
  cudaDeviceSynchronize();
  
  gradientMagnitudeKernel<<<dim_grid, dim_block>>>
        (d_gradient_x, d_gradient_y, d_sobel, width, height);
  cudaDeviceSynchronize();
  
  err = cudaMemcpy(h_sobel, d_sobel, img_bytes_f, cudaMemcpyDeviceToHost);
  checkCudaErr(err);
  
  Mat sobel(height, width, CV_32F, h_sobel);
  convertScaleAbs(sobel, sobel);
  
  free(h_sobel);
  cudaFree(d_img); cudaFree(d_sobel); cudaFree(d_gradient_x);
  cudaFree(d_gradient_y);
  
  //namedWindow("Original image", WINDOW_NORMAL);
  namedWindow("Filtered image", WINDOW_NORMAL);
  //imshow("Original image", img);
  imshow("Filtered image", sobel);
  waitKey(0);
  return 0;
}
