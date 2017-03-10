#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

__global__ void matByConst(unsigned char *img, int alpha, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    img[i] *= 2;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: img2grayscale.out <image_name>" << endl;
    return -1;
  }

  Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  Mat result(img.size(), CV_8UC1);
  unsigned char *h_img = (unsigned char *) img.data;
  unsigned char *d_img;
  
  int N = img.rows * img.cols;
  size_t num_bytes = N * sizeof(unsigned char);
  int block_size = 256;
  int blocks = ceil(N / block_size);
 
  cudaMalloc((void **) &d_img, num_bytes);
  cudaMemcpy(d_img, h_img, num_bytes, cudaMemcpyHostToDevice);

  matByConst<<<blocks, block_size>>>(d_img, 2, N);
  
  cudaError_t err = cudaMemcpy((unsigned char *) &result.data, d_img, num_bytes, cudaMemcpyDeviceToHost);
  
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  } 
  imshow("Image in grayscale", img);
  waitKey(0);
  imshow("Image after multiplication", result);
  cudaFree(d_img);
  return 0;
}

