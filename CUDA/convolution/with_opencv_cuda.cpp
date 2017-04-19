#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {

  Mat h_src;
  Mat h_sobel;
  int ddepth = CV_32F;

  if (argc < 2) {
    cout << "You must a image path" << endl;
    return -1;
  }

  h_src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if (!h_src.data) { 
    cout << "Error reading the image. Ensure it exist" << endl;
    return -1; 
  }

  namedWindow("Sobel with opencv and cuda", CV_WINDOW_NORMAL);
  
  gpu::GpuMat d_src, d_grad_x, d_grad_y, d_sobel;
  d_src.upload(h_src);

  gpu::Sobel(d_src, d_grad_x, ddepth, 1, 0);
  gpu::Sobel(d_src, d_grad_y, ddepth, 0, 1);

  gpu::magnitude(d_grad_x, d_grad_y, d_sobel);
  
  d_sobel.download(h_sobel);
  convertScaleAbs(h_sobel, h_sobel);
    
  imshow("Sobel with opencv and cuda", h_sobel);

  waitKey(0);

  return 0;
  }
