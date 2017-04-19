#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv ) {

  Mat src;
  Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_32F;

  int c;

  src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if( !src.data ) { 
    return -1; 
  }

  namedWindow("Sobel with opencv", CV_WINDOW_NORMAL);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

  Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  magnitude(grad_x, grad_y, grad);
  
  convertScaleAbs(grad, grad);

  imshow("Sobel with opencv", grad );

  waitKey(0);

  return 0;
  }
