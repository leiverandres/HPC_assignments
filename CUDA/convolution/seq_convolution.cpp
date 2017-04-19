#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void conv(Mat img, Mat result, Mat kernel, int height, int width, 
          int kernel_size) {
  int offset = kernel_size / 2;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float acum = 0.0;
      int start_i = i - offset;
      int start_j = j - offset;
      for (int k = 0; k < kernel_size; k++) {
        for (int l = 0; l < kernel_size; l++) {
          int cur_row = start_i + k;
          int cur_col = start_j + l;
          if (cur_row >= 0 && cur_col >= 0 && 
              cur_row < height && cur_col < width) {
            acum += img.at<uchar>(cur_row, cur_col) * 
                    kernel.at<float>(k, l);
          }
        }
      }
      result.at<float>(i, j) = acum;
    }
  }
}

void gradientMagnitude(Mat x_derivate, Mat y_derivate, Mat result, 
                      int height, int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float x = x_derivate.at<float>(i, j);
      float y = y_derivate.at<float>(i, j);
      result.at<uchar>(i, j) = (uchar) sqrt((x * x) + (y * y));
    }
  }                      
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "You must provide a image name" << endl;
    return -1;
  }

  Mat img = imread(argv[1], 0);
  
  int width = img.cols;
  int height = img.rows;
  
  Mat sobel_x = Mat::zeros(height, width, CV_32F);
  Mat sobel_y = Mat::zeros(height, width, CV_32F);
  Mat sobel = Mat::zeros(height, width, CV_8UC1);
  
  float kernel_x_data[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float kernel_y_data[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  int kernel_size = 3;
  
  Mat kernel_x(kernel_size, kernel_size, CV_32F, kernel_x_data);
  Mat kernel_y(kernel_size, kernel_size, CV_32F, kernel_y_data);
  
  conv(img, sobel_x, kernel_x, height, width, kernel_size);
  conv(img, sobel_y, kernel_y, height, width, kernel_size);
  gradientMagnitude(sobel_x, sobel_y, sobel, height, width);
  
  convertScaleAbs(sobel, sobel);
  
  namedWindow("Original image", WINDOW_NORMAL);
  namedWindow("Filtered image", WINDOW_NORMAL);
  imshow("Original image", img);
  imshow("Filtered image", sobel);
  waitKey(0);
  return 0;
}
