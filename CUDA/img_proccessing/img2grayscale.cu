#include <iostream>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: img2grayscale.out <image_name>" << endl;
    return -1;
  }
  Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  uchar *img_ptr = img.data;
  
  imshow("Image in grayscale", img);
  waitKey(0);
  return 0;
}

