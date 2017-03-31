#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>


using namespace std;

namespace utils {
void InitMat(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = rand() % 50;
  }
}

void ShowMat(string name, float *mat, int rows, int cols) {
  cout << name << " = [";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cout << mat[i * cols + j] << " ";
    }
    cout << ";" << endl;
  }
  cout << "]" << endl;
}

bool CheckResults(float *x, float *y, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (x[i * cols + j] != y[i * cols + j]) {
        return false;
      }
    }
  }
  return true;
}
};

#endif
