#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.hpp"

void SeqMultiplication(float *A, float *B, float *C, int A_ROWS, int A_COLS, int B_COLS) {
  for (int i = 0; i < A_ROWS; i++) {
    for (int j = 0; j < B_COLS; j++) {
      float sum = 0.0;
      for (int k = 0; k < A_COLS; k++) {
        sum += A[i * A_COLS + k] * B[k * B_COLS + j]; 
      }
      C[i * B_COLS + j] = sum;
    }
  }
}

void PrintUsage(string program) {
  cout << "Usage: " << program << " A_ROWS A_COLS B_COLS" << endl;
  cout << "* Is not needed to pass B_ROWS because B_ROWS must be equal to A_COLS" << endl;
  cout << endl;
}

int main(int argc, char **argv) {
  int A_ROWS, A_COLS, B_ROWS, B_COLS;
  if (argc < 4) {
    PrintUsage(argv[0]);
    return -1;
  } else {
    A_ROWS = atoi(argv[1]);
    A_COLS = atoi(argv[2]);
    B_ROWS = A_COLS;
    B_COLS = atoi(argv[3]);
  }
  clock_t start, end;
  double time_used;
  
  size_t size_a = A_ROWS * A_COLS * sizeof(float);
  size_t size_b = B_ROWS * B_COLS * sizeof(float);
  size_t size_c = A_ROWS * B_COLS * sizeof(float);

  float *a = (float *) malloc(size_a);
  float *b = (float *) malloc(size_b);
  float *c = (float *) malloc(size_c);  

  utils::InitMat(a, A_ROWS, A_COLS);
  utils::InitMat(b, B_ROWS, B_COLS); 

  start = clock();

  SeqMultiplication(a, b, c, A_ROWS, A_COLS, B_COLS);

  end = clock();

  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("%.10f ",time_used);  // time in CPU
  
  free(a); free(b); free(c);
  return 0;
}
