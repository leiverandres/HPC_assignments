#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0
#define ROWA 5
#define COLA 5
#define ROWB COLA
#define COLB 4
#define MULT_TAG 0
#define RESULT_TAG 1

void fillRandMat(int *x, int row_size, int col_size) {
  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      x[j + i * col_size] = rand() % 50;
    }
  }
}

void showMat(int *x, int row_size, int col_size) {
  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      printf("%d ", x[j + i * col_size]);
    }
    printf("\n");
  }
  printf("\n");
}

void trans(int *x, int *result, int row_size, int col_size) {
  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      result[i + j * col_size] = x[j + i * row_size];   
    }
  }
}

void matrixVecMult(int *A, int *v, int *r) {
  for (int i = 0; i < ROWA; i++) {
    r[i] = 0;
    for (int j = 0; j < COLA; j++) {
      r[i] += A[i * COLA + j] * v[j];
    }
  }
}

int main() {
  int world_rank = -1, world_size = -1;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int chunk_cols = COLB / (world_size - 1);
  int offset = COLB % (world_size - 1);

  if (world_rank == MASTER) {
    int *mat_a = (int *) malloc(ROWA * COLA * sizeof(int));
    int *mat_b = (int *) malloc(ROWB * COLB * sizeof(int));
    int *mat_c = (int *) malloc(ROWA * COLB * sizeof(int));
    int *trans_b = (int *) malloc(COLB * ROWB * sizeof(int));
    int *trans_c = (int *) malloc(COLB * ROWA * sizeof(int));
    fillRandMat(mat_a, ROWA, COLA);
    fillRandMat(mat_b, ROWB, COLB);
    trans(mat_b, trans_b, ROWB, COLB);
    showMat(mat_a, ROWA, COLA);
    showMat(mat_b, ROWB, COLB);
    for (int i = 1; i < world_size; i++) {
      MPI_Send(mat_a, ROWA * COLA, MPI_INT, i, MULT_TAG, MPI_COMM_WORLD);
      MPI_Send(trans_b + (i - 1) * (chunk_cols * ROWB), chunk_cols * ROWB, MPI_INT, i, MULT_TAG, MPI_COMM_WORLD);
    }

    for (int i = 1; i < world_size; i++) {
      MPI_Recv(trans_c + (i - 1) * (chunk_cols * ROWA), chunk_cols * ROWA, MPI_INT, i, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (offset) {
      for (int i = 0; i < offset; i++) {
        matrixVecMult(mat_a, trans_b + (COLB - offset + i) * COLA, trans_c + (COLB - offset + i) * ROWA);
      }
    }

    trans(trans_c, mat_c, COLB, ROWA);
    showMat(mat_c, ROWA, COLB);
    free(mat_a);
    free(mat_b);
    free(mat_c);
    free(trans_b);
  } else {
    int *mat_a_proc = (int *) malloc(ROWA * COLA * sizeof(int));
    int *cols = (int *) malloc(chunk_cols * ROWB * sizeof(int));
    int *result = (int *) malloc(ROWA * chunk_cols * sizeof(int));
    MPI_Recv(mat_a_proc, ROWA * COLA, MPI_INT, MASTER, MULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(cols, chunk_cols * ROWB, MPI_INT, MASTER, MULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < chunk_cols; i++) {
      matrixVecMult(mat_a_proc, cols + i * COLA, result + i * ROWA);
    }
    MPI_Send(result, chunk_cols * ROWA, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD); 
    free(mat_a_proc);
    free(cols);
    free(result);
  }
  return 0;
}
