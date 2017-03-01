#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define N 15
#define MASTER 0
#define ADD_TAG 0
#define RESULT_TAG 1

void seqSum(int *a, int *b, int *c) {
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];
}

void fillRandomVec(int *x, int len) {
  for (int i = 0; i < len; i++)
    x[i] = rand() % 50;
}

void showVec(int *x, int len) {
  for (int i = 0; i < len; i++)
    printf("%d ", x[i]);
  printf("\n");
}

int main() {
  MPI_Init(NULL, NULL);
  int world_size = -1, world_rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int chunk_size = N / (world_size - 1);
  int offset = N % (world_size - 1);
  
  if (world_rank == MASTER) {
    printf("There are %d processes\n", world_size);
    printf("The chunk size is %d\n", chunk_size);
    printf("The offset is %d\n", offset);
    int *a = (int *) malloc(N * sizeof(int));
    int *b = (int *) malloc(N * sizeof(int));
    int *c = (int *) malloc(N * sizeof(int));
    fillRandomVec(a, N);
    fillRandomVec(b, N);
    showVec(a, N);
    showVec(b, N);
    for (int i = 1; i < world_size; i++) {
      if (offset > 0 && i == (world_size - 1)) {
        MPI_Send(a + (i - 1) * chunk_size, chunk_size + offset, MPI_INT, i, ADD_TAG, MPI_COMM_WORLD);
        MPI_Send(b + (i - 1) * chunk_size, chunk_size + offset, MPI_INT, i, ADD_TAG, MPI_COMM_WORLD);
      } else {
        MPI_Send(a + (i - 1) * chunk_size, chunk_size, MPI_INT, i, ADD_TAG, MPI_COMM_WORLD);
        MPI_Send(b + (i - 1) * chunk_size, chunk_size, MPI_INT, i, ADD_TAG, MPI_COMM_WORLD);
      }
    }  

    for (int i = 1; i < world_size; i++) {
      if (offset > 0 && i == (world_size - 1)) {
        MPI_Recv(c + (i - 1) * chunk_size, chunk_size + offset, MPI_INT, i, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else { 
        MPI_Recv(c + (i - 1) * chunk_size, chunk_size, MPI_INT, i, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    showVec(c, N);
    free(a);
    free(b);
    free(c);
  } else {
    int *a_proc, *b_proc, *c_proc;
    if (offset > 0 && world_rank == (world_size - 1)) { 
      a_proc = (int *) malloc((chunk_size + offset) * sizeof(int));
      b_proc = (int *) malloc((chunk_size + offset) * sizeof(int));
      c_proc = (int *) malloc((chunk_size + offset) * sizeof(int));
      MPI_Status *status;
      MPI_Recv(a_proc, chunk_size + offset, MPI_INT, MASTER, ADD_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(b_proc, chunk_size + offset, MPI_INT, MASTER, ADD_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //printf("I'm %d proc and i got:\n", world_rank);
      //showVec(a_proc, chunk_size + offset);
      //showVec(b_proc, chunk_size + offset);
      for (int i = 0; i < chunk_size + offset; i++)
        c_proc[i] = a_proc[i] + b_proc[i];
    
      MPI_Send(c_proc, chunk_size + offset, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD);
    } else {
      a_proc = (int *) malloc(chunk_size * sizeof(int));
      b_proc = (int *) malloc(chunk_size * sizeof(int));
      c_proc = (int *) malloc(chunk_size * sizeof(int));
      MPI_Recv(a_proc, chunk_size, MPI_INT, MASTER, ADD_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(b_proc, chunk_size, MPI_INT, MASTER, ADD_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //printf("I'm %d proc and i got:\n", world_rank);
      //showVec(a_proc, chunk_size);
      //showVec(b_proc, chunk_size);
      for (int i = 0; i < chunk_size; i++)
        c_proc[i] = a_proc[i] + b_proc[i];
    
      MPI_Send(c_proc, chunk_size, MPI_INT, MASTER, RESULT_TAG, MPI_COMM_WORLD);
    }
    free(a_proc);
    free(b_proc);
    free(c_proc);
  }

  // seqSum(a, b, c);
  return 0;
}
