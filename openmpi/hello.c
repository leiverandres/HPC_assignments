#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, int* argv[]) {
  int pid = -1, np = -1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid); // Get the rank of process
  MPI_Comm_size(MPI_COMM_WORLD, &np); // Get Number of processes

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Hello from from processor %s, rank %d out of %d processors\n",
         processor_name, pid, np);

  MPI_Finalize();
  return 0;  
}
