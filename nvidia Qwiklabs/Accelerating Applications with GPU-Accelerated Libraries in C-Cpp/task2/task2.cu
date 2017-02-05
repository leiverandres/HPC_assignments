#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
#include "timer.h"

#include <stdio.h>

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

#define SIZE 10000

int main()
{
    const int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
 
    size_t numbytes = size * size * sizeof( float );

    // Allocate all our host-side (CPU) data
    h_a = (float *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_b = (float *) malloc( numbytes );
    if( h_b == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_c = (float *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    // Set these constants so we get a simple matrix multiply with cublasSgemm
    float alpha = 1.0;
    float beta  = 0.0;
  
    StartTimer();
  
    // Allocate device-side (GPU) memory
    cudaMalloc( (void **)&d_a, numbytes );
    cudaMalloc( (void **)&d_b, numbytes );
    cudaMalloc( (void **)&d_c, numbytes );

    // Generate size * size random numbers
    printf("Create random numbers\n");
    // FIXME: Replace the following for-loop with two curandGenerateNormal calls

    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateNormal(gen, h_a, size*size, 0.0, float(RAND_MAX));    
    curandGenerateNormal(gen, h_b, size*size, 0.0, float(RAND_MAX));


  	// Copy the a and b matrices to the GPU memory using cudaMemcpy
    cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice );

    // Launch cublasSgemm on the GPU
    printf("Launching GPU sgemm\n");
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha, 
                 d_a, size,
                 d_b, size,
                 &beta,
                 d_c, size );

    // Finally, copy the resulting c array back to the host  
    cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost );

    double runtime = GetTimer();

    fprintf(stdout, "Total time is %f sec\n", runtime / 1000.0f );

    cublasDestroy( handle );
	curandDestroyGenerator(gen);	
  
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );

    free( h_a );
    free( h_b );
    free( h_c );

    return 0;
}
