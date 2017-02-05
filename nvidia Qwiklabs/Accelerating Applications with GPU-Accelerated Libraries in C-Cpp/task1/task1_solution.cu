#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "timer.h"

#include <stdio.h>

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

#define SIZE 1024

// A single-threaded version of matrix multiply
void host_sgemm( int m, int n, int k, float *a, float *b, float *c )
{
  for( int j = 0; j < n; j++ )
  {
    for( int i = 0; i < m; i++ )
    {
      for( int koff = 0; koff < k; koff++ )
      {
        c[INDX(i, j, m)] += a[INDX( i, koff, m )] * b[INDX( koff, j, n )];
      } /* end for i */
    } /* end jb */
  } /* end for j */
} /* end host_sgemm */

int main()
{
    const int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

    float *h_a, *h_b, *h_c, *h_cdef;
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

    h_cdef = (float *) malloc( numbytes );
    if( h_cdef == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    // Clear the result matrices to zero
    memset( h_c, 0, numbytes );
    memset( h_cdef, 0, numbytes );

    // Initialize the a and b matrices to random data
    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
      h_b[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
    }

    // Allocate device-side (GPU) memory
    cudaMalloc( (void **)&d_a, numbytes );
    cudaMalloc( (void **)&d_b, numbytes );
    cudaMalloc( (void **)&d_c, numbytes );

    // First run the CPU verison of dgemm so we can compare the results
    StartTimer();
  
    printf("Launching CPU sgemm\n");
    host_sgemm( size, size, size, h_a, h_b, h_cdef );

    double runtime = GetTimer();

    fprintf(stdout, "Total time CPU is %f sec\n", runtime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) runtime / 1000.0 ) * 1.e-9 );

    // Now run the GPU version of sgemm using the cuBLAS library
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    // Set these constants so we get a simple matrix multiply with cublasDgemm
    float alpha = 1.0;
    float beta  = 0.0;

    StartTimer();

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

    runtime = GetTimer();

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", runtime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) runtime / 1000.0 ) * 1.e-9 );

    cublasDestroy( handle );

    // Do some error checking to verify our GPU & CPU verisons are within
    // an acceptable error bound
    float temp = 0.0;
    for( int i = 0; i < size * size; i++ )
    {
        temp += ( h_c[i] - h_cdef[i] ) * ( h_c[i] - h_cdef[i] );
    } /* end for */
  
    printf("error is %f\n",temp);
    if( temp > 10 ) printf("Error value is suspiciously high!\n");

    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_cdef );

    return 0;
}
