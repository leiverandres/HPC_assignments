#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "timer.h"

int main()
{
  thrust::host_vector<int> h_vec( 500000000 );
  
  thrust::generate( h_vec.begin(), h_vec.end(), rand );
  
  StartTimer();
  
  thrust::device_vector<int> d_vec = h_vec;
  
  thrust::sort( d_vec.begin(), d_vec.end() );
  
  thrust::copy( d_vec.begin(), d_vec.end(), h_vec.begin() );
  
  double runtime = GetTimer();
  printf("Runtime: %f s\n", runtime / 1000);
  
  return 0;
}