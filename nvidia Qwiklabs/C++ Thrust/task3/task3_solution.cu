#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

template <typename T>
struct modZeroOrTen
{
  __host__ __device__
  T operator()(const T& x, const T& y) const
  {
    return ((x % y) == 0)?0:10;
  }
};

int main(void)
{
  // Allocate three device_vectors with 10 elements
  thrust::device_vector<int> X(10);
  thrust::device_vector<int> Y(10);
  thrust::device_vector<int> Z(10);

  // Initialize X to 0,1,2,3,... using thrust::sequence
  thrust::sequence(X.begin(), X.end());

  // Fill Z with twos with thrust::fill
  thrust::fill(Z.begin(), Z.end(), 2);

  // Compute Y = (X MOD 2)== 0 ? 1 : 10 with thrust::transform and a custom functor
  thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), modZeroOrTen<int>());

  // Print Y using the thrust::copy and the following functor:
  // std::ostream_iterator<int>(std::cout, "\n")
  thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));
   
  return 0;    
}