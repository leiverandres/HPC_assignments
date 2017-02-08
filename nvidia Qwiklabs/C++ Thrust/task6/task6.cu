#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
  thrust::device_vector<int> x(1000);

  std::cerr << "Before transform." << std::endl;

  // transform into a bogus location
  thrust::transform(x.begin(), x.end(), thrust::device_pointer_cast<int>(0), thrust::negate<int>());

  std::cerr << "After transform." << std::endl;

  return 0;
}