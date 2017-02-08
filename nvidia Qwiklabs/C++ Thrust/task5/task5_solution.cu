#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include <thrust/sort.h>
#include <iostream>

int main(void)
{
  // allocate some device vectors
  thrust::device_vector<int> vecA(100);
  thrust::device_vector<int> vecB;

  try
  {
    vecB.resize(1 << 30);
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Couldn't allocate vecB" << std::endl;
    exit(-1);
  }
  
  try
  {
    vecA[100] = 50;
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "Error occured during assignment: " << e.what() << std::endl;
    exit(-1);
  }

  // sort vecB then copy to vecA
  try
  {
    thrust::sort(vecB.begin(), vecB.end());
    thrust::copy(vecB.begin(), vecB.end(), vecA.begin());
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Ran out of memory while sorting" << std::endl;
    exit(-1);
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "Some other error happened during sort & copy: " << e.what() << std::endl;
    exit(-1);
  }
  
  std::cout << "Finished running program" << std::endl;

  return 0;
}