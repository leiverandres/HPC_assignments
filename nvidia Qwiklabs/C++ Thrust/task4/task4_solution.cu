#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>

void constantIterator()
{
  std::cout << "Constant Iterator:\n";

  thrust::constant_iterator<int> first(5);
  thrust::constant_iterator<int> last = first + 5;

  // sum of [first, last)
  int value = thrust::reduce(first, last);
  
  std::cout << "  value = " << value << ". Should be 20\n\n";
}

void transformIterator()
{
  std::cout << "Transform Iterator:\n";

  thrust::device_vector<int> d_vec(3);
  thrust::sequence( d_vec.begin(), d_vec.end(), 10, 10 ); // d_vec[0] = 10, d_vec[1] = 20, ...

  // sum of [first, last)
  int value = thrust::reduce(thrust::make_transform_iterator(d_vec.begin(), thrust::negate<int>()),
                             thrust::make_transform_iterator(d_vec.end(),   thrust::negate<int>()));

  std::cout << "  value = " << value << ". Should be -60\n\n";
}

void zipIterator()
{
  std::cout << "Zip Iterator:\n";
  
  thrust::device_vector<int>  A(4);
  thrust::device_vector<char> B(4);
  A[0] = 10;  A[1] = 20;  A[2] = 30;  A[3] = 30;
  B[0] = 'x'; B[1] = 'y'; B[2] = 'z'; B[3] = 'a';
  

  // maximum of [first, last)
  thrust::maximum< thrust::tuple<int,char> > binary_op;

  thrust::tuple<int,char> init;
  init = thrust::make_tuple(-1*INT_MAX, 'a'); // Speficy the initial value for the maximize reduction
  										      // This is the lowest possible value we could have

  thrust::tuple<int,char> value;
  value = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin())), 
                         thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end())),
                         init, 
                         binary_op);
  
  std::cout << "  value = (" << value.get<0>() << "," 
    			  	         << value.get<1>() << "). Should be (30,z)\n\n";
}

// No need to modify the main() function
int main(void)
{
  constantIterator();
  
  transformIterator();
  
  zipIterator();
   
  return 0;    
}