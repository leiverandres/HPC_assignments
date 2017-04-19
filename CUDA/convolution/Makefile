CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main: seq cuda cv cv-cuda

seq : seq_convolution.cpp
	g++ $(CFLAGS) seq_convolution.cpp $(LIBS) -o seq_convolution.out

cuda: cuda_convolution.cu
	nvcc $(CFLAGS) cuda_convolution.cu $(LIBS) -o cuda_convolution.out

cv: with_opencv.cpp
	g++ $(CFLAGS) with_opencv.cpp $(LIBS) -o with_opencv.out

cv-cuda: with_opencv_cuda.cpp 
	g++ $(CFLAGS) with_opencv_cuda.cpp $(LIBS) -o with_opencv_cuda.out
