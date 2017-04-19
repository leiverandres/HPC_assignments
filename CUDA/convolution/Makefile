GS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

seq : seq_convolution.cpp
	g++ $(CFLAGS) seq_convolution.cpp $(LIBS) -o seq_convolution.out

cv: with_opencv.cpp
	g++ $(CFLAGS) with_opencv.cpp $(LIBS) -o with_opencv.out
