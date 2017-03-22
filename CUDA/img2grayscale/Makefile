GS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : img2grayscale.cu
	nvcc $(CFLAGS) img2grayscale.cu $(LIBS) -o img2grayscale.out
