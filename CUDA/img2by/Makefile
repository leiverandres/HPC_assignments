FLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : imgby2.cu
		nvcc $(CFLAGS) imgby2.cu $(LIBS) -o imgby2.out
